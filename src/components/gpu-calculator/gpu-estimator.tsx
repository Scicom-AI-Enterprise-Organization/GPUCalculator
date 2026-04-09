"use client";

import { useState, useMemo, useCallback } from "react";
import { Loader2, CheckCircle2, XCircle, Link as LinkIcon, Database } from "lucide-react";
import type { BenchmarkData, BenchmarkPoint } from "@/lib/read-benchmarks";

// vram in GB, tflops = FP16 dense TFLOPS, bw = memory bandwidth in GB/s
const GPU_SPECS: Record<string, { vram: number; tflops: number; bw: number; label: string }> = {
  b200:     { vram: 192, tflops: 1800, bw: 8000, label: "B200 (192 GB)" },
  h200:     { vram: 141, tflops: 989,  bw: 4800, label: "H200 SXM (141 GB)" },
  h100:     { vram: 80,  tflops: 989,  bw: 3350, label: "H100 SXM (80 GB)" },
  a100_80:  { vram: 80,  tflops: 312,  bw: 2039, label: "A100 SXM 80GB" },
  a100_40:  { vram: 40,  tflops: 312,  bw: 1555, label: "A100 PCIe 40GB" },
  l40s:     { vram: 48,  tflops: 362,  bw: 864,  label: "L40S (48 GB)" },
  a10:      { vram: 24,  tflops: 125,  bw: 600,  label: "A10 (24 GB)" },
  rtx_4090: { vram: 24,  tflops: 330,  bw: 1008, label: "RTX 4090 (24 GB)" },
};

const PRECISION: Record<string, { bytesPerParam: number; label: string }> = {
  fp32: { bytesPerParam: 4, label: "FP32" },
  bf16: { bytesPerParam: 2, label: "BF16 / FP16" },
  fp8:  { bytesPerParam: 1, label: "FP8" },
  int8: { bytesPerParam: 1, label: "INT8" },
  int4: { bytesPerParam: 0.5, label: "INT4 / GPTQ / AWQ" },
};

const KV_PRECISION: Record<string, { bytes: number; label: string }> = {
  bf16: { bytes: 2, label: "BF16 / FP16" },
  fp8:  { bytes: 1, label: "FP8" },
};

// Log-scale context length: slider 0–100 maps to 1024–131072
const CTX_MIN = 1024;
const CTX_MAX = 131072;
const CTX_LOG_MIN = Math.log2(CTX_MIN);
const CTX_LOG_MAX = Math.log2(CTX_MAX);
function ctxSliderToValue(slider: number): number {
  const log = CTX_LOG_MIN + (slider / 100) * (CTX_LOG_MAX - CTX_LOG_MIN);
  return Math.round(Math.pow(2, log));
}
function ctxValueToSlider(value: number): number {
  const log = Math.log2(Math.max(CTX_MIN, Math.min(CTX_MAX, value)));
  return Math.round(((log - CTX_LOG_MIN) / (CTX_LOG_MAX - CTX_LOG_MIN)) * 100);
}
function formatCtx(value: number): string {
  if (value >= 1024) return `${(value / 1024).toFixed(value % 1024 === 0 ? 0 : 1)}K`;
  return String(value);
}

interface MoeConfig {
  numExperts: number;
  activeExperts: number;
  sharedExperts: number;
}

interface ModelConfig {
  modelName: string;
  numLayers: number;
  hiddenSize: number;
  numAttentionHeads: number;
  numKvHeads: number;
  headDim: number;
  intermediateSize: number;
  vocabSize: number;
  moe: MoeConfig | null;
}

interface InterpolatedEstimate {
  throughputPerGpu: number;
  ttftMs: number;
  e2eMs: number;
  confidence: "interpolated" | "extrapolated" | "nearest";
  basedOn: string; // description of what data was used
}

interface EstimationResult {
  modelVram: number;  // per GPU
  kvCacheVram: number;  // per GPU
  overheadVram: number;  // per GPU
  totalVram: number;  // per GPU
  gpuVram: number;
  numGpus: number;
  tp: number;
  dp: number;
  tpRecommendation: string;
  utilizationPercent: number;
  kvPerTokenBytes: number;
  usingHfConfig: boolean;
  activeParamsB: number;
  isMoe: boolean;
  ttftMs: number;
  tpotMs: number;
  e2eMs: number;
}

function resolveHfUrl(url: string): string {
  let resolved = url.trim();
  resolved = resolved.replace("/blob/", "/resolve/");

  if (!resolved.includes("config.json")) {
    if (!resolved.endsWith("/")) resolved += "/";
    resolved += "resolve/main/config.json";
  }

  if (!resolved.includes("/resolve/")) {
    resolved = resolved.replace(
      /huggingface\.co\/([^/]+\/[^/]+)\/?$/,
      "huggingface.co/$1/resolve/main/config.json"
    );
  }

  return resolved;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function extractConfig(json: any): ModelConfig {
  const cfg = json.text_config || json;

  const numAttentionHeads = cfg.num_attention_heads ?? cfg.n_head ?? 32;
  const hiddenSize = cfg.hidden_size ?? cfg.n_embd ?? 4096;
  const headDim = cfg.head_dim ?? Math.floor(hiddenSize / numAttentionHeads);

  let modelName = cfg.model_type || "";
  if (json.architectures?.length) {
    modelName = json.architectures[0].replace(/ForCausalLM$/, "");
  }
  if (json._name_or_path) {
    modelName = json._name_or_path;
  }

  let moe: MoeConfig | null = null;
  const numExperts = cfg.num_local_experts ?? cfg.n_routed_experts ?? cfg.num_experts ?? null;
  if (numExperts && numExperts > 1) {
    moe = {
      numExperts,
      activeExperts: cfg.num_experts_per_tok ?? cfg.num_selected_experts ?? cfg.top_k ?? 2,
      sharedExperts: cfg.n_shared_experts ?? cfg.num_shared_experts ?? 0,
    };
  }

  return {
    modelName,
    numLayers: cfg.num_hidden_layers ?? cfg.n_layer ?? 32,
    hiddenSize,
    numAttentionHeads,
    numKvHeads: cfg.num_key_value_heads ?? numAttentionHeads,
    headDim,
    intermediateSize: cfg.intermediate_size ?? cfg.n_inner ?? hiddenSize * 4,
    vocabSize: cfg.vocab_size ?? 32000,
    moe,
  };
}

function computeActiveRatio(
  isMoe: boolean,
  numExperts: number,
  activeExperts: number,
  sharedExperts: number
): number {
  if (!isMoe || numExperts <= 1) return 1;
  const sharedFraction = 0.35;
  const expertFraction = 1 - sharedFraction;
  const activeExpertRatio = Math.min((activeExperts + sharedExperts) / numExperts, 1);
  return sharedFraction + expertFraction * activeExpertRatio;
}

function estimateGpus(
  paramsB: number,
  quant: string,
  kvPrecision: string,
  contextLength: number,
  gpuType: string,
  concurrentRequests: number,
  inputTokens: number,
  outputTokens: number,
  hfConfig: ModelConfig | null,
  isMoe: boolean,
  moeNumExperts: number,
  moeActiveExperts: number,
  moeSharedExperts: number
): EstimationResult {
  const q = PRECISION[quant];
  const kv = KV_PRECISION[kvPrecision];
  const gpu = GPU_SPECS[gpuType];

  const modelVramTotal = paramsB * q.bytesPerParam; // GB, total model weights

  const activeRatio = computeActiveRatio(isMoe, moeNumExperts, moeActiveExperts, moeSharedExperts);
  const activeParamsB = paramsB * activeRatio;

  let kvPerTokenBytes: number;
  if (hfConfig) {
    kvPerTokenBytes = 2 * hfConfig.numLayers * hfConfig.numKvHeads * hfConfig.headDim * kv.bytes;
  } else {
    kvPerTokenBytes = paramsB * 0.00003 * 1e9 * (kv.bytes / 2);
  }
  const kvCacheVramTotal = (kvPerTokenBytes * contextLength * concurrentRequests) / 1e9;

  const overheadVramTotal = activeParamsB * q.bytesPerParam * 0.1;

  // Per-GPU VRAM with TP/DP:
  // - Model weights: sharded across TP, replicated across DP → per GPU = modelVram / tp
  // - KV cache: split across DP shards, sharded across TP → per GPU = kvCache / (tp * dp)
  // - Overhead: sharded across TP → per GPU = overhead / tp
  // Find minimum (tp, dp) where per-GPU VRAM fits
  // Maximize TP (power of 2), then find minimum DP

  let bestTp = 1;
  let bestDp = 1;
  let numGpus = 1;

  const fitsOnGpu = (tp: number, dp: number) => {
    const perGpu = modelVramTotal / tp
      + kvCacheVramTotal / (tp * dp)
      + overheadVramTotal / tp;
    return perGpu <= gpu.vram;
  };

  if (fitsOnGpu(1, 1)) {
    bestTp = 1;
    bestDp = 1;
    numGpus = 1;
  } else {
    // Try TP values from highest to lowest, find min DP for each
    const tpCandidates = [8, 4, 2];
    let found = false;
    for (const tp of tpCandidates) {
      // Find minimum DP that fits
      for (let dp = 1; dp <= 64; dp++) {
        if (fitsOnGpu(tp, dp)) {
          bestTp = tp;
          bestDp = dp;
          numGpus = tp * dp;
          found = true;
          break;
        }
      }
      if (found) break;
    }
    if (!found) {
      // Fallback: TP8 with large DP
      bestTp = 8;
      bestDp = Math.ceil(kvCacheVramTotal / (8 * gpu.vram)) + 1;
      numGpus = bestTp * bestDp;
    }
  }

  // Compute per-GPU breakdown for display
  const modelVram = modelVramTotal / bestTp;
  const kvCacheVram = kvCacheVramTotal / (bestTp * bestDp);
  const overheadVram = overheadVramTotal / bestTp;
  const totalVram = modelVram + kvCacheVram + overheadVram;

  let tpRecommendation: string;
  if (numGpus === 1) {
    tpRecommendation = "No parallelism needed";
  } else if (bestDp === 1) {
    tpRecommendation = `TP${bestTp}`;
  } else {
    tpRecommendation = `TP${bestTp}/DP${bestDp}`;
    if (numGpus > 8) tpRecommendation += " (multi-node)";
  }

  const utilizationPercent = (totalVram / gpu.vram) * 100;

  // --- Latency estimation ---
  // All estimates are for a single request under concurrent load

  // TTFT (prefill phase — compute bound)
  // FLOPs per token ≈ 2 * active_params (linear layers) + 2 * n_layers * seq_len * hidden_size (attention)
  // Simplified: attention FLOPs scale quadratically with input length
  const prefillUtilization = 0.35;
  const linearFlops = 2 * activeParamsB * 1e9 * inputTokens;
  // Attention FLOPs: 2 * num_layers * input_tokens^2 * hidden_size (Q*K + attn*V)
  const numLayers = hfConfig ? hfConfig.numLayers : Math.round(paramsB * 0.6); // rough estimate
  const hiddenSize = hfConfig ? hfConfig.hiddenSize : Math.round(Math.sqrt(paramsB * 1e9 / numLayers / 12));
  const attentionFlops = 2 * numLayers * inputTokens * inputTokens * hiddenSize;
  const prefillFlops = linearFlops + attentionFlops;
  // With TP, only TP GPUs in one shard do compute; DP doesn't help single-request latency
  const tpComputeFlops = bestTp * gpu.tflops * 1e12 * prefillUtilization;
  // Under concurrent load, prefill requests share compute; effective throughput is divided
  const concurrentPrefillFactor = 1 + Math.log2(Math.max(1, concurrentRequests / bestDp));
  const ttftMs = (prefillFlops / tpComputeFlops) * concurrentPrefillFactor * 1000;

  // TPOT (decode phase — memory bandwidth bound)
  // Each decode step reads: model weights + KV cache for current sequence
  const activeModelBytes = activeParamsB * q.bytesPerParam * 1e9;
  // KV cache bytes read per decode step grows with sequence length (input + generated so far)
  // Average sequence position during decode ≈ inputTokens + outputTokens/2
  const avgSeqLen = inputTokens + outputTokens / 2;
  const kvReadBytes = kvPerTokenBytes * avgSeqLen;
  const totalReadPerStep = activeModelBytes / bestTp + kvReadBytes;
  const tpBandwidth = bestTp * gpu.bw * 1e9; // bytes/s
  const decodeUtilization = 0.65;
  // Under concurrent load, bandwidth is shared across concurrent decodes in the DP shard
  const concurrentDecodeFactor = Math.max(1, concurrentRequests / bestDp);
  const tpotMs = (totalReadPerStep / (tpBandwidth * decodeUtilization)) * concurrentDecodeFactor * 1000;

  // E2E latency for a single request
  const e2eMs = ttftMs + outputTokens * tpotMs;

  return {
    modelVram,
    kvCacheVram,
    overheadVram,
    totalVram,
    gpuVram: gpu.vram,
    numGpus,
    tp: bestTp,
    dp: bestDp,
    tpRecommendation,
    utilizationPercent,
    kvPerTokenBytes,
    usingHfConfig: hfConfig !== null,
    activeParamsB,
    isMoe,
    ttftMs,
    tpotMs,
    e2eMs,
  };
}

// Map estimator GPU keys to benchmark GPU labels
const GPU_KEY_TO_BENCHMARK: Record<string, string> = {
  b200: "B200",
  h200: "H200 SXM",
  h100: "H100 SXM",
  a100_80: "A100 SXM",
  a100_40: "A100 SXM",
  l40s: "",
  a10: "",
  rtx_4090: "",
};

// Benchmark model metadata: total params, active params, and architecture type
const BENCHMARK_MODELS: Record<string, { totalB: number; activeB: number; isMoe: boolean; precision?: string }> = {
  "gpt-oss-120b":   { totalB: 120, activeB: 5.1, isMoe: true },
  "qwen3-32b":      { totalB: 32,  activeB: 32,  isMoe: false },
  "qwen3-14b":      { totalB: 14,  activeB: 14,  isMoe: false },
  "qwen3-8b":       { totalB: 8,   activeB: 8,   isMoe: false },
  "qwen3.5-27b":    { totalB: 27,  activeB: 27,  isMoe: false },
  "qwen3.5-35b":    { totalB: 35,  activeB: 3,   isMoe: true },
  "qwen3.5-122b":   { totalB: 122, activeB: 10,  isMoe: true },
  "llama3.1-70b":   { totalB: 70,  activeB: 70,  isMoe: false },
  "glm-4.7":        { totalB: 358, activeB: 32,  isMoe: true },
  "glm-4.7-fp8":    { totalB: 358, activeB: 32,  isMoe: true, precision: "fp8" },
};

// Linear interpolation
function lerp(x: number, x0: number, x1: number, y0: number, y1: number): number {
  if (x1 === x0) return y0;
  return y0 + (y1 - y0) * ((x - x0) / (x1 - x0));
}

// Log-linear interpolation (better for throughput/latency scaling)
function logLerp(x: number, x0: number, x1: number, y0: number, y1: number): number {
  if (x1 === x0 || y0 <= 0 || y1 <= 0) return lerp(x, x0, x1, y0, y1);
  const logY = lerp(Math.log(x), Math.log(x0), Math.log(x1), Math.log(y0), Math.log(y1));
  return Math.exp(logY);
}

// Interpolate a metric across context lengths for a given set of points (same model+gpu+config)
function interpolateByCtx(
  points: BenchmarkPoint[],
  targetCtx: number,
  metric: (p: BenchmarkPoint) => number
): { value: number; confidence: "interpolated" | "extrapolated" | "nearest" } | null {
  if (points.length === 0) return null;

  const sorted = [...points].sort((a, b) => a.ctx - b.ctx);
  const ctxValues = sorted.map((p) => p.ctx);

  // Exact match
  const exact = sorted.find((p) => p.ctx === targetCtx);
  if (exact) return { value: metric(exact), confidence: "interpolated" };

  // Single point — nearest
  if (sorted.length === 1) return { value: metric(sorted[0]), confidence: "nearest" };

  // Within range — interpolate
  if (targetCtx >= ctxValues[0] && targetCtx <= ctxValues[ctxValues.length - 1]) {
    // Find surrounding points
    let lo = sorted[0], hi = sorted[1];
    for (let i = 0; i < sorted.length - 1; i++) {
      if (sorted[i].ctx <= targetCtx && sorted[i + 1].ctx >= targetCtx) {
        lo = sorted[i];
        hi = sorted[i + 1];
        break;
      }
    }
    return {
      value: logLerp(targetCtx, lo.ctx, hi.ctx, metric(lo), metric(hi)),
      confidence: "interpolated",
    };
  }

  // Outside range — extrapolate from nearest two points
  const [a, b] = targetCtx < ctxValues[0]
    ? [sorted[0], sorted[1]]
    : [sorted[sorted.length - 2], sorted[sorted.length - 1]];
  return {
    value: Math.max(0, logLerp(targetCtx, a.ctx, b.ctx, metric(a), metric(b))),
    confidence: "extrapolated",
  };
}

// Interpolate a metric across model sizes
function interpolateByModelSize(
  sizeMetricPairs: { size: number; value: number }[],
  targetSize: number
): { value: number; confidence: "interpolated" | "extrapolated" | "nearest" } | null {
  if (sizeMetricPairs.length === 0) return null;

  const sorted = [...sizeMetricPairs].sort((a, b) => a.size - b.size);

  // Exact match
  const exact = sorted.find((p) => p.size === targetSize);
  if (exact) return { value: exact.value, confidence: "interpolated" };

  if (sorted.length === 1) return { value: sorted[0].value, confidence: "nearest" };

  // Within range
  if (targetSize >= sorted[0].size && targetSize <= sorted[sorted.length - 1].size) {
    let lo = sorted[0], hi = sorted[1];
    for (let i = 0; i < sorted.length - 1; i++) {
      if (sorted[i].size <= targetSize && sorted[i + 1].size >= targetSize) {
        lo = sorted[i];
        hi = sorted[i + 1];
        break;
      }
    }
    return {
      value: logLerp(targetSize, lo.size, hi.size, lo.value, hi.value),
      confidence: "interpolated",
    };
  }

  // Extrapolate
  const [a, b] = targetSize < sorted[0].size
    ? [sorted[0], sorted[1]]
    : [sorted[sorted.length - 2], sorted[sorted.length - 1]];
  return {
    value: Math.max(0, logLerp(targetSize, a.size, b.size, a.value, b.value)),
    confidence: "extrapolated",
  };
}

function interpolateFromBenchmarks(
  points: BenchmarkPoint[],
  gpuKey: string,
  targetModelSizeB: number,
  targetCtx: number,
  targetIsMoe: boolean,
  targetActiveParamsB: number,
): InterpolatedEstimate | null {
  const gpuLabel = GPU_KEY_TO_BENCHMARK[gpuKey];
  if (!gpuLabel) return null;

  const gpuPoints = points.filter((p) => p.gpu === gpuLabel);
  if (gpuPoints.length === 0) return null;

  // Group by model
  const modelGroups = new Map<string, BenchmarkPoint[]>();
  for (const p of gpuPoints) {
    const key = p.model;
    if (!modelGroups.has(key)) modelGroups.set(key, []);
    modelGroups.get(key)!.push(p);
  }

  // For each model, interpolate by context length first
  // Use active params as the interpolation axis — this makes MoE and dense comparable
  const sizeToThroughput: { size: number; value: number }[] = [];
  const sizeToTtft: { size: number; value: number }[] = [];
  const sizeToE2e: { size: number; value: number }[] = [];
  const usedModels: string[] = [];

  for (const [modelName, modelPoints] of modelGroups) {
    const meta = BENCHMARK_MODELS[modelName];
    if (!meta) continue;

    // Filter by architecture: match MoE with MoE, dense with dense
    // If no same-architecture matches exist, we'll fall back to all models below
    if (meta.isMoe !== targetIsMoe) continue;

    // Use TP8/DP1 config if available for consistency
    const tp8Points = modelPoints.filter((p) => p.config === "TP8/DP1");
    const usePoints = tp8Points.length > 0 ? tp8Points : modelPoints;

    // Deduplicate by ctx (pick best throughput per ctx)
    const byCtx = new Map<number, BenchmarkPoint>();
    for (const p of usePoints) {
      const existing = byCtx.get(p.ctx);
      if (!existing || p.throughputPerGpu > existing.throughputPerGpu) {
        byCtx.set(p.ctx, p);
      }
    }
    const dedupedPoints = [...byCtx.values()];

    // Use active params as the size axis for interpolation
    const interpSize = meta.activeB;

    const throughputResult = interpolateByCtx(dedupedPoints, targetCtx, (p) => p.throughputPerGpu);
    const ttftResult = interpolateByCtx(dedupedPoints, targetCtx, (p) => p.ttft);
    const e2eResult = interpolateByCtx(dedupedPoints, targetCtx, (p) => p.e2eLatency * 1000);

    if (throughputResult) sizeToThroughput.push({ size: interpSize, value: throughputResult.value });
    if (ttftResult) sizeToTtft.push({ size: interpSize, value: ttftResult.value });
    if (e2eResult) sizeToE2e.push({ size: interpSize, value: e2eResult.value });
    usedModels.push(`${modelName}(${interpSize}B active)`);
  }

  // Fall back to all models if no same-architecture matches
  if (sizeToThroughput.length === 0) {
    for (const [modelName, modelPoints] of modelGroups) {
      const meta = BENCHMARK_MODELS[modelName];
      if (!meta) continue;

      const tp8Points = modelPoints.filter((p) => p.config === "TP8/DP1");
      const usePoints = tp8Points.length > 0 ? tp8Points : modelPoints;

      const byCtx = new Map<number, BenchmarkPoint>();
      for (const p of usePoints) {
        const existing = byCtx.get(p.ctx);
        if (!existing || p.throughputPerGpu > existing.throughputPerGpu) {
          byCtx.set(p.ctx, p);
        }
      }
      const dedupedPoints = [...byCtx.values()];
      const interpSize = meta.activeB;

      const throughputResult = interpolateByCtx(dedupedPoints, targetCtx, (p) => p.throughputPerGpu);
      const ttftResult = interpolateByCtx(dedupedPoints, targetCtx, (p) => p.ttft);
      const e2eResult = interpolateByCtx(dedupedPoints, targetCtx, (p) => p.e2eLatency * 1000);

      if (throughputResult) sizeToThroughput.push({ size: interpSize, value: throughputResult.value });
      if (ttftResult) sizeToTtft.push({ size: interpSize, value: ttftResult.value });
      if (e2eResult) sizeToE2e.push({ size: interpSize, value: e2eResult.value });
      usedModels.push(`${modelName}(${interpSize}B active)`);
    }
  }

  if (sizeToThroughput.length === 0) return null;

  // Interpolate across active model sizes
  const targetInterpSize = targetIsMoe ? targetActiveParamsB : targetModelSizeB;

  const throughputEst = interpolateByModelSize(sizeToThroughput, targetInterpSize);
  const ttftEst = interpolateByModelSize(sizeToTtft, targetInterpSize);
  const e2eEst = interpolateByModelSize(sizeToE2e, targetInterpSize);

  if (!throughputEst || !ttftEst || !e2eEst) return null;

  const worstConfidence = [throughputEst, ttftEst, e2eEst].some((e) => e.confidence === "extrapolated")
    ? "extrapolated"
    : [throughputEst, ttftEst, e2eEst].some((e) => e.confidence === "nearest")
      ? "nearest"
      : "interpolated";

  // Build description
  const archLabel = targetIsMoe ? "MoE" : "dense";
  const ctxValues = [...new Set(gpuPoints.map((p) => p.ctx))].sort((a, b) => a - b);
  const basedOn = `${gpuLabel} ${archLabel} data: ${usedModels.join(", ")} at ${ctxValues.map((c) => c >= 1000 ? `${(c / 1000)}K` : c).join(", ")} ctx`;

  return {
    throughputPerGpu: Math.max(1, Math.round(throughputEst.value)),
    ttftMs: Math.max(0, ttftEst.value),
    e2eMs: Math.max(0, e2eEst.value),
    confidence: worstConfidence,
    basedOn,
  };
}

function findMatchingBenchmarks(
  points: BenchmarkPoint[],
  gpuKey: string,
  inputTokens: number,
): BenchmarkPoint[] {
  const gpuLabel = GPU_KEY_TO_BENCHMARK[gpuKey];
  if (!gpuLabel) return [];

  // Find benchmarks matching the GPU
  const gpuMatches = points.filter((p) => p.gpu === gpuLabel);
  if (gpuMatches.length === 0) return [];

  // Find the closest context length available
  const availableCtx = [...new Set(gpuMatches.map((p) => p.ctx))].sort((a, b) => a - b);
  const closestCtx = availableCtx.reduce((prev, curr) =>
    Math.abs(curr - inputTokens) < Math.abs(prev - inputTokens) ? curr : prev
  );

  // Return all benchmarks at the closest context length for this GPU
  return gpuMatches
    .filter((p) => p.ctx === closestCtx)
    .sort((a, b) => b.throughputPerGpu - a.throughputPerGpu);
}

function VramBar({ label, value, total, color }: { label: string; value: number; total: number; color: string }) {
  const pct = Math.min((value / total) * 100, 100);
  return (
    <div className="flex items-center gap-2 sm:gap-3">
      <div className="w-24 shrink-0 text-xs text-muted-foreground sm:w-28">{label}</div>
      <div className="flex min-w-0 flex-1 items-center gap-2">
        <div className="flex-1">
          <div className="h-5 w-full rounded-full bg-muted">
            <div
              className="h-5 rounded-full transition-all duration-300"
              style={{ width: `${Math.max(pct, 2)}%`, backgroundColor: color }}
            />
          </div>
        </div>
        <span className="w-16 text-right text-xs font-medium text-foreground">
          {value.toFixed(1)} GB
        </span>
      </div>
    </div>
  );
}

function formatLatency(ms: number): string {
  if (ms < 1) return `${(ms * 1000).toFixed(0)} us`;
  if (ms < 1000) return `${ms.toFixed(1)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

export function GpuEstimator({ benchmarkData }: { benchmarkData: BenchmarkData }) {
  const [paramsB, setParamsB] = useState(70);
  const [quant, setQuant] = useState("bf16");
  const [kvPrecision, setKvPrecision] = useState("bf16");
  const [contextLength, setContextLength] = useState(8192);
  const [gpuType, setGpuType] = useState("h100");
  const [concurrentRequests, setConcurrentRequests] = useState(10);
  const OUTPUT_TOKENS = 128;

  // MoE settings
  const [isMoe, setIsMoe] = useState(false);
  const [moeNumExperts, setMoeNumExperts] = useState(8);
  const [moeActiveExperts, setMoeActiveExperts] = useState(2);
  const [moeSharedExperts, setMoeSharedExperts] = useState(0);

  // HuggingFace config
  const [hfUrl, setHfUrl] = useState("");
  const [hfConfig, setHfConfig] = useState<ModelConfig | null>(null);
  const [hfLoading, setHfLoading] = useState(false);
  const [hfError, setHfError] = useState<string | null>(null);

  const fetchHfConfig = useCallback(async () => {
    if (!hfUrl.trim()) return;

    setHfLoading(true);
    setHfError(null);
    setHfConfig(null);

    try {
      const resolvedUrl = resolveHfUrl(hfUrl);
      const res = await fetch(resolvedUrl);
      if (!res.ok) {
        throw new Error(`Failed to fetch (${res.status}). Check the URL or model visibility.`);
      }
      const json = await res.json();
      const config = extractConfig(json);
      setHfConfig(config);

      if (config.moe) {
        setIsMoe(true);
        setMoeNumExperts(config.moe.numExperts);
        setMoeActiveExperts(config.moe.activeExperts);
        setMoeSharedExperts(config.moe.sharedExperts);
      } else {
        setIsMoe(false);
      }
    } catch (e) {
      setHfError(e instanceof Error ? e.message : "Failed to fetch config");
    } finally {
      setHfLoading(false);
    }
  }, [hfUrl]);

  const clearHfConfig = useCallback(() => {
    setHfConfig(null);
    setHfUrl("");
    setHfError(null);
  }, []);

  const result = useMemo(
    () =>
      estimateGpus(
        paramsB, quant, kvPrecision, contextLength, gpuType,
        concurrentRequests, contextLength, OUTPUT_TOKENS, hfConfig,
        isMoe, moeNumExperts, moeActiveExperts, moeSharedExperts
      ),
    [paramsB, quant, kvPrecision, contextLength, gpuType, concurrentRequests,
     OUTPUT_TOKENS, hfConfig,
     isMoe, moeNumExperts, moeActiveExperts, moeSharedExperts]
  );

  const gpu = GPU_SPECS[gpuType];

  const matchingBenchmarks = useMemo(
    () => findMatchingBenchmarks(benchmarkData.points, gpuType, contextLength),
    [benchmarkData.points, gpuType, contextLength]
  );

  const interpolated = useMemo(
    () => interpolateFromBenchmarks(
      benchmarkData.points, gpuType, paramsB, contextLength,
      isMoe, result.activeParamsB
    ),
    [benchmarkData.points, gpuType, paramsB, contextLength, isMoe, result.activeParamsB]
  );

  return (
    <div className="grid min-w-0 gap-6 lg:grid-cols-2">
      {/* Input Panel */}
      <div className="min-w-0 rounded-xl border border-border bg-card p-4 sm:p-6">
        <h3 className="mb-4 text-base font-semibold">Configuration</h3>
        <div className="space-y-5">
          {/* HuggingFace Config URL */}
          <div>
            <label className="mb-1.5 flex items-center gap-1.5 text-sm font-medium">
              <LinkIcon className="h-3.5 w-3.5" />
              HuggingFace Config
              <span className="text-xs font-normal text-muted-foreground">(optional)</span>
            </label>
            <p className="mb-2 text-[11px] text-muted-foreground">
              Paste a link to a model&apos;s config.json for precise KV cache estimation and auto-detection of MoE architecture.
            </p>
            <div className="flex gap-2">
              <input
                type="text"
                value={hfUrl}
                onChange={(e) => {
                  setHfUrl(e.target.value);
                  if (hfError) setHfError(null);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") fetchHfConfig();
                }}
                placeholder="https://huggingface.co/org/model/blob/main/config.json"
                className="flex-1 rounded-md border border-border bg-background px-3 py-1.5 text-xs focus:outline-none focus:ring-2 focus:ring-primary/50"
              />
              <button
                onClick={fetchHfConfig}
                disabled={hfLoading || !hfUrl.trim()}
                className="rounded-md border border-border bg-background px-3 py-1.5 text-xs font-medium transition-colors hover:bg-muted disabled:opacity-50"
              >
                {hfLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : "Load"}
              </button>
            </div>

            {hfError && (
              <div className="mt-2 flex items-center gap-1.5 text-xs text-red-500">
                <XCircle className="h-3.5 w-3.5" />
                {hfError}
              </div>
            )}

            {hfConfig && (
              <div className="mt-2 rounded-lg border border-green-200 bg-green-50 p-3 dark:border-green-900 dark:bg-green-950/30">
                <div className="mb-1.5 flex items-center justify-between">
                  <div className="flex items-center gap-1.5 text-xs font-medium text-green-700 dark:text-green-400">
                    <CheckCircle2 className="h-3.5 w-3.5" />
                    Config loaded
                    {hfConfig.modelName && (
                      <span className="font-normal text-green-600 dark:text-green-500">
                        &mdash; {hfConfig.modelName}
                      </span>
                    )}
                  </div>
                  <button
                    onClick={clearHfConfig}
                    className="text-[10px] text-green-600 underline hover:text-green-800 dark:text-green-500 dark:hover:text-green-300"
                  >
                    Clear
                  </button>
                </div>
                <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-[11px] text-green-700 dark:text-green-400">
                  <span>Layers: <span className="font-semibold">{hfConfig.numLayers}</span></span>
                  <span>Hidden: <span className="font-semibold">{hfConfig.hiddenSize.toLocaleString()}</span></span>
                  <span>Attn heads: <span className="font-semibold">{hfConfig.numAttentionHeads}</span></span>
                  <span>KV heads: <span className="font-semibold">{hfConfig.numKvHeads}</span></span>
                  <span>Head dim: <span className="font-semibold">{hfConfig.headDim}</span></span>
                  <span>Vocab: <span className="font-semibold">{hfConfig.vocabSize.toLocaleString()}</span></span>
                  {hfConfig.moe && (
                    <>
                      <span>Experts: <span className="font-semibold">{hfConfig.moe.numExperts}</span></span>
                      <span>Active: <span className="font-semibold">{hfConfig.moe.activeExperts}{hfConfig.moe.sharedExperts > 0 ? ` + ${hfConfig.moe.sharedExperts} shared` : ""}</span></span>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Model Size */}
          <div>
            <label className="mb-1.5 block text-sm font-medium">
              Model Size: <span className="text-primary">{paramsB}B</span> parameters
              {isMoe && (
                <span className="ml-1 text-xs font-normal text-muted-foreground">
                  (active: ~{result.activeParamsB.toFixed(1)}B)
                </span>
              )}
            </label>
            <input
              type="range"
              min={1}
              max={1000}
              step={1}
              value={paramsB}
              onChange={(e) => setParamsB(Number(e.target.value))}
              className="w-full accent-primary"
            />
            <div className="mt-1 flex justify-between text-[10px] text-muted-foreground">
              <span>1B</span>
              <span>100B</span>
              <span>500B</span>
              <span>1000B</span>
            </div>
          </div>

          {/* MoE Toggle & Config */}
          <div>
            <div className="mb-1.5 flex items-center gap-2">
              <label className="text-sm font-medium">Mixture of Experts (MoE)</label>
              <button
                onClick={() => setIsMoe(!isMoe)}
                className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                  isMoe ? "bg-primary" : "bg-muted-foreground/30"
                }`}
              >
                <span
                  className={`inline-block h-3.5 w-3.5 rounded-full bg-white transition-transform ${
                    isMoe ? "translate-x-4" : "translate-x-0.5"
                  }`}
                />
              </button>
            </div>

            {isMoe && (
              <div className="mt-2 rounded-lg border border-border bg-muted/30 p-3">
                <div className="grid grid-cols-3 gap-3">
                  <div>
                    <label className="mb-1 block text-[11px] text-muted-foreground">Total Experts</label>
                    <input
                      type="number"
                      min={2}
                      max={256}
                      value={moeNumExperts}
                      onChange={(e) => setMoeNumExperts(Math.max(2, Number(e.target.value)))}
                      className="w-full rounded-md border border-border bg-background px-2 py-1 text-xs focus:outline-none focus:ring-2 focus:ring-primary/50"
                    />
                  </div>
                  <div>
                    <label className="mb-1 block text-[11px] text-muted-foreground">Active / Token</label>
                    <input
                      type="number"
                      min={1}
                      max={moeNumExperts}
                      value={moeActiveExperts}
                      onChange={(e) => setMoeActiveExperts(Math.max(1, Math.min(moeNumExperts, Number(e.target.value))))}
                      className="w-full rounded-md border border-border bg-background px-2 py-1 text-xs focus:outline-none focus:ring-2 focus:ring-primary/50"
                    />
                  </div>
                  <div>
                    <label className="mb-1 block text-[11px] text-muted-foreground">Shared Experts</label>
                    <input
                      type="number"
                      min={0}
                      max={moeNumExperts}
                      value={moeSharedExperts}
                      onChange={(e) => setMoeSharedExperts(Math.max(0, Number(e.target.value)))}
                      className="w-full rounded-md border border-border bg-background px-2 py-1 text-xs focus:outline-none focus:ring-2 focus:ring-primary/50"
                    />
                  </div>
                </div>
                <p className="mt-2 text-[10px] text-muted-foreground">
                  All {paramsB}B weights stay in VRAM, but only ~{result.activeParamsB.toFixed(1)}B are active per token.
                  Activation overhead is based on active parameters.
                </p>
              </div>
            )}
          </div>

          {/* Precision */}
          <div>
            <label className="mb-1.5 block text-sm font-medium">Precision</label>
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(PRECISION).map(([key, q]) => (
                <button
                  key={key}
                  onClick={() => setQuant(key)}
                  className={`rounded-lg border px-3 py-2 text-xs font-medium transition-colors ${
                    quant === key
                      ? "border-primary bg-primary/10 text-primary"
                      : "border-border bg-background text-muted-foreground hover:border-primary/50"
                  }`}
                >
                  {q.label}
                </button>
              ))}
            </div>
          </div>

          {/* KV Cache Precision */}
          <div>
            <label className="mb-1.5 block text-sm font-medium">KV Cache Precision</label>
            <div className="flex gap-2">
              {Object.entries(KV_PRECISION).map(([key, kv]) => (
                <button
                  key={key}
                  onClick={() => setKvPrecision(key)}
                  className={`rounded-lg border px-3 py-2 text-xs font-medium transition-colors ${
                    kvPrecision === key
                      ? "border-primary bg-primary/10 text-primary"
                      : "border-border bg-background text-muted-foreground hover:border-primary/50"
                  }`}
                >
                  {kv.label}
                </button>
              ))}
            </div>
          </div>

          {/* Context Length */}
          <div>
            <label className="mb-1.5 block text-sm font-medium">
              Context Length: <span className="text-primary">{formatCtx(contextLength)}</span>
            </label>
            <input
              type="range"
              min={0}
              max={100}
              step={1}
              value={ctxValueToSlider(contextLength)}
              onChange={(e) => setContextLength(ctxSliderToValue(Number(e.target.value)))}
              className="w-full accent-primary"
            />
            <div className="mt-1 flex justify-between text-[10px] text-muted-foreground">
              <span>1K</span>
              <span>4K</span>
              <span>16K</span>
              <span>64K</span>
              <span>128K</span>
            </div>
          </div>

          {/* Concurrent Requests */}
          <div>
            <label className="mb-1.5 block text-sm font-medium">
              Concurrent Requests: <span className="text-primary">{concurrentRequests}</span>
            </label>
            <input
              type="range"
              min={1}
              max={256}
              step={1}
              value={concurrentRequests}
              onChange={(e) => setConcurrentRequests(Number(e.target.value))}
              className="w-full accent-primary"
            />
            <div className="mt-1 flex justify-between text-[10px] text-muted-foreground">
              <span>1</span>
              <span>64</span>
              <span>128</span>
              <span>256</span>
            </div>
          </div>

          {/* GPU Type */}
          <div>
            <label className="mb-1.5 block text-sm font-medium">Target GPU</label>
            <select
              value={gpuType}
              onChange={(e) => setGpuType(e.target.value)}
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
            >
              {Object.entries(GPU_SPECS).map(([key, spec]) => (
                <option key={key} value={key}>
                  {spec.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Results Panel */}
      <div className="min-w-0 space-y-4">
        {/* GPU Count Result */}
        <div className="rounded-xl border-2 border-primary/30 bg-primary/5 p-4 sm:p-6">
          <div className="text-sm text-muted-foreground">Estimated GPUs Required</div>
          <div className="mt-1 text-5xl font-bold text-primary">{result.numGpus}</div>
          <div className="mt-2 text-sm text-muted-foreground">{result.tpRecommendation}</div>
        </div>

        {/* Latency Estimates */}
        <div className="rounded-xl border border-border bg-card p-4 sm:p-6">
          <h4 className="mb-3 text-sm font-semibold">Estimated Latency</h4>

          {/* Data-interpolated estimate (primary when available) */}
          {interpolated && (
            <>
              <div className="mb-1 flex items-center gap-1.5">
                <Database className="h-3 w-3 text-primary" />
                <span className="text-[10px] font-medium text-primary">
                  Data-driven ({interpolated.confidence})
                </span>
              </div>
              <div className="mb-3 grid grid-cols-1 gap-3 sm:grid-cols-3">
                <div className="rounded-lg border border-primary/20 bg-primary/5 p-3">
                  <div className="text-[10px] uppercase tracking-wider text-muted-foreground">TTFT</div>
                  <div className="mt-1 text-lg font-bold text-foreground">{formatLatency(interpolated.ttftMs)}</div>
                  <div className="mt-0.5 text-[10px] text-muted-foreground">Time to first token</div>
                </div>
                <div className="rounded-lg border border-primary/20 bg-primary/5 p-3">
                  <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Throughput</div>
                  <div className="mt-1 text-lg font-bold text-foreground">{interpolated.throughputPerGpu.toLocaleString()}</div>
                  <div className="mt-0.5 text-[10px] text-muted-foreground">tok/s/GPU</div>
                </div>
                <div className="rounded-lg border border-primary/20 bg-primary/5 p-3">
                  <div className="text-[10px] uppercase tracking-wider text-muted-foreground">E2E</div>
                  <div className="mt-1 text-lg font-bold text-foreground">{formatLatency(interpolated.e2eMs)}</div>
                  <div className="mt-0.5 text-[10px] text-muted-foreground">{contextLength} in / {OUTPUT_TOKENS} out</div>
                </div>
              </div>
              <p className="mb-4 text-[10px] text-muted-foreground">
                {interpolated.confidence === "interpolated"
                  ? "Interpolated from real benchmark data."
                  : interpolated.confidence === "extrapolated"
                    ? "Extrapolated beyond available data range — treat as rough estimate."
                    : "Based on nearest available data point."}
                {" "}{interpolated.basedOn}.
                {" "}100 concurrent requests, 128 output tokens.
              </p>
            </>
          )}

          {/* Theoretical estimate */}
          <div className={interpolated ? "mt-1" : ""}>
            {interpolated && (
              <div className="mb-1 text-[10px] font-medium text-muted-foreground">
                Theoretical (FLOPS / bandwidth model)
              </div>
            )}
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
              <div className="rounded-lg bg-muted/50 p-3">
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground">TTFT</div>
                <div className={`mt-1 font-bold text-foreground ${interpolated ? "text-sm" : "text-lg"}`}>{formatLatency(result.ttftMs)}</div>
                <div className="mt-0.5 text-[10px] text-muted-foreground">Time to first token</div>
              </div>
              <div className="rounded-lg bg-muted/50 p-3">
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground">TPOT</div>
                <div className={`mt-1 font-bold text-foreground ${interpolated ? "text-sm" : "text-lg"}`}>{formatLatency(result.tpotMs)}</div>
                <div className="mt-0.5 text-[10px] text-muted-foreground">Per output token</div>
              </div>
              <div className="rounded-lg bg-muted/50 p-3">
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground">E2E</div>
                <div className={`mt-1 font-bold text-foreground ${interpolated ? "text-sm" : "text-lg"}`}>{formatLatency(result.e2eMs)}</div>
                <div className="mt-0.5 text-[10px] text-muted-foreground">{contextLength} in / {OUTPUT_TOKENS} out</div>
              </div>
            </div>
          </div>
          <p className="mt-3 text-[10px] text-muted-foreground">
            {interpolated ? "Theoretical: " : ""}Single request estimate. Prefill is compute-bound (~35% of {gpu.tflops} TFLOPS).
            Decode is memory-bandwidth-bound (~65% of {gpu.bw} GB/s).
            {isMoe && ` MoE decode reads only active weights (~${result.activeParamsB.toFixed(1)}B).`}
          </p>
        </div>

        {/* VRAM Breakdown — Per GPU */}
        <div className="rounded-xl border border-border bg-card p-4 sm:p-6">
          <h4 className="mb-3 text-sm font-semibold">VRAM Breakdown (per GPU)</h4>
          <div className="space-y-3">
            <VramBar
              label="Model Weights"
              value={result.modelVram}
              total={result.gpuVram}
              color="#3b82f6"
            />
            <VramBar
              label="KV Cache"
              value={result.kvCacheVram}
              total={result.gpuVram}
              color="#f97316"
            />
            <VramBar
              label={result.isMoe ? "Overhead (active)" : "Overhead"}
              value={result.overheadVram}
              total={result.gpuVram}
              color="#6b7280"
            />
          </div>
          <div className="mt-4 border-t border-border pt-3">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">VRAM per GPU</span>
              <span className="font-semibold">{result.totalVram.toFixed(1)} GB</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">GPU VRAM</span>
              <span className="font-semibold">{result.gpuVram.toFixed(0)} GB</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Utilization</span>
              <span className="font-semibold">{result.utilizationPercent.toFixed(0)}%</span>
            </div>
          </div>
        </div>

        {/* VRAM Breakdown — Overall */}
        {result.numGpus > 1 && (() => {
          const overallModel = result.modelVram * result.tp * result.dp;
          const overallKv = result.kvCacheVram * result.tp * result.dp;
          const overallOverhead = result.overheadVram * result.tp * result.dp;
          const overallTotal = overallModel + overallKv + overallOverhead;
          const overallAvailable = result.numGpus * result.gpuVram;
          return (
            <div className="rounded-xl border border-border bg-card p-4 sm:p-6">
              <h4 className="mb-3 text-sm font-semibold">VRAM Breakdown (overall across {result.numGpus} GPUs)</h4>
              <div className="space-y-3">
                <VramBar
                  label="Model Weights"
                  value={overallModel}
                  total={overallAvailable}
                  color="#3b82f6"
                />
                <VramBar
                  label="KV Cache"
                  value={overallKv}
                  total={overallAvailable}
                  color="#f97316"
                />
                <VramBar
                  label={result.isMoe ? "Overhead (active)" : "Overhead"}
                  value={overallOverhead}
                  total={overallAvailable}
                  color="#6b7280"
                />
              </div>
              <div className="mt-4 border-t border-border pt-3">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Total VRAM needed</span>
                  <span className="font-semibold">{overallTotal.toFixed(1)} GB</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Available VRAM ({result.numGpus} GPUs)</span>
                  <span className="font-semibold">{overallAvailable.toFixed(0)} GB</span>
                </div>
              </div>
            </div>
          );
        })()}

        {/* Quick Reference */}
        <div className="overflow-hidden rounded-xl border border-border bg-card p-4 sm:p-6">
          <h4 className="mb-3 text-sm font-semibold">
            Quick Reference
            {result.usingHfConfig && (
              <span className="ml-2 text-[10px] font-normal text-green-600 dark:text-green-400">
                Using HuggingFace config
              </span>
            )}
          </h4>
          <div className="space-y-2 text-xs text-muted-foreground">
            <div className="flex flex-wrap justify-between gap-x-4">
              <span>Model weights / GPU</span>
              <span className="text-right">{paramsB}B &times; {PRECISION[quant].bytesPerParam} bytes{result.tp > 1 ? ` / TP${result.tp}` : ""} = {result.modelVram.toFixed(1)} GB</span>
            </div>
            {result.isMoe && (
              <div className="flex flex-wrap justify-between gap-x-4">
                <span>Active params (MoE)</span>
                <span className="text-right">
                  ~{result.activeParamsB.toFixed(1)}B of {paramsB}B
                  ({moeActiveExperts}{moeSharedExperts > 0 ? `+${moeSharedExperts}` : ""}/{moeNumExperts} experts)
                </span>
              </div>
            )}
            <div className="flex flex-wrap justify-between gap-x-4">
              <span>KV cache per token</span>
              <span className="text-right">
                {result.usingHfConfig && hfConfig
                  ? `2 x ${hfConfig.numLayers}L x ${hfConfig.numKvHeads}KV x ${hfConfig.headDim}d x ${KV_PRECISION[kvPrecision].bytes}B = ${(result.kvPerTokenBytes / 1024).toFixed(1)} KB`
                  : `~${(result.kvPerTokenBytes / 1024).toFixed(1)} KB (estimated)`}
              </span>
            </div>
            <div className="flex flex-wrap justify-between gap-x-4">
              <span>KV cache / GPU</span>
              <span className="text-right">
                {(result.kvPerTokenBytes / 1024).toFixed(1)} KB &times; {contextLength.toLocaleString()} tokens &times; {concurrentRequests} req{result.numGpus > 1 ? ` / ${result.numGpus} GPUs` : ""} = {result.kvCacheVram.toFixed(1)} GB
              </span>
            </div>
            <div className="flex flex-wrap justify-between gap-x-4">
              <span>Framework overhead / GPU</span>
              <span className="text-right">
                ~10% of {result.isMoe ? `${result.activeParamsB.toFixed(1)}B active` : "model"}{result.tp > 1 ? ` / TP${result.tp}` : ""} = {result.overheadVram.toFixed(1)} GB
              </span>
            </div>
            <div className="mt-1 flex flex-wrap justify-between gap-x-4 border-t border-border pt-2">
              <span>TTFT</span>
              <span className="text-right">(linear + attention FLOPs for {contextLength.toLocaleString()} tokens) / (TP{result.tp} &times; {gpu.tflops} TFLOPS &times; 35%) &times; concurrency factor</span>
            </div>
            <div className="flex flex-wrap justify-between gap-x-4">
              <span>TPOT</span>
              <span className="text-right">(weights/TP{result.tp} + KV for ~{(contextLength + OUTPUT_TOKENS / 2).toLocaleString()} seq len) / (TP{result.tp} &times; {gpu.bw} GB/s &times; 65%) &times; {concurrentRequests / result.dp} batch</span>
            </div>
          </div>
        </div>

        {/* Real Benchmark Reference */}
        <div className="rounded-xl border border-border bg-card p-4 sm:p-6">
          <h4 className="mb-1 flex items-center gap-1.5 text-sm font-semibold">
            <Database className="h-3.5 w-3.5" />
            Real Benchmark Reference
          </h4>
          {matchingBenchmarks.length > 0 ? (
            <>
              <p className="mb-3 text-[10px] text-muted-foreground">
                Actual results on {GPU_KEY_TO_BENCHMARK[gpuType]} at {matchingBenchmarks[0].ctx.toLocaleString()} input tokens
                (closest to your {contextLength.toLocaleString()}).
                8 GPUs, {matchingBenchmarks[0].concurrency} concurrent, {matchingBenchmarks[0].numPrompts} prompts, {OUTPUT_TOKENS} output tokens.
              </p>
              <div className="-mx-4 overflow-x-auto px-4 sm:-mx-6 sm:px-6">
                <table className="w-full min-w-[480px] text-xs">
                  <thead>
                    <tr className="border-b border-border text-left text-[10px] uppercase tracking-wider text-muted-foreground">
                      <th className="pb-2 pr-3 font-medium">Model</th>
                      <th className="pb-2 pr-3 font-medium">Engine</th>
                      <th className="pb-2 pr-3 font-medium">Config</th>
                      <th className="whitespace-nowrap pb-2 pr-3 text-right font-medium">Thpt/GPU</th>
                      <th className="pb-2 pr-3 text-right font-medium">TTFT</th>
                      <th className="whitespace-nowrap pb-2 text-right font-medium">E2E</th>
                    </tr>
                  </thead>
                  <tbody>
                    {matchingBenchmarks.map((b, i) => (
                      <tr key={i} className="border-b border-border/50 last:border-0">
                        <td className="py-1.5 pr-3 font-medium text-foreground">{b.model}</td>
                        <td className="py-1.5 pr-3">{b.engine}</td>
                        <td className="py-1.5 pr-3">{b.config}</td>
                        <td className="whitespace-nowrap py-1.5 pr-3 text-right font-medium text-foreground">{b.throughputPerGpu.toLocaleString()} tok/s</td>
                        <td className="whitespace-nowrap py-1.5 pr-3 text-right">{b.ttft.toFixed(0)} ms</td>
                        <td className="whitespace-nowrap py-1.5 text-right">{b.e2eLatency < 1 ? `${(b.e2eLatency * 1000).toFixed(0)} ms` : `${b.e2eLatency.toFixed(2)} s`}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {matchingBenchmarks[0].ctx !== contextLength && (
                <p className="mt-2 text-[10px] text-muted-foreground">
                  Note: Closest available context length is {matchingBenchmarks[0].ctx.toLocaleString()} (you selected {contextLength.toLocaleString()}).
                </p>
              )}
            </>
          ) : (
            <p className="mt-2 text-xs text-muted-foreground">
              No benchmark data available for {gpu.label}. Real benchmarks are available for B200, H200 SXM, H100 SXM, and A100 SXM.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
