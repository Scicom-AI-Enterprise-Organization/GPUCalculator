"use client";

import { useState, useMemo, useCallback } from "react";
import { Loader2, CheckCircle2, XCircle, Link as LinkIcon } from "lucide-react";

const GPU_SPECS: Record<string, { vram: number; label: string }> = {
  b200: { vram: 192, label: "B200 (192 GB)" },
  h200: { vram: 141, label: "H200 SXM (141 GB)" },
  h100: { vram: 80, label: "H100 SXM (80 GB)" },
  a100_80: { vram: 80, label: "A100 SXM 80GB" },
  a100_40: { vram: 40, label: "A100 PCIe 40GB" },
  l40s: { vram: 48, label: "L40S (48 GB)" },
  a10: { vram: 24, label: "A10 (24 GB)" },
  rtx_4090: { vram: 24, label: "RTX 4090 (24 GB)" },
};

const PRECISION: Record<string, { bytesPerParam: number; label: string }> = {
  fp32: { bytesPerParam: 4, label: "FP32" },
  bf16: { bytesPerParam: 2, label: "BF16 / FP16" },
  fp8: { bytesPerParam: 1, label: "FP8" },
  int8: { bytesPerParam: 1, label: "INT8" },
  int4: { bytesPerParam: 0.5, label: "INT4 / GPTQ / AWQ" },
};

const KV_PRECISION: Record<string, { bytes: number; label: string }> = {
  bf16: { bytes: 2, label: "BF16 / FP16" },
  fp8: { bytes: 1, label: "FP8" },
};

const CONTEXT_OPTIONS = [
  { value: 2048, label: "2K" },
  { value: 4096, label: "4K" },
  { value: 8192, label: "8K" },
  { value: 16384, label: "16K" },
  { value: 32768, label: "32K" },
  { value: 65536, label: "64K" },
  { value: 131072, label: "128K" },
];

interface ModelConfig {
  modelName: string;
  numLayers: number;
  hiddenSize: number;
  numAttentionHeads: number;
  numKvHeads: number;
  headDim: number;
  intermediateSize: number;
  vocabSize: number;
}

interface EstimationResult {
  modelVram: number;
  kvCacheVram: number;
  overheadVram: number;
  totalVram: number;
  gpuVram: number;
  numGpus: number;
  tpRecommendation: string;
  utilizationPercent: number;
  kvPerTokenBytes: number;
  usingHfConfig: boolean;
}

function resolveHfUrl(url: string): string {
  // Convert blob URLs to resolve URLs for raw file access
  // https://huggingface.co/org/model/blob/main/config.json
  // → https://huggingface.co/org/model/resolve/main/config.json
  let resolved = url.trim();
  resolved = resolved.replace("/blob/", "/resolve/");

  // If user just pastes a model page URL without config.json, append it
  if (!resolved.includes("config.json")) {
    if (!resolved.endsWith("/")) resolved += "/";
    resolved += "resolve/main/config.json";
  }

  // Ensure it uses /resolve/
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
  // Some models (e.g. multimodal) nest text config
  const cfg = json.text_config || json;

  const numAttentionHeads = cfg.num_attention_heads ?? cfg.n_head ?? 32;
  const hiddenSize = cfg.hidden_size ?? cfg.n_embd ?? 4096;
  const headDim = cfg.head_dim ?? Math.floor(hiddenSize / numAttentionHeads);

  // Extract model name from model_type or architectures
  let modelName = cfg.model_type || "";
  if (json.architectures?.length) {
    modelName = json.architectures[0].replace(/ForCausalLM$/, "");
  }
  if (json._name_or_path) {
    modelName = json._name_or_path;
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
  };
}

function estimateGpus(
  paramsB: number,
  quant: string,
  kvPrecision: string,
  contextLength: number,
  gpuType: string,
  concurrentRequests: number,
  hfConfig: ModelConfig | null
): EstimationResult {
  const q = PRECISION[quant];
  const kv = KV_PRECISION[kvPrecision];
  const gpu = GPU_SPECS[gpuType];

  // Model weights VRAM
  const modelVram = paramsB * q.bytesPerParam;

  // KV cache estimation
  let kvPerTokenBytes: number;
  if (hfConfig) {
    // Precise: 2 (K+V) * num_layers * num_kv_heads * head_dim * bytes
    kvPerTokenBytes = 2 * hfConfig.numLayers * hfConfig.numKvHeads * hfConfig.headDim * kv.bytes;
  } else {
    // Rough empirical estimate (baseline is bf16 = 2 bytes)
    kvPerTokenBytes = paramsB * 0.00003 * 1e9 * (kv.bytes / 2);
  }
  const kvCacheVram = (kvPerTokenBytes * contextLength * concurrentRequests) / 1e9;

  // Framework overhead (~10% of model weights)
  const overheadVram = modelVram * 0.1;

  const totalVram = modelVram + kvCacheVram + overheadVram;

  // Calculate number of GPUs needed
  const rawGpus = totalVram / gpu.vram;
  let numGpus: number;
  if (rawGpus <= 1) {
    numGpus = 1;
  } else {
    numGpus = 1;
    while (numGpus < rawGpus) numGpus *= 2;
  }

  // TP recommendation
  let tpRecommendation: string;
  if (numGpus === 1) {
    tpRecommendation = "No parallelism needed";
  } else if (numGpus <= 2) {
    tpRecommendation = `TP${numGpus}`;
  } else if (numGpus <= 8) {
    tpRecommendation = `TP${numGpus} or TP${numGpus / 2}/DP2`;
  } else {
    const tp = Math.min(numGpus, 8);
    const dp = numGpus / tp;
    tpRecommendation = `TP${tp}/DP${dp} (multi-node)`;
  }

  const utilizationPercent = (totalVram / (numGpus * gpu.vram)) * 100;

  return {
    modelVram,
    kvCacheVram,
    overheadVram,
    totalVram,
    gpuVram: gpu.vram,
    numGpus,
    tpRecommendation,
    utilizationPercent,
    kvPerTokenBytes,
    usingHfConfig: hfConfig !== null,
  };
}

function VramBar({ label, value, total, color }: { label: string; value: number; total: number; color: string }) {
  const pct = Math.min((value / total) * 100, 100);
  return (
    <div className="flex items-center gap-3">
      <div className="w-28 text-xs text-muted-foreground">{label}</div>
      <div className="flex flex-1 items-center gap-2">
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

export function GpuEstimator() {
  const [paramsB, setParamsB] = useState(70);
  const [quant, setQuant] = useState("bf16");
  const [kvPrecision, setKvPrecision] = useState("bf16");
  const [contextLength, setContextLength] = useState(8192);
  const [gpuType, setGpuType] = useState("h100");
  const [concurrentRequests, setConcurrentRequests] = useState(10);

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
    () => estimateGpus(paramsB, quant, kvPrecision, contextLength, gpuType, concurrentRequests, hfConfig),
    [paramsB, quant, kvPrecision, contextLength, gpuType, concurrentRequests, hfConfig]
  );

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      {/* Input Panel */}
      <div className="rounded-xl border border-border bg-card p-6">
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
              Paste a link to a model&apos;s config.json for precise KV cache estimation using actual layer count and head dimensions.
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

            {/* Status */}
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
                </div>
              </div>
            )}
          </div>

          {/* Model Size */}
          <div>
            <label className="mb-1.5 block text-sm font-medium">
              Model Size: <span className="text-primary">{paramsB}B</span> parameters
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
            <label className="mb-1.5 block text-sm font-medium">Context Length</label>
            <div className="flex flex-wrap gap-2">
              {CONTEXT_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => setContextLength(opt.value)}
                  className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
                    contextLength === opt.value
                      ? "border-primary bg-primary/10 text-primary"
                      : "border-border bg-background text-muted-foreground hover:border-primary/50"
                  }`}
                >
                  {opt.label}
                </button>
              ))}
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
      <div className="space-y-4">
        {/* GPU Count Result */}
        <div className="rounded-xl border-2 border-primary/30 bg-primary/5 p-6">
          <div className="text-sm text-muted-foreground">Estimated GPUs Required</div>
          <div className="mt-1 text-5xl font-bold text-primary">{result.numGpus}</div>
          <div className="mt-2 text-sm text-muted-foreground">{result.tpRecommendation}</div>
        </div>

        {/* VRAM Breakdown */}
        <div className="rounded-xl border border-border bg-card p-6">
          <h4 className="mb-3 text-sm font-semibold">VRAM Breakdown</h4>
          <div className="space-y-3">
            <VramBar
              label="Model Weights"
              value={result.modelVram}
              total={result.numGpus * result.gpuVram}
              color="#3b82f6"
            />
            <VramBar
              label="KV Cache"
              value={result.kvCacheVram}
              total={result.numGpus * result.gpuVram}
              color="#f97316"
            />
            <VramBar
              label="Overhead"
              value={result.overheadVram}
              total={result.numGpus * result.gpuVram}
              color="#6b7280"
            />
          </div>
          <div className="mt-4 border-t border-border pt-3">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Total VRAM needed</span>
              <span className="font-semibold">{result.totalVram.toFixed(1)} GB</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Available VRAM ({result.numGpus} GPUs)</span>
              <span className="font-semibold">{(result.numGpus * result.gpuVram).toFixed(0)} GB</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Utilization</span>
              <span className="font-semibold">{result.utilizationPercent.toFixed(0)}%</span>
            </div>
          </div>
        </div>

        {/* Quick Reference */}
        <div className="rounded-xl border border-border bg-card p-6">
          <h4 className="mb-3 text-sm font-semibold">
            Quick Reference
            {result.usingHfConfig && (
              <span className="ml-2 text-[10px] font-normal text-green-600 dark:text-green-400">
                Using HuggingFace config
              </span>
            )}
          </h4>
          <div className="space-y-2 text-xs text-muted-foreground">
            <div className="flex justify-between">
              <span>Model weights</span>
              <span>{paramsB}B &times; {PRECISION[quant].bytesPerParam} bytes/param = {result.modelVram.toFixed(1)} GB</span>
            </div>
            <div className="flex justify-between">
              <span>KV cache per token</span>
              <span>
                {result.usingHfConfig && hfConfig
                  ? `2 x ${hfConfig.numLayers}L x ${hfConfig.numKvHeads}KV x ${hfConfig.headDim}d x ${KV_PRECISION[kvPrecision].bytes}B = ${(result.kvPerTokenBytes / 1024).toFixed(1)} KB`
                  : `~${(result.kvPerTokenBytes / 1024).toFixed(1)} KB (estimated)`}
              </span>
            </div>
            <div className="flex justify-between">
              <span>KV cache total</span>
              <span>
                {(result.kvPerTokenBytes / 1024).toFixed(1)} KB &times; {contextLength.toLocaleString()} tokens &times; {concurrentRequests} req = {result.kvCacheVram.toFixed(1)} GB
              </span>
            </div>
            <div className="flex justify-between">
              <span>Framework overhead</span>
              <span>~10% of model = {result.overheadVram.toFixed(1)} GB</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
