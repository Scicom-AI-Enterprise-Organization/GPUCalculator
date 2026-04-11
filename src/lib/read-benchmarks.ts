import fs from "fs";
import path from "path";

export interface BenchmarkPoint {
  engine: string;
  gpu: string;
  model: string;
  tp: number;
  dp: number;
  config: string;
  ctx: number;
  throughputPerGpu: number;
  e2eLatency: number;
  ttft: number;
  outputThroughput: number;
  totalTokenThroughput: number;
  numPrompts: number;
  concurrency: number;
}

export interface BenchmarkData {
  points: BenchmarkPoint[];
  gpus: string[];
  models: string[];
  engines: string[];
  configs: string[];
  contextLengths: number[];
}

const GPU_LABEL_MAP: Record<string, string> = {
  runpod_b200: "B200",
  runpod_h100sxm: "H100 SXM",
  runpod_h200sxm: "H200 SXM",
  runpod_a100sxm: "A100 SXM",
};

const MODEL_NAME_MAP: Record<string, string> = {
  "llama-3.1-70b": "llama3.1-70b",
};

function normalizeModel(dir: string): string {
  const lower = dir.toLowerCase();
  return MODEL_NAME_MAP[lower] || lower;
}

const NUM_GPUS = 8;

function parseVllmFile(filePath: string, gpuDir: string, modelDir: string): BenchmarkPoint | null {
  try {
    const raw = fs.readFileSync(filePath, "utf-8");
    const data = JSON.parse(raw);

    // Skip failed benchmarks (all requests failed, metrics are 0)
    if (data.completed === 0 || data.total_token_throughput === 0) return null;

    const fileName = path.basename(filePath, ".json");
    // Two naming conventions:
    //   (1) run_N_TP{tp}_DP{dp}_CTX{ctx}_C{c}_P{p}_O{o}         — b200 set
    //   (2) ..._TP{tp}_DP{dp}_in{ctx}_out{o}_p{p}_c{c}_{hash}   — h100/h200 set (incl. multi-concurrency)
    let tp: number, dp: number, ctx: number;
    const m1 = fileName.match(/_TP(\d+)_DP(\d+)_CTX(\d+)/);
    const m2 = fileName.match(/_TP(\d+)_DP(\d+)_in(\d+)_out\d+_p\d+_c\d+/i);
    if (m1) {
      tp = parseInt(m1[1]);
      dp = parseInt(m1[2]);
      ctx = parseInt(m1[3]);
    } else if (m2) {
      tp = parseInt(m2[1]);
      dp = parseInt(m2[2]);
      ctx = parseInt(m2[3]);
    } else {
      return null;
    }
    const gpu = GPU_LABEL_MAP[gpuDir] || gpuDir;

    return {
      engine: "vLLM",
      gpu,
      model: normalizeModel(modelDir),
      tp,
      dp,
      config: `TP${tp}/DP${dp}`,
      ctx,
      throughputPerGpu: Math.round(data.total_token_throughput / NUM_GPUS),
      e2eLatency: parseFloat((data.mean_e2el_ms / 1000).toFixed(2)),
      ttft: parseFloat((data.mean_ttft_ms).toFixed(2)),
      outputThroughput: parseFloat(data.output_throughput.toFixed(2)),
      totalTokenThroughput: parseFloat(data.total_token_throughput.toFixed(2)),
      numPrompts: data.num_prompts,
      concurrency: data.max_concurrency,
    };
  } catch {
    return null;
  }
}

function parseSglangTxt(filePath: string, gpuDir: string, modelDir: string): BenchmarkPoint | null {
  try {
    const raw = fs.readFileSync(filePath, "utf-8");

    const fileName = path.basename(filePath, ".txt");
    const match = fileName.match(/tp(\d+)_dp(\d+).*?_in(\d+)_out(\d+)_p(\d+)_c(\d+)/i);
    if (!match) return null;

    const tp = parseInt(match[1]);
    const dp = parseInt(match[2]);
    const ctx = parseInt(match[3]);
    const numPrompts = parseInt(match[5]);
    const concurrency = parseInt(match[6]);

    const totalThroughputMatch = raw.match(/Total token throughput \(tok\/s\):\s+([\d.]+)/);
    const e2eMatch = raw.match(/Mean E2E Latency \(ms\):\s+([\d.]+)/);
    const ttftMatch = raw.match(/Mean TTFT \(ms\):\s+([\d.]+)/);
    const outputThroughputMatch = raw.match(/Output token throughput \(tok\/s\):\s+([\d.]+)/);

    if (!totalThroughputMatch || !e2eMatch) return null;

    const totalTokenThroughput = parseFloat(totalThroughputMatch[1]);
    const gpu = GPU_LABEL_MAP[gpuDir] || gpuDir;

    return {
      engine: "SGLang",
      gpu,
      model: normalizeModel(modelDir),
      tp,
      dp,
      config: `TP${tp}/DP${dp}`,
      ctx,
      throughputPerGpu: Math.round(totalTokenThroughput / NUM_GPUS),
      e2eLatency: parseFloat((parseFloat(e2eMatch[1]) / 1000).toFixed(2)),
      ttft: ttftMatch ? parseFloat(parseFloat(ttftMatch[1]).toFixed(2)) : 0,
      outputThroughput: outputThroughputMatch ? parseFloat(parseFloat(outputThroughputMatch[1]).toFixed(2)) : 0,
      totalTokenThroughput: parseFloat(totalTokenThroughput.toFixed(2)),
      numPrompts,
      concurrency,
    };
  } catch {
    return null;
  }
}

export async function readBenchmarkData(): Promise<BenchmarkData> {
  const benchmarksDir = path.join(process.cwd(), "data", "llm-benchmaq", "benchmarks");
  const points: BenchmarkPoint[] = [];

  // Process vLLM benchmarks (vllm/<gpu>/<model>/)
  const vllmDir = path.join(benchmarksDir, "vllm");
  if (fs.existsSync(vllmDir)) {
    const gpuDirs = fs.readdirSync(vllmDir).filter((f) =>
      fs.statSync(path.join(vllmDir, f)).isDirectory()
    );

    for (const gpuDir of gpuDirs) {
      const gpuPath = path.join(vllmDir, gpuDir);
      const models = fs.readdirSync(gpuPath).filter((f) =>
        fs.statSync(path.join(gpuPath, f)).isDirectory()
      );

      for (const modelDir of models) {
        const modelPath = path.join(gpuPath, modelDir);
        const files = fs.readdirSync(modelPath).filter((f) => f.endsWith(".json"));

        for (const file of files) {
          const point = parseVllmFile(path.join(modelPath, file), gpuDir, modelDir);
          if (point) points.push(point);
        }
      }
    }
  }

  // Process SGLang benchmarks
  const sglangDir = path.join(benchmarksDir, "sglang");
  if (fs.existsSync(sglangDir)) {
    const gpuDirs = fs.readdirSync(sglangDir).filter((f) =>
      fs.statSync(path.join(sglangDir, f)).isDirectory()
    );

    for (const gpuDir of gpuDirs) {
      const gpuPath = path.join(sglangDir, gpuDir);
      const models = fs.readdirSync(gpuPath).filter((f) =>
        fs.statSync(path.join(gpuPath, f)).isDirectory()
      );

      for (const modelDir of models) {
        const modelPath = path.join(gpuPath, modelDir);
        const files = fs.readdirSync(modelPath).filter((f) => f.endsWith(".txt"));

        for (const file of files) {
          const point = parseSglangTxt(path.join(modelPath, file), gpuDir, modelDir);
          if (point) points.push(point);
        }
      }
    }
  }

  // Sort by context length for proper line connections
  points.sort((a, b) => a.ctx - b.ctx);

  const gpus = [...new Set(points.map((p) => p.gpu))].sort();
  const models = [...new Set(points.map((p) => p.model))].sort();
  const engines = [...new Set(points.map((p) => p.engine))].sort();
  const configs = [...new Set(points.map((p) => p.config))].sort();
  const contextLengths = [...new Set(points.map((p) => p.ctx))].sort((a, b) => a - b);

  return { points, gpus, models, engines, configs, contextLengths };
}
