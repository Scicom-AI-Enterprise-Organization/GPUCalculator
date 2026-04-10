# GPU Calculator

LLM inference benchmark dashboard and GPU cost estimator.

## Features

### Benchmark Results (`/`)

Interactive scatter plot comparing **Token Throughput per GPU** vs **End-to-End Latency** and **TTFT vs Context Length** across different hardware and software configurations.

- Multi-select filters for GPU, model, engine, and parallelism config
- Log scale toggle for better visualization of wide data ranges
- Pareto frontier overlay highlighting optimal throughput-latency tradeoffs
- Shareable URLs — filter state is encoded in query params

![Benchmark Results](benchmark-results.png)

### GPU Estimator (`/gpu-estimator`)

Estimate GPU count, VRAM usage, and latency for LLM inference.

![GPU Calculator](gpu-calculator.png)

**Inputs:** HuggingFace `config.json` link (auto-detects architecture, MoE, KV heads), model size, precision (FP32/BF16/FP8/INT8/INT4), KV cache precision, context length (2K–128K), request shape, concurrency (1–256), target GPU.

**Outputs:** GPU count (power-of-2 for TP), TP/DP recommendation, VRAM breakdown (weights, KV cache, activations).

**Latency estimation** is shown two ways:

| Method | How it works |
|--------|-------------|
| Data-driven | Log-linear interpolation from real benchmarks, by architecture type (MoE vs dense) |
| Theoretical | `TTFT = 2 * active_params * input_tokens / (GPUs * TFLOPS * 35%)`, `TPOT = model_bytes / (GPUs * BW * 65%)` |

## Benchmark Coverage

| Engine | GPUs | Models |
|--------|------|--------|
| vLLM | B200, H100 SXM, H200 SXM | GLM-4.7, GLM-4.7-FP8, GPT-OSS-120B, Llama-3.1-70B, Qwen3-8B/14B/32B |
| SGLang | A100 SXM, H100 SXM, H200 SXM | GPT-OSS-120B, Llama-3.1-70B, Qwen3-8B/14B, Qwen3.5-27B/35B/122B |

Data sourced from [llm-benchmaq](https://github.com/Scicom-AI-Enterprise-Organization/llm-benchmaq) (git submodule at `data/llm-benchmaq/`).

## Setup

```bash
git clone --recurse-submodules https://github.com/Scicom-AI-Enterprise-Organization/GPUCalculator.git
cd GPUCalculator
npm install
npm run dev
```

If already cloned without submodules:

```bash
git submodule update --init --remote
```

### Docker

```bash
docker compose up --build
```

## Tech Stack

Next.js 16, React 19, Recharts, Tailwind CSS v4, Radix UI
