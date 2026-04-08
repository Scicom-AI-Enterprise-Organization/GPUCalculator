# GPU Calculator

LLM inference benchmark dashboard and GPU estimator.

## Benchmark Results

Interactive scatter plot of Token Throughput per GPU vs End-to-End Latency. Filter by GPU type, model, inference engine (vLLM / SGLang), and parallelism config.

![Benchmark Results](benchmark-results.png)

## GPU Estimator

Estimate the number of GPUs needed for LLM inference.

![GPU Calculator](gpu-calculator.png)

### Inputs

- **HuggingFace Config** — Paste a link to a model's `config.json` for precise KV cache estimation using actual layer count, KV heads, and head dimensions. Auto-detects MoE architecture.
- **Model Size** — Total parameter count (1B–1000B)
- **MoE** — Toggle for Mixture of Experts models. Configure total experts, active experts per token, and shared experts. All weights stay in VRAM but activation overhead is based on active parameters only.
- **Precision** — Model weight precision: FP32, BF16/FP16, FP8, INT8, INT4/GPTQ/AWQ
- **KV Cache Precision** — Separate from model precision: BF16/FP16 (default) or FP8
- **Context Length** — 2K to 128K tokens
- **Concurrent Requests** — 1 to 256
- **Target GPU** — B200, H200 SXM, H100 SXM, A100 SXM/PCIe, L40S, A10, RTX 4090

### Outputs

- Estimated GPU count (rounded to power of 2 for tensor parallelism)
- TP/DP parallelism recommendation
- VRAM breakdown: model weights, KV cache, activation overhead
- Full calculation reference with formulas

## Benchmark Data

Sourced from [llm-benchmaq](https://github.com/Scicom-AI-Enterprise-Organization/llm-benchmaq) as a git submodule under `data/llm-benchmaq/`.

```bash
# sync latest data
git submodule update --remote
```

## Running Locally

```bash
npm install
npm run dev
```

## Docker

```bash
docker compose up --build
```

## Tech Stack

- [Next.js 16](https://nextjs.org), [React 19](https://react.dev), [Recharts](https://recharts.org), [Tailwind CSS v4](https://tailwindcss.com), [Radix UI](https://www.radix-ui.com)
