# GPU Calculator

LLM inference benchmark dashboard and GPU estimator.

## Benchmark Results

Interactive scatter plot of Token Throughput per GPU vs End-to-End Latency. Filter by GPU type, model, inference engine (vLLM / SGLang), and parallelism config.

![Benchmark Results](benchmark-results.png)

## GPU Estimator

Estimate GPU count, VRAM usage, and latency for LLM inference.

![GPU Calculator](gpu-calculator.png)

### Inputs

- **HuggingFace Config** — Paste a link to a model's `config.json` for precise KV cache estimation using actual layer count, KV heads, and head dimensions. Auto-detects MoE architecture.
- **Model Size** — Total parameter count (1B-1000B)
- **MoE** — Toggle for Mixture of Experts models. Configure total experts, active experts per token, and shared experts. All weights stay in VRAM but activation overhead and decode bandwidth are based on active parameters only.
- **Precision** — Model weight precision: FP32, BF16/FP16, FP8, INT8, INT4/GPTQ/AWQ
- **KV Cache Precision** — Separate from model precision: BF16/FP16 (default) or FP8
- **Context Length** — 2K to 128K tokens
- **Request Shape** — Input and output token counts for latency estimation
- **Concurrent Requests** — 1 to 256
- **Target GPU** — B200, H200 SXM, H100 SXM, A100 SXM/PCIe, L40S, A10, RTX 4090

### Outputs

- Estimated GPU count (rounded to power of 2 for tensor parallelism)
- TP/DP parallelism recommendation
- **Latency estimates** — TTFT (prefill, compute-bound), TPOT (decode, memory-bandwidth-bound), and E2E latency for a single request
- VRAM breakdown: model weights, KV cache, activation overhead
- Full calculation reference with formulas

### Latency Model

| Phase | Bound by | Formula |
|-------|----------|---------|
| TTFT (prefill) | Compute | `2 * active_params * input_tokens / (num_gpus * TFLOPS * 35%)` |
| TPOT (decode) | Memory BW | `active_model_bytes / (num_gpus * bandwidth * 65%)` |
| E2E | Both | `TTFT + output_tokens * TPOT` |

For MoE models, decode reads only active expert weights, reducing TPOT. Utilization factors are conservative single-request estimates — actual throughput improves with batching.

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
