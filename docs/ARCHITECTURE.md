# Architecture

How kvcache-bench measures KV cache compression tradeoffs.

## The Problem

Ollama and llama.cpp support multiple KV cache quantization types (f16, q8_0, q4_0), but the KV cache type is a **server-level** setting. You can't switch it per-request. To compare types, you must restart Ollama with different environment variables each time. Nobody does this systematically.

kvcache-bench automates the full cycle: kill Ollama, set env vars, restart, warm up, benchmark, repeat.

## Benchmark Flow

```
CLI parses args
  |
  v
detect_gpu() -- nvidia-smi query for GPU name, VRAM, driver
  |
  v
For each KV type (f16, q8_0, q4_0):
  |
  +-- restart_ollama_with_kv(kv_type)
  |     |-- Kill Ollama (taskkill on Windows, pkill on Linux)
  |     |-- Wait 3 seconds
  |     |-- Set OLLAMA_KV_CACHE_TYPE, OLLAMA_FLASH_ATTENTION=1, OLLAMA_NUM_PARALLEL=1
  |     |-- Start `ollama serve` as background process
  |     |-- Poll check_ollama() up to 30 times (1s each)
  |     +-- Warm up: run_inference("Hi", num_ctx=512, max_tokens=1)
  |
  +-- For each context length (default: 4096):
  |     |
  |     +-- For each prompt type (short, code, reasoning):
  |           |
  |           +-- run_single_bench()
  |                 |-- VramTracker.start() -- record VRAM baseline via nvidia-smi
  |                 |-- run_inference() or run_chat() via Ollama API
  |                 |-- VramTracker.stop() -- record VRAM peak
  |                 |-- Extract timing from Ollama response fields
  |                 |-- check_quality() -- auto-grade the response
  |                 +-- Return BenchResult dataclass
  |
  v
format_results_table() -- markdown table to stdout
  |
  v
Optional: save JSON, generate charts, print recommendation
```

## What It Measures

### Generation Speed (eval_rate)

Tokens per second during text generation. Extracted from Ollama's `eval_count / eval_duration` response fields. This is the number users care about most.

### Prefill Speed (prompt_eval_rate)

Tokens per second processing the input prompt. From `prompt_eval_count / prompt_eval_duration`. Matters for long-context workloads.

### VRAM Delta

Extra VRAM consumed beyond the model's baseline. Measured by calling `nvidia-smi --query-gpu=memory.used` before and during inference. The `VramTracker` class records baseline on start and peak on stop. Delta = peak - baseline.

This measures KV cache memory specifically, since model weights are already loaded during the warm-up phase.

### Quality

Simple auto-graders per prompt type:

| Prompt Type | Check |
|-------------|-------|
| `short` | Response contains "paris" (case-insensitive) |
| `code` | Response contains `def ` and `return` |
| `reasoning` | Response contains "9" |
| `tool_call` | Checks for tool_calls in response |

Not a perplexity measurement. Just a sanity check that compression didn't break generation.

## Ollama Restart Mechanism

The key design constraint: Ollama reads `OLLAMA_KV_CACHE_TYPE` at startup and applies it globally. There's no per-request override.

```
restart_ollama_with_kv("q8_0"):
  1. Kill existing: taskkill /F /IM ollama.exe (Windows) or pkill -f ollama (Linux)
  2. Sleep 3s
  3. Set env: OLLAMA_KV_CACHE_TYPE=q8_0, OLLAMA_FLASH_ATTENTION=1, OLLAMA_NUM_PARALLEL=1
  4. Start: ollama serve (Popen, detached)
  5. Poll: check_ollama() hits http://localhost:11434/ up to 30 times
  6. Warm up: run a 1-token inference to force model load
  7. Sleep 2s to let VRAM settle
```

`OLLAMA_FLASH_ATTENTION=1` is always set because KV cache quantization requires flash attention. `OLLAMA_NUM_PARALLEL=1` keeps benchmarks consistent.

## VRAM Tracking

The `VramTracker` class calls `nvidia-smi` via subprocess. It takes a baseline reading before inference and a peak reading after.

Limitations:
- Only works with NVIDIA GPUs (nvidia-smi dependency)
- Sampling granularity is coarse (before/after, not continuous)
- Other processes using the GPU can affect readings
- The warm-up phase loads model weights, so the delta during benchmarking reflects KV cache growth specifically

## Ollama API Integration

Two endpoints:

- `/api/generate` -- for standard prompts (short, code, reasoning, long). Uses `stream: false` to get full timing in one response.
- `/api/chat` -- for tool_call prompts. Same pattern but with messages array and tools parameter.

Both set `temperature: 0.0` and `think: false` for reproducibility. The response includes `prompt_eval_count`, `prompt_eval_duration`, `eval_count`, and `eval_duration` in nanoseconds.

## Standard Prompts

Five built-in prompts, each testing a different capability:

| Type | Prompt | Tokens |
|------|--------|--------|
| `short` | "What is the capital of France? Answer in one word." | ~22 |
| `code` | "Write a Python function to check if a number is prime. Just the function, no explanation." | ~29 |
| `reasoning` | "A farmer has 17 sheep. All but 9 die. How many sheep are left? Think step by step." | ~35 |
| `long` | "You are an expert software engineer. " x100 + analysis prompt | ~600+ |
| `tool_call` | "Get the current weather in Tokyo." (with tool schema) | ~22 |

Default runs use `short`, `code`, and `reasoning`. The `long` and `tool_call` prompts are opt-in via `--prompts`.

## Chart Generation

Three charts generated from JSON results (requires matplotlib):

1. **chart_vram_by_context.png** -- Line chart. X-axis: context length (4K, 8K, 16K). Y-axis: VRAM delta in MB. One line per KV type. Shows how VRAM scales with context and how compression flattens the curve.

2. **chart_speed_comparison.png** -- Bar chart. One bar per KV type showing average generation speed in tok/s. Demonstrates that quantized KV caches don't slow generation.

3. **chart_vram_savings.png** -- Bar chart. VRAM saved vs f16 baseline in MB. Shows the absolute VRAM reduction from each compression level.

## Recommendation Engine

After benchmarking, the CLI prints a recommendation. Logic:
1. Average generation speed per KV type across all runs
2. Max VRAM delta per KV type
3. If q8_0 speed loss < 10% vs f16: recommend q8_0
4. Otherwise: recommend q4_0
5. If neither helps: keep f16

The recommendation includes the exact `OLLAMA_KV_CACHE_TYPE=<type> OLLAMA_FLASH_ATTENTION=1` env vars to set.
