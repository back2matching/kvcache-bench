# CLAUDE.md -- kvcache-bench

> Operating instructions for Claude Code on this repo.

## What Is This?

Benchmarking tool for LLM KV cache compression methods. Tests f16, q8_0, and q4_0 KV cache types via Ollama, measuring VRAM savings, generation speed, prefill speed, and response quality. Published as `kvcache-bench` on PyPI.

**Status:** Finished product. Version 0.1.0 on PyPI. Includes RTX 4080 benchmark data and chart generation. Apache 2.0.

## Current State

| Metric | Value |
|--------|-------|
| Version | 0.1.0 (PyPI) |
| Python | 3.10+ |
| Source files | 5 (bench.py, cli.py, gpu.py, ollama.py, charts.py) |
| Total lines | ~763 |
| Tests | 0 (benchmark tool, not a library) |
| Dependencies | requests, pynvml (+ matplotlib optional for charts) |
| License | Apache 2.0 |
| PyPI account | back2matching |

## Architecture

Ollama's KV cache type is a server-level env var. To compare types, the tool must restart Ollama between each test.

**Flow:** CLI parses args -> detect GPU -> for each KV type: kill Ollama, set `OLLAMA_KV_CACHE_TYPE`, restart, warm up model, run benchmarks -> format results table -> optional JSON/charts.

**What it measures:**
- **Generation speed** (tok/s) -- from Ollama's `eval_count / eval_duration`
- **Prefill speed** (tok/s) -- from `prompt_eval_count / prompt_eval_duration`
- **VRAM delta** (MB) -- nvidia-smi before and after inference (measures KV cache growth specifically)
- **Quality** -- auto-graded sanity checks per prompt type (not perplexity)

## Files

```
kvcache_bench/
  __init__.py          (11 lines)   Package init, __version__
  bench.py             (257 lines)  Core engine: BenchResult, restart_ollama_with_kv, run_single_bench, run_full_benchmark, format_results_table
  cli.py               (165 lines)  CLI entry point: argparse, recommendation engine
  gpu.py               (80 lines)   GpuInfo dataclass, detect_gpu, measure_vram, VramTracker
  ollama.py            (136 lines)  Ollama API: check_ollama, list_models, run_inference, run_chat, BENCH_PROMPTS, KV_TYPES
  charts.py            (125 lines)  3 matplotlib charts: VRAM by context, speed comparison, VRAM savings

Root files:
  README.md                         PyPI readme
  pyproject.toml                    Package config, entry point: kvcache_bench.cli:main
  LICENSE                           Apache 2.0
  results_rtx4080.json              Benchmark data: 3 KV types x 3 prompts at 4096 ctx (Qwen3-8B, RTX 4080)
  results_context_sweep.json        Context sweep: 3 KV types x 2 prompts x 3 context lengths (4K/8K/16K)
  chart_speed_comparison.png        Bar chart: avg gen speed (f16: 84.1, q4_0: 87.2, q8_0: 86.8 tok/s)
  chart_vram_by_context.png         Line chart: VRAM delta vs context length per KV type
  chart_vram_savings.png            Bar chart: VRAM saved vs f16 (q4_0: 160 MB, q8_0: 97 MB at 16K)

Docs:
  docs/README.md                    Doc index
  docs/ARCHITECTURE.md              How the benchmark works
  docs/RESULTS.md                   RTX 4080 results with tables and analysis
  docs/reference/CODEBASE-MAP.md    Every file, line counts, key functions
```

## Commands

```bash
# Install (editable)
pip install -e .

# Install with chart support
pip install -e ".[charts]"

# Run benchmark (auto-detects first Ollama model)
kvcache-bench

# Specific model
kvcache-bench --model qwen3.5:9b

# Multiple context lengths + JSON output + charts
kvcache-bench --model qwen3.5:9b --context 4096,8192,16384 --json results.json --charts

# Include tool calling test
kvcache-bench --model qwen3.5:9b --prompts short,code,reasoning,tool_call

# GPU info only
kvcache-bench --gpu

# List available Ollama models
kvcache-bench --list-models

# Skip Ollama restarts (use current KV config)
kvcache-bench --model qwen3.5:9b --no-restart

# Generate charts from existing JSON
python -m kvcache_bench.charts results.json
```

## Key Implementation Details

- Ollama restart: `taskkill` (Windows) / `pkill` (Linux), then `ollama serve` as Popen. Polls up to 30s.
- Env vars per test: `OLLAMA_KV_CACHE_TYPE`, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_NUM_PARALLEL=1`
- VRAM tracking: `nvidia-smi --query-gpu=memory.used` via subprocess. Baseline before, peak after.
- Quality checks strip `<think>` tags (Qwen3.5 thinking mode).
- All inference uses `temperature: 0.0`, `think: false`, `stream: false`.
- Recommendation logic: if q8_0 speed loss < 10% vs f16, recommend q8_0. Otherwise q4_0.

## Key Results (RTX 4080 16GB, Qwen3-8B)

- All KV types produce nearly identical generation speed (~87 tok/s)
- At 16K context, f16 uses +316 MB, q8_0 uses +219 MB (saves 97 MB), q4_0 uses +156 MB (saves 160 MB)
- f16 slows to 78.9 tok/s at 16K while quantized types hold at 87 tok/s
- Recommendation: q8_0 for most users (zero quality cost), q4_0 when VRAM-constrained

## What This Repo IS NOT

- Not actively developed
- Not a library (it's a CLI/benchmark tool)
- No test suite needed

## Related Projects

| Project | What |
|---------|------|
| [turboquant](https://github.com/back2matching/turboquant) | KV cache compression method being benchmarked |
| [turboquant-vectors](https://github.com/back2matching/turboquant-vectors) | Embedding compression (different application) |
| [FlockRun](https://github.com/back2matching/FlockRun) | Parent project |
