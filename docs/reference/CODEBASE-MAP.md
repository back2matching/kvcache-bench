# Codebase Map

Every file in the repo, what it does, and key functions.

## Root Files

| File | Lines | What |
|------|-------|------|
| `README.md` | 132 | PyPI readme. Install, usage, example output, research summary. |
| `CLAUDE.md` | ~60 | Dev instructions for Claude Code. |
| `pyproject.toml` | 34 | Package config. Name: kvcache-bench. Deps: requests, pynvml. Entry point: `kvcache_bench.cli:main`. |
| `LICENSE` | 16 | Apache 2.0 |
| `.gitignore` | 7 | Ignores pycache, eggs, dist, build, pytest_cache, results*.json |
| `results_rtx4080.json` | 155 | Benchmark data: 3 KV types x 3 prompts at 4096 context. Qwen3-8B on RTX 4080. (gitignored but tracked) |
| `results_context_sweep.json` | 308 | Context sweep data: 3 KV types x 2 prompts x 3 context lengths (4K/8K/16K). |
| `chart_speed_comparison.png` | -- | Bar chart: avg generation speed per KV type (f16: 84.1, q4_0: 87.2, q8_0: 86.8 tok/s) |
| `chart_vram_by_context.png` | -- | Line chart: VRAM delta vs context length per KV type |
| `chart_vram_savings.png` | -- | Bar chart: VRAM saved vs f16 (q4_0: 160 MB, q8_0: 97 MB at 16K context) |

## Package: `kvcache_bench/`

### `__init__.py` (11 lines)

Package docstring and `__version__ = "0.1.0"`.

### `bench.py` (257 lines)

Core benchmark engine. Orchestrates the full test matrix.

| Function/Class | Lines | What |
|----------------|-------|------|
| `BenchResult` | 17-33 | Dataclass. All fields for one benchmark measurement: model, kv_type, context_length, speeds, VRAM, quality. |
| `set_kv_cache_env()` | 35-40 | Sets `OLLAMA_KV_CACHE_TYPE` env var. Note: mixed K/V not yet supported by Ollama. |
| `restart_ollama_with_kv()` | 43-94 | Kill Ollama, set env vars, start `ollama serve`, poll until ready (up to 30s). Platform-aware (Windows taskkill vs Linux pkill). |
| `check_quality()` | 97-112 | Auto-grade responses. Strips `<think>` tags (Qwen3.5). Checks: "paris" for short, "def"+"return" for code, "9" for reasoning. |
| `run_single_bench()` | 115-181 | Run one measurement. Starts VramTracker, calls Ollama API, extracts timing from response, returns BenchResult. |
| `run_full_benchmark()` | 184-238 | Main loop. Iterates KV types x context lengths x prompts. Restarts Ollama per KV type, warms up model, runs tests. |
| `format_results_table()` | 241-257 | Formats list of BenchResult into a markdown table string. |

### `cli.py` (165 lines)

CLI entry point. Registered as `kvcache-bench` command via pyproject.toml.

| Function | Lines | What |
|----------|-------|------|
| `main()` | 12-164 | Argument parsing (argparse), dispatch to GPU info / list models / run benchmark. Handles JSON output, chart generation, and the recommendation engine. |

Key CLI flags:
- `--model` / `-m` -- Ollama model name
- `--context` / `-c` -- Comma-separated context lengths (default: 4096)
- `--types` / `-t` -- KV types to test (default: f16,q8_0,q4_0)
- `--prompts` / `-p` -- Prompt types (short, code, reasoning, long, tool_call)
- `--json` / `-j` -- Save results to JSON file
- `--charts` -- Generate PNG charts (needs matplotlib)
- `--no-restart` -- Use current Ollama KV config
- `--list-models` -- List available Ollama models
- `--gpu` -- Show GPU info and exit

Recommendation logic (lines 120-161): Averages speed per KV type, compares to f16 baseline. Recommends q8_0 if speed loss < 10%, otherwise q4_0.

### `gpu.py` (80 lines)

NVIDIA GPU detection and VRAM monitoring.

| Function/Class | Lines | What |
|----------------|-------|------|
| `GpuInfo` | 9-15 | Dataclass: name, vram_total_mb, vram_used_mb, vram_free_mb, driver_version. |
| `detect_gpu()` | 18-39 | Calls `nvidia-smi --query-gpu=...` via subprocess. Returns GpuInfo or None. |
| `measure_vram()` | 42-51 | Quick VRAM reading. Calls `nvidia-smi --query-gpu=memory.used`. Returns int MB. |
| `VramTracker` | 54-79 | Class for tracking VRAM over a time window. `start()` records baseline, `stop()` records peak and returns dict with baseline_mb, peak_mb, delta_mb, samples count. |

### `ollama.py` (136 lines)

Ollama API integration.

| Function/Constant | Lines | What |
|-------------------|-------|------|
| `OLLAMA_BASE` | 10 | `http://localhost:11434` |
| `KV_TYPES` | 13 | `["f16", "q8_0", "q4_0"]` |
| `MIXED_KV` | 16-22 | List of (K, V) type tuples for future mixed testing. |
| `OllamaResult` | 25-36 | Dataclass (not currently used by bench.py, kept for future). |
| `check_ollama()` | 39-45 | GET `/` with 3s timeout. Returns bool. |
| `list_models()` | 48-55 | GET `/api/tags`. Returns list of model name strings. |
| `run_inference()` | 58-84 | POST `/api/generate`. Non-streaming. Returns raw JSON dict with timing fields. |
| `run_chat()` | 87-112 | POST `/api/chat`. For tool_call testing. Returns raw JSON dict. |
| `BENCH_PROMPTS` | 116-122 | Dict of 5 standard benchmark prompts (short, code, long, reasoning, tool_call). |
| `BENCH_TOOL` | 124-135 | Tool schema for `get_weather` function (tool_call benchmark). |

### `charts.py` (125 lines)

Chart generation using matplotlib.

| Function | Lines | What |
|----------|-------|------|
| `generate_charts()` | 8-111 | Reads JSON results file, generates 3 PNG charts. Groups data by context length and KV type. Colors: f16 red, q8_0 blue, q4_0 green. |
| `__main__` block | 114-125 | Direct execution: takes JSON path as arg, or tries default paths. |

Charts produced:
1. `chart_vram_by_context.png` -- Line chart, VRAM delta vs context length. Only generated when multiple context lengths exist.
2. `chart_speed_comparison.png` -- Bar chart, average generation speed per KV type.
3. `chart_vram_savings.png` -- Bar chart, VRAM saved vs f16 baseline.

## Other Directories

| Path | What |
|------|------|
| `tests/` | Empty. No formal tests (benchmark tool, not a library). |
| `dist/` | Build artifacts (gitignored). |
| `kvcache_bench.egg-info/` | Editable install metadata (gitignored). |
| `.pytest_cache/` | Pytest cache (gitignored). |
| `docs/` | This documentation. |

## Dependency Graph

```
cli.py
  +-- bench.py
  |     +-- gpu.py (detect_gpu, measure_vram, VramTracker)
  |     +-- ollama.py (check_ollama, list_models, run_inference, run_chat, BENCH_PROMPTS, BENCH_TOOL, KV_TYPES, MIXED_KV)
  +-- gpu.py (detect_gpu)
  +-- ollama.py (check_ollama, list_models, KV_TYPES)
  +-- charts.py (generate_charts) -- optional, lazy import

External deps: requests, pynvml, matplotlib (optional)
```
