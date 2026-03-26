# CLAUDE.md — kvcache-bench

> Operating instructions for Claude Code on this repo.

## What Is This?

Benchmarking tool for LLM KV cache compression methods. Measures VRAM savings, perplexity impact, and generation speed across FP16, Q8_0, Q4_0, and TurboQuant cache types.

**Status:** Published 0.1.0 on PyPI. Finished product — includes RTX 4080 benchmark data and chart generation. No further development planned.

## Current State

| Metric | Value |
|--------|-------|
| Version | 0.1.0 (PyPI) |
| Tests | 0 (benchmark tool, not library) |
| Dependencies | torch, transformers, matplotlib |

## Key Files

```
kvcache_bench/          — benchmark runner + chart generator
results_rtx4080.json    — real benchmark data from RTX 4080 16GB
results_context_sweep.json — context length sweep data
chart_*.png             — generated comparison charts
```

## What This Repo IS

- A finished benchmark tool for KV cache compression comparison
- Published data from RTX 4080 testing
- Reference for anyone comparing KV cache methods

## What This Repo IS NOT

- Not actively developed
- Not a library (it's a CLI/benchmark tool)

## Related Projects

| Project | What |
|---------|------|
| **turboquant** | The KV cache compression method being benchmarked |
| **turboquant-vectors** | Embedding compression (different application) |
| **FlockRun** | Parent project |

## Commands

```bash
pip install -e .
kvcache-bench run --model "TinyLlama/TinyLlama-1.1B" --cache-types fp16,q8_0,q4_0
kvcache-bench chart results.json
```

## PyPI

- Account: back2matching
- Package: kvcache-bench 0.1.0
