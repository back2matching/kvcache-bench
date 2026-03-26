# kvcache-bench Documentation

## Documents

| Doc | What |
|-----|------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | How the benchmark works. Flow, measurements, Ollama restarts, quality checks. |
| [RESULTS.md](RESULTS.md) | Real benchmark data from RTX 4080 16GB with Qwen3-8B. Tables, analysis, charts. |
| [reference/CODEBASE-MAP.md](reference/CODEBASE-MAP.md) | Every file in the repo. What it does, line counts, key functions. |

## Quick Reference

```bash
# Install
pip install kvcache-bench

# Run benchmark (auto-detects first model)
kvcache-bench

# Specific model, multiple contexts, save JSON + charts
kvcache-bench --model qwen3.5:9b --context 4096,8192,16384 --json results.json --charts

# GPU info
kvcache-bench --gpu

# List models
kvcache-bench --list-models
```

## Project Links

- [PyPI: kvcache-bench](https://pypi.org/project/kvcache-bench/)
- [GitHub: back2matching/kvcache-bench](https://github.com/back2matching/kvcache-bench)
- [README (root)](../README.md)
- [CLAUDE.md (dev instructions)](../CLAUDE.md)
