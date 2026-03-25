# kvcache-bench

Benchmark every KV cache compression method on your GPU. One command, real numbers.

```
kvcache-bench --model qwen3.5:9b
```

```
| KV Type | Context | Prompt    | Gen tok/s | Prefill tok/s | VRAM +MB | Quality |
|---------|---------|-----------|-----------|---------------|----------|---------|
| f16     | 4096    | short     | 86.9      | 509.5         | +80      | PASS    |
| f16     | 16384   | short     | 78.8      | 784.6         | +316     | PASS    |
| q8_0    | 4096    | short     | 86.7      | 793.1         | +48      | PASS    |
| q8_0    | 16384   | short     | 87.2      | 741.9         | +219     | PASS    |
| q4_0    | 4096    | short     | 86.7      | 798.0         | +59      | PASS    |
| q4_0    | 16384   | short     | 86.7      | 522.7         | +156     | PASS    |

──────────────────────────────────────────────────
RECOMMENDATION
──────────────────────────────────────────────────

  Use q8_0 (8-bit KV cache)
  Speed: 87 tok/s (-0.1% vs f16)
  VRAM: saves 97 MB vs f16 (2x compression)
  Quality: negligible loss (+0.004 perplexity)

  Set: OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_FLASH_ATTENTION=1
```

*Real output from Qwen3.5-9B on RTX 4080 16GB.*

## Why

When you run a local LLM, the KV cache eats your VRAM. Ollama and llama.cpp support different KV cache quantization types (f16, q8_0, q4_0), but nobody tells you what the actual tradeoff is on YOUR hardware.

Current state of the world:
- You Google "ollama kv cache quantization" and find forum posts with conflicting advice
- You manually test each config, eyeball nvidia-smi, and guess
- No tool compares them systematically

kvcache-bench fixes this. It tests every KV cache type on your GPU and gives you a comparison table with speed, VRAM, and quality.

## Install

```bash
pip install kvcache-bench
```

## Usage

```bash
# Auto-detect your first model, test all KV types
kvcache-bench

# Specific model
kvcache-bench --model qwen3.5:9b

# Test at multiple context lengths (where KV savings matter most)
kvcache-bench --model llama3.1:8b --context 4096,8192,16384

# Include tool calling test
kvcache-bench --model qwen3.5:9b --prompts short,code,reasoning,tool_call

# Save results as JSON
kvcache-bench --model qwen3.5:9b --json results.json

# Just show GPU info
kvcache-bench --gpu

# List available models
kvcache-bench --list-models
```

## What It Tests

For each KV cache type (f16, q8_0, q4_0), it measures:

| Metric | How |
|--------|-----|
| **Generation speed** | Tokens per second during generation |
| **Prefill speed** | Tokens per second processing the prompt |
| **VRAM delta** | Extra VRAM used beyond model weights (measured via nvidia-smi) |
| **Quality** | Auto-checked against expected answers (Paris, code structure, reasoning) |

## How It Works

1. Detects your GPU and Ollama installation
2. For each KV cache type: restarts Ollama with `OLLAMA_KV_CACHE_TYPE=<type>`, warms up the model, runs benchmark prompts
3. Measures VRAM before and during inference via nvidia-smi
4. Extracts timing from Ollama's API response (prompt_eval_duration, eval_duration)
5. Checks response quality with simple auto-graders
6. Produces a markdown table (and optional JSON)

## What the Research Says

Based on llama.cpp community benchmarks and our testing:

| KV Type | VRAM Savings | Perplexity Impact | Best For |
|---------|-------------|-------------------|----------|
| f16 | Baseline | None | When you have VRAM to spare |
| q8_0 | 2x | +0.004 (negligible) | **Default recommendation.** Free VRAM, zero quality cost. |
| q4_0 | 4x | +0.2 (noticeable) | When you need max context length or are VRAM-constrained |

The sweet spot for most users: **q8_0**. Halves your KV cache VRAM with essentially zero quality loss.

## Requirements

- Python 3.10+
- NVIDIA GPU with nvidia-smi
- Ollama installed and running

## Roadmap

- [ ] Mixed K/V types (q8 keys + q4 values)
- [ ] Context length sweep charts
- [ ] HuggingFace backend (vLLM, TGI)
- [ ] TurboQuant integration
- [ ] Multi-model matrix
- [ ] HuggingFace Spaces leaderboard
- [ ] Community result submissions

## License

Apache 2.0

## Related

- [turboquant](https://github.com/back2matching/turboquant) -- TurboQuant KV cache compression (sub-4-bit)
- [NVIDIA kvpress](https://github.com/NVIDIA/kvpress) -- KV cache eviction/pruning methods
- [llama.cpp](https://github.com/ggml-org/llama.cpp) -- Where KV cache quantization lives
