"""
kvcache-bench: Benchmark every KV cache compression method on your GPU.

Usage:
    kvcache-bench --model qwen3.5:9b
    kvcache-bench --model llama3.1:8b --context 8192,16384,32768
    kvcache-bench --all-types --json results.json
"""

__version__ = "0.1.0"
