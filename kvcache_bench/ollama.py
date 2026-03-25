"""Ollama backend: run inference with different KV cache types."""

import json
import time
import requests
from dataclasses import dataclass, field
from typing import Optional


OLLAMA_BASE = "http://localhost:11434"

# KV cache types supported by Ollama/llama.cpp
KV_TYPES = ["f16", "q8_0", "q4_0"]

# Mixed K/V configurations worth testing
MIXED_KV = [
    ("f16", "f16"),      # Baseline
    ("q8_0", "q8_0"),    # 8-bit both
    ("q4_0", "q4_0"),    # 4-bit both
    ("q8_0", "q4_0"),    # 8-bit keys, 4-bit values (sweet spot per research)
    ("q4_0", "q8_0"),    # 4-bit keys, 8-bit values (keys are more sensitive)
]


@dataclass
class OllamaResult:
    model: str
    kv_type_k: str
    kv_type_v: str
    context_length: int
    prompt_tokens: int
    generated_tokens: int
    prompt_eval_rate: float  # tok/s
    eval_rate: float  # tok/s
    total_duration_ms: float
    response_text: str = ""


def check_ollama() -> bool:
    """Check if Ollama is running."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def list_models() -> list[str]:
    """List available Ollama models."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        data = r.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def run_inference(
    model: str,
    prompt: str,
    num_ctx: int = 4096,
    max_tokens: int = 100,
    temperature: float = 0.0,
) -> Optional[dict]:
    """Run a single inference call via Ollama API."""
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_ctx": num_ctx,
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "think": False,
                },
            },
            timeout=300,
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def run_chat(
    model: str,
    messages: list[dict],
    num_ctx: int = 4096,
    max_tokens: int = 100,
    tools: Optional[list] = None,
) -> Optional[dict]:
    """Run a chat completion via Ollama API."""
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": max_tokens,
            "temperature": 0.0,
            "think": False,
        },
    }
    if tools:
        body["tools"] = tools
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/chat", json=body, timeout=300)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# Standard prompts for benchmarking
BENCH_PROMPTS = {
    "short": "What is the capital of France? Answer in one word.",
    "code": "Write a Python function to check if a number is prime. Just the function, no explanation.",
    "long": "You are an expert software engineer. " * 100 + "\n\nAnalyze the time complexity of binary search and explain why it's O(log n).",
    "reasoning": "A farmer has 17 sheep. All but 9 die. How many sheep are left? Think step by step.",
    "tool_call": "Get the current weather in Tokyo.",
}

BENCH_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string", "description": "City name"}},
            "required": ["location"],
        },
    },
}
