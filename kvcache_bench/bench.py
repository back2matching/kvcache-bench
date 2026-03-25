"""Core benchmark engine: runs comparison across KV cache types."""

import time
import json
import os
import subprocess
from dataclasses import dataclass, asdict, field
from typing import Optional
from kvcache_bench.gpu import detect_gpu, measure_vram, VramTracker
from kvcache_bench.ollama import (
    check_ollama, list_models, run_inference, run_chat,
    BENCH_PROMPTS, BENCH_TOOL, KV_TYPES, MIXED_KV,
)


@dataclass
class BenchResult:
    model: str
    kv_type: str  # "f16", "q8_0", "q4_0", or "q8_0/q4_0" for mixed
    context_length: int
    prompt_type: str
    prompt_tokens: int
    generated_tokens: int
    prompt_eval_rate: float  # tok/s
    eval_rate: float  # tok/s
    vram_baseline_mb: int
    vram_peak_mb: int
    vram_delta_mb: int
    total_time_s: float
    response_preview: str
    correct: Optional[bool] = None  # For quality checks
    error: Optional[str] = None


def set_kv_cache_env(kv_type_k: str, kv_type_v: str):
    """Set Ollama KV cache environment variables. Requires Ollama restart."""
    os.environ["OLLAMA_KV_CACHE_TYPE"] = kv_type_k
    # Note: Ollama doesn't support separate K/V types via env vars.
    # For mixed K/V testing, we'd need llama.cpp server directly.
    # For now, both K and V use the same type.


def restart_ollama_with_kv(kv_type: str) -> bool:
    """
    Restart Ollama with a new KV cache type.

    This is the key insight: Ollama's KV cache type is server-level,
    so we must restart between tests. We kill, set env, restart.
    """
    # On Windows, Ollama runs as a tray app or service
    try:
        # Kill existing Ollama
        if os.name == 'nt':
            subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], capture_output=True, timeout=5)
            subprocess.run(["taskkill", "/F", "/IM", "ollama app.exe"], capture_output=True, timeout=5)
        else:
            subprocess.run(["pkill", "-f", "ollama"], capture_output=True, timeout=5)

        time.sleep(3)

        # Set env vars
        os.environ["OLLAMA_FLASH_ATTENTION"] = "1"
        os.environ["OLLAMA_KV_CACHE_TYPE"] = kv_type
        os.environ["OLLAMA_NUM_PARALLEL"] = "1"  # Consistent for benchmarking

        # Start Ollama
        if os.name == 'nt':
            ollama_path = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
            if not os.path.exists(ollama_path):
                ollama_path = "ollama"  # Fall back to PATH
            subprocess.Popen(
                [ollama_path, "serve"],
                env={**os.environ},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.Popen(
                ["ollama", "serve"],
                env={**os.environ},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        # Wait for it to come up
        for _ in range(30):
            time.sleep(1)
            if check_ollama():
                return True
        return False

    except Exception as e:
        print(f"  Failed to restart Ollama: {e}")
        return False


def check_quality(prompt_type: str, response: str) -> Optional[bool]:
    """Simple quality checks for standard prompts."""
    # Strip thinking tags (Qwen3.5 uses <think>...</think>)
    import re
    clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    if not clean:
        clean = response  # Fallback to full response
    clean_lower = clean.lower().strip()

    if prompt_type == "short":
        return "paris" in clean_lower
    elif prompt_type == "reasoning":
        return "9" in clean
    elif prompt_type == "code":
        return "def " in clean and "return" in clean
    return None  # Can't auto-check


def run_single_bench(
    model: str,
    kv_type: str,
    prompt_type: str,
    context_length: int = 4096,
) -> BenchResult:
    """Run a single benchmark measurement."""
    prompt = BENCH_PROMPTS.get(prompt_type, prompt_type)

    tracker = VramTracker()
    tracker.start()

    t0 = time.perf_counter()

    if prompt_type == "tool_call":
        result = run_chat(
            model,
            [{"role": "user", "content": prompt}],
            num_ctx=context_length,
            max_tokens=100,
            tools=[BENCH_TOOL],
        )
    else:
        result = run_inference(model, prompt, num_ctx=context_length, max_tokens=150)

    elapsed = time.perf_counter() - t0
    vram = tracker.stop()

    if not result or "error" in result:
        return BenchResult(
            model=model, kv_type=kv_type, context_length=context_length,
            prompt_type=prompt_type, prompt_tokens=0, generated_tokens=0,
            prompt_eval_rate=0, eval_rate=0,
            vram_baseline_mb=vram["baseline_mb"], vram_peak_mb=vram["peak_mb"],
            vram_delta_mb=vram["delta_mb"], total_time_s=elapsed,
            response_preview="", error=result.get("error", "Unknown error"),
        )

    response_text = result.get("response", "")
    if not response_text and "message" in result:
        msg = result["message"]
        if isinstance(msg, dict):
            response_text = msg.get("content", "")
            if msg.get("tool_calls"):
                response_text = json.dumps(msg["tool_calls"][0]["function"])

    prompt_eval_count = result.get("prompt_eval_count", 0)
    eval_count = result.get("eval_count", 0)
    prompt_eval_dur = result.get("prompt_eval_duration", 1) / 1e9  # ns to s
    eval_dur = result.get("eval_duration", 1) / 1e9

    return BenchResult(
        model=model,
        kv_type=kv_type,
        context_length=context_length,
        prompt_type=prompt_type,
        prompt_tokens=prompt_eval_count,
        generated_tokens=eval_count,
        prompt_eval_rate=round(prompt_eval_count / prompt_eval_dur, 1) if prompt_eval_dur > 0 else 0,
        eval_rate=round(eval_count / eval_dur, 1) if eval_dur > 0 else 0,
        vram_baseline_mb=vram["baseline_mb"],
        vram_peak_mb=vram["peak_mb"],
        vram_delta_mb=vram["delta_mb"],
        total_time_s=round(elapsed, 2),
        response_preview=response_text[:200],
        correct=check_quality(prompt_type, response_text),
    )


def run_full_benchmark(
    model: str,
    kv_types: list[str] = None,
    context_lengths: list[int] = None,
    prompt_types: list[str] = None,
    auto_restart: bool = True,
) -> list[BenchResult]:
    """Run full benchmark matrix."""
    if kv_types is None:
        kv_types = KV_TYPES
    if context_lengths is None:
        context_lengths = [4096]
    if prompt_types is None:
        prompt_types = ["short", "code", "reasoning"]

    results = []
    gpu = detect_gpu()

    print(f"\n{'='*70}")
    print(f"KVCache-Bench v0.1.0")
    print(f"{'='*70}")
    if gpu:
        print(f"GPU: {gpu.name} ({gpu.vram_total_mb} MB VRAM)")
    print(f"Model: {model}")
    print(f"KV types: {', '.join(kv_types)}")
    print(f"Context lengths: {', '.join(str(c) for c in context_lengths)}")
    print(f"Prompts: {', '.join(prompt_types)}")
    print(f"{'='*70}\n")

    for kv_type in kv_types:
        print(f"\n--- KV type: {kv_type} ---")

        if auto_restart:
            print(f"  Restarting Ollama with OLLAMA_KV_CACHE_TYPE={kv_type}...")
            if not restart_ollama_with_kv(kv_type):
                print(f"  FAILED to restart Ollama. Skipping.")
                continue
            # Warm up: load model
            print(f"  Warming up model...")
            run_inference(model, "Hi", num_ctx=512, max_tokens=1)
            time.sleep(2)

        for ctx in context_lengths:
            for pt in prompt_types:
                print(f"  [{kv_type}] ctx={ctx}, prompt={pt}...", end=" ", flush=True)
                result = run_single_bench(model, kv_type, pt, ctx)
                results.append(result)

                if result.error:
                    print(f"ERROR: {result.error}")
                else:
                    quality = "?" if result.correct is None else ("PASS" if result.correct else "FAIL")
                    print(f"{result.eval_rate} tok/s, VRAM +{result.vram_delta_mb}MB, quality={quality}")

    return results


def format_results_table(results: list[BenchResult]) -> str:
    """Format results as a markdown table."""
    lines = []
    lines.append("")
    lines.append(f"| KV Type | Context | Prompt | Gen tok/s | Prefill tok/s | VRAM +MB | Quality |")
    lines.append(f"|---------|---------|--------|-----------|---------------|----------|---------|")

    for r in results:
        if r.error:
            lines.append(f"| {r.kv_type} | {r.context_length} | {r.prompt_type} | ERROR | - | - | {r.error[:30]} |")
        else:
            q = "?" if r.correct is None else ("PASS" if r.correct else "FAIL")
            lines.append(f"| {r.kv_type} | {r.context_length} | {r.prompt_type} | {r.eval_rate} | {r.prompt_eval_rate} | +{r.vram_delta_mb} | {q} |")

    lines.append("")
    return "\n".join(lines)
