"""CLI entry point: kvcache-bench command."""

import argparse
import json
import sys
from pathlib import Path
from kvcache_bench.gpu import detect_gpu
from kvcache_bench.ollama import check_ollama, list_models, KV_TYPES
from kvcache_bench.bench import run_full_benchmark, format_results_table


def main():
    parser = argparse.ArgumentParser(
        prog="kvcache-bench",
        description="Benchmark every KV cache compression method on your GPU.",
    )
    parser.add_argument("--model", "-m", help="Ollama model name (e.g., qwen3.5:9b)")
    parser.add_argument("--context", "-c", default="4096",
                        help="Context lengths, comma-separated (default: 4096)")
    parser.add_argument("--types", "-t", default=",".join(KV_TYPES),
                        help=f"KV cache types to test (default: {','.join(KV_TYPES)})")
    parser.add_argument("--prompts", "-p", default="short,code,reasoning",
                        help="Prompt types: short,code,reasoning,long,tool_call")
    parser.add_argument("--json", "-j", help="Save results to JSON file")
    parser.add_argument("--charts", action="store_true", help="Generate comparison charts (requires matplotlib)")
    parser.add_argument("--no-restart", action="store_true",
                        help="Don't restart Ollama between tests (use current KV type)")
    parser.add_argument("--list-models", action="store_true", help="List available Ollama models")
    parser.add_argument("--gpu", action="store_true", help="Show GPU info and exit")
    args = parser.parse_args()

    # GPU info
    if args.gpu:
        gpu = detect_gpu()
        if gpu:
            print(f"GPU: {gpu.name}")
            print(f"VRAM: {gpu.vram_used_mb}/{gpu.vram_total_mb} MB ({gpu.vram_free_mb} MB free)")
            print(f"Driver: {gpu.driver_version}")
        else:
            print("No NVIDIA GPU detected.")
        return

    # Check Ollama
    if not check_ollama():
        print("Ollama is not running. Start it with: ollama serve")
        sys.exit(1)

    # List models
    if args.list_models:
        models = list_models()
        if models:
            print("Available models:")
            for m in models:
                print(f"  - {m}")
        else:
            print("No models found. Pull one with: ollama pull qwen3.5:9b")
        return

    # Need a model
    if not args.model:
        models = list_models()
        if models:
            args.model = models[0]
            print(f"No model specified, using: {args.model}")
        else:
            print("No model specified and no models available.")
            print("Usage: kvcache-bench --model qwen3.5:9b")
            sys.exit(1)

    # Parse args
    context_lengths = [int(c) for c in args.context.split(",")]
    kv_types = args.types.split(",")
    prompt_types = args.prompts.split(",")

    # Run
    from dataclasses import asdict
    results = run_full_benchmark(
        model=args.model,
        kv_types=kv_types,
        context_lengths=context_lengths,
        prompt_types=prompt_types,
        auto_restart=not args.no_restart,
    )

    # Output
    table = format_results_table(results)
    print(table)

    # Save JSON
    if args.json:
        out_path = Path(args.json)
        with open(out_path, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nResults saved to {out_path}")

    # Charts
    if args.charts and args.json:
        try:
            from kvcache_bench.charts import generate_charts
            print("\nGenerating charts...")
            generate_charts(args.json)
        except ImportError:
            print("\nCharts require matplotlib: pip install matplotlib")

    # Summary
    if results:
        gpu = detect_gpu()
        print(f"\nGPU: {gpu.name if gpu else 'Unknown'}")
        print(f"Model: {args.model}")
        print(f"Tests: {len(results)} ({len(kv_types)} types x {len(context_lengths)} contexts x {len(prompt_types)} prompts)")

        # Find best config
        valid = [r for r in results if not r.error and r.eval_rate > 0]
        if valid:
            fastest = max(valid, key=lambda r: r.eval_rate)
            lowest_vram = min(valid, key=lambda r: r.vram_delta_mb)
            print(f"\nFastest: {fastest.kv_type} ({fastest.eval_rate} tok/s)")
            print(f"Lowest VRAM: {lowest_vram.kv_type} (+{lowest_vram.vram_delta_mb} MB)")

            # Recommendation
            # q8_0 is usually the sweet spot (negligible quality loss, good VRAM savings)
            q8_results = [r for r in valid if r.kv_type == "q8_0"]
            if q8_results:
                q8_avg_rate = sum(r.eval_rate for r in q8_results) / len(q8_results)
                f16_results = [r for r in valid if r.kv_type == "f16"]
                if f16_results:
                    f16_avg_rate = sum(r.eval_rate for r in f16_results) / len(f16_results)
                    speed_diff = abs(q8_avg_rate - f16_avg_rate) / f16_avg_rate * 100
                    if speed_diff < 5:
                        print(f"\nRecommendation: Use q8_0. Near-zero speed difference ({speed_diff:.1f}%) with 2x VRAM savings.")


if __name__ == "__main__":
    main()
