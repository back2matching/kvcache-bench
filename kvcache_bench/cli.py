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

            # Smart recommendation based on actual data
            print(f"\n{'─'*50}")
            print("RECOMMENDATION")
            print(f"{'─'*50}")

            by_type = {}
            for r in valid:
                if r.kv_type not in by_type:
                    by_type[r.kv_type] = []
                by_type[r.kv_type].append(r)

            f16_speed = sum(r.eval_rate for r in by_type.get('f16', valid)) / max(len(by_type.get('f16', valid)), 1)
            f16_vram = max((r.vram_delta_mb for r in by_type.get('f16', valid)), default=0)

            best = None
            for kv in ['q8_0', 'q4_0']:
                if kv not in by_type:
                    continue
                kv_speed = sum(r.eval_rate for r in by_type[kv]) / len(by_type[kv])
                kv_vram = max((r.vram_delta_mb for r in by_type[kv]), default=0)
                speed_loss = (f16_speed - kv_speed) / f16_speed * 100 if f16_speed > 0 else 0
                vram_saved = f16_vram - kv_vram

                if kv == 'q8_0' and speed_loss < 10:
                    best = kv
                    print(f"\n  Use q8_0 (8-bit KV cache)")
                    print(f"  Speed: {kv_speed:.0f} tok/s ({speed_loss:+.1f}% vs f16)")
                    print(f"  VRAM: saves {vram_saved} MB vs f16 (2x compression)")
                    print(f"  Quality: negligible loss (+0.004 perplexity)")
                    print(f"\n  Set: OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_FLASH_ATTENTION=1")
                    break
                elif kv == 'q4_0':
                    best = kv
                    print(f"\n  Use q4_0 (4-bit KV cache)")
                    print(f"  Speed: {kv_speed:.0f} tok/s ({speed_loss:+.1f}% vs f16)")
                    print(f"  VRAM: saves {vram_saved} MB vs f16 (4x compression)")
                    print(f"  Quality: noticeable at long context (+0.2 perplexity)")
                    print(f"\n  Set: OLLAMA_KV_CACHE_TYPE=q4_0 OLLAMA_FLASH_ATTENTION=1")

            if not best:
                print("\n  Keep f16 (no clear benefit from compression on this test)")


if __name__ == "__main__":
    main()
