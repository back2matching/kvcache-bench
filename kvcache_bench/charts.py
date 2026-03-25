"""Chart generation for kvcache-bench results."""

import json
from pathlib import Path
from typing import Optional


def generate_charts(results_path: str, output_dir: Optional[str] = None):
    """Generate comparison charts from benchmark results JSON."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with open(results_path) as f:
        results = json.load(f)

    if output_dir is None:
        output_dir = str(Path(results_path).parent)

    # Group by context length
    by_ctx = {}
    for r in results:
        ctx = r['context_length']
        if ctx not in by_ctx:
            by_ctx[ctx] = {}
        kv = r['kv_type']
        if kv not in by_ctx[ctx]:
            by_ctx[ctx][kv] = {'vram': [], 'speed': [], 'prefill': []}
        by_ctx[ctx][kv]['vram'].append(r['vram_delta_mb'])
        by_ctx[ctx][kv]['speed'].append(r['eval_rate'])
        by_ctx[ctx][kv]['prefill'].append(r['prompt_eval_rate'])

    kv_types = sorted(set(r['kv_type'] for r in results))
    contexts = sorted(by_ctx.keys())
    colors = {'f16': '#e74c3c', 'q8_0': '#3498db', 'q4_0': '#2ecc71', 'tq4_0': '#9b59b6'}

    # --- Chart 1: VRAM by context length ---
    if len(contexts) > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        for kv in kv_types:
            vram_avgs = []
            for ctx in contexts:
                if kv in by_ctx[ctx]:
                    vals = by_ctx[ctx][kv]['vram']
                    vram_avgs.append(max(vals) if vals else 0)
                else:
                    vram_avgs.append(0)
            ax.plot(contexts, vram_avgs, 'o-', label=kv, color=colors.get(kv, '#95a5a6'), linewidth=2, markersize=8)

        ax.set_xlabel('Context Length (tokens)', fontsize=12)
        ax.set_ylabel('VRAM Delta (MB)', fontsize=12)
        ax.set_title('KV Cache VRAM Usage by Context Length', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.set_xticks(contexts)
        ax.set_xticklabels([f'{c//1024}K' for c in contexts])
        plt.tight_layout()
        chart_path = Path(output_dir) / 'chart_vram_by_context.png'
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"  Saved: {chart_path}")

    # --- Chart 2: Speed comparison bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = range(len(kv_types))
    speeds = []
    for kv in kv_types:
        all_speeds = [r['eval_rate'] for r in results if r['kv_type'] == kv and r['eval_rate'] > 0]
        speeds.append(sum(all_speeds) / len(all_speeds) if all_speeds else 0)

    bars = ax.bar(x_pos, speeds, color=[colors.get(kv, '#95a5a6') for kv in kv_types], width=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(kv_types, fontsize=12)
    ax.set_ylabel('Generation Speed (tok/s)', fontsize=12)
    ax.set_title('Average Generation Speed by KV Cache Type', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, speed in zip(bars, speeds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{speed:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    chart_path = Path(output_dir) / 'chart_speed_comparison.png'
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"  Saved: {chart_path}")

    # --- Chart 3: VRAM savings summary ---
    fig, ax = plt.subplots(figsize=(8, 5))
    f16_vram = max((r['vram_delta_mb'] for r in results if r['kv_type'] == 'f16'), default=1)
    savings = []
    for kv in kv_types:
        kv_vram = max((r['vram_delta_mb'] for r in results if r['kv_type'] == kv), default=0)
        savings.append(max(0, f16_vram - kv_vram))

    bars = ax.bar(x_pos, savings, color=[colors.get(kv, '#95a5a6') for kv in kv_types], width=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(kv_types, fontsize=12)
    ax.set_ylabel('VRAM Saved vs FP16 (MB)', fontsize=12)
    ax.set_title('VRAM Savings by KV Cache Type', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, s in zip(bars, savings):
        if s > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{s:.0f} MB', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    chart_path = Path(output_dir) / 'chart_vram_savings.png'
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"  Saved: {chart_path}")

    print(f"\n  {len(kv_types)} KV types, {len(contexts)} context lengths, {len(results)} total measurements")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        generate_charts(sys.argv[1])
    else:
        # Try default paths
        for p in ['results_context_sweep.json', 'results_rtx4080.json']:
            if Path(p).exists():
                print(f"Generating charts from {p}:")
                generate_charts(p)
                break
