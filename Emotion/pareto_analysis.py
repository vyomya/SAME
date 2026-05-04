"""
pareto_analysis.py
==================
Reads all run_meta.json + trainer_state.json from checkpoint folders,
extracts UA, RTF, trainable params, training time, and plots:
  1. Compute-Performance Pareto frontier (RTF vs UA)
  2. Trainable params vs UA
  3. Training time vs UA
  4. Per-axis breakdowns (audio_len, lora_r, unfreeze_layers, model_size)

Usage:
  python pareto_analysis.py \
      --checkpoints_dir /scratch/zt1/project/msml605/user/mokshdag/checkpoints/checkpoints/ \
      --output_dir /scratch/zt1/project/msml605/user/mokshdag/pareto_results/
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # no display needed on cluster
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PARSE ARGS
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints_dir", type=str,
                   default="/scratch/zt1/project/msml605/user/mokshdag/checkpoints/checkpoints/")
    p.add_argument("--output_dir", type=str,
                   default="/scratch/zt1/project/msml605/user/mokshdag/pareto_results/")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# LOAD RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def load_results(checkpoints_dir):
    results = []
    for run_dir in sorted(Path(checkpoints_dir).iterdir()):
        if not run_dir.is_dir():
            continue

        meta_path = run_dir / "run_meta.json"
        if not meta_path.exists():
            print(f"  SKIP (no run_meta.json): {run_dir.name}")
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        # Get best eval UA from trainer_state.json
        best_ua = None
        best_acc = None
        trainer_state_path = run_dir / "trainer_state.json"
        if trainer_state_path.exists():
            with open(trainer_state_path) as f:
                state = json.load(f)
            uas = [x.get("eval_ua", 0) for x in state["log_history"] if "eval_ua" in x]
            accs = [x.get("eval_accuracy", 0) for x in state["log_history"] if "eval_accuracy" in x]
            if uas:
                best_ua = max(uas)
            if accs:
                best_acc = max(accs)

        # Get RTF from eval_results.json if available
        rtf = None
        eval_path = run_dir / "eval_results.json"
        if eval_path.exists():
            with open(eval_path) as f:
                eval_res = json.load(f)
            rtf = eval_res.get("rtf")
            if best_ua is None:
                best_ua = eval_res.get("ua", 0) / 100.0

        # Parse run name for axis labels
        name = run_dir.name
        audio_len = meta.get("max_audio_len", 6.0)
        lora_r    = meta.get("lora_r", None)
        mode      = meta.get("mode", "lora")
        model     = meta.get("model_name", "").split("/")[-1]
        trainable = meta.get("trainable_params", 0)
        total     = meta.get("total_params", 1)
        train_hrs = meta.get("training_hours", 0)

        # Parse unfreeze from name
        unfreeze = 4  # default
        if "unfreeze" in name:
            try:
                unfreeze = int(name.split("unfreeze")[-1])
            except:
                pass

        if best_ua is None:
            print(f"  SKIP (no UA found): {name}")
            continue

        results.append({
            "name":       name,
            "model":      model,
            "mode":       mode,
            "audio_len":  audio_len,
            "lora_r":     lora_r,
            "unfreeze":   unfreeze,
            "ua":         best_ua * 100 if best_ua <= 1.0 else best_ua,
            "accuracy":   best_acc * 100 if best_acc and best_acc <= 1.0 else (best_acc or 0),
            "rtf":        rtf,
            "trainable_M": trainable / 1e6,
            "total_M":    total / 1e6,
            "trainable_pct": 100 * trainable / max(total, 1),
            "train_hrs":  train_hrs,
        })
        print(f"  OK: {name} | UA={results[-1]['ua']:.1f}% | RTF={rtf}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PARETO FRONTIER
# ─────────────────────────────────────────────────────────────────────────────

def compute_pareto(points, x_key, y_key, minimize_x=True, maximize_y=True):
    """
    Find Pareto-optimal points.
    For compute-performance: minimize RTF (x), maximize UA (y).
    """
    pareto = []
    for p in points:
        dominated = False
        for q in points:
            if p == q:
                continue
            x_better = (q[x_key] <= p[x_key]) if minimize_x else (q[x_key] >= p[x_key])
            y_better = (q[y_key] >= p[y_key]) if maximize_y else (q[y_key] <= p[y_key])
            x_strict = (q[x_key] < p[x_key])  if minimize_x else (q[x_key] > p[x_key])
            y_strict = (q[y_key] > p[y_key])   if maximize_y else (q[y_key] < p[y_key])
            if x_better and y_better and (x_strict or y_strict):
                dominated = True
                break
        if not dominated:
            pareto.append(p)
    return sorted(pareto, key=lambda p: p[x_key])


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "audio_len":  {2.0: "#e74c3c", 4.0: "#2ecc71", 6.0: "#3498db"},
    "lora_r":     {8: "#9b59b6", 16: "#2ecc71", 32: "#e67e22", 64: "#e74c3c"},
    "unfreeze":   {0: "#e74c3c", 4: "#2ecc71", 8: "#3498db", 12: "#9b59b6"},
    "model":      {"wav2vec2-large-robust": "#2ecc71", "wav2vec2-base": "#e74c3c",
                   "whisper-small": "#3498db", "wavlm-large": "#9b59b6"},
}


def plot_pareto_frontier(results, output_dir):
    """Main Pareto plot: RTF vs UA with frontier highlighted."""
    # Only use results with RTF
    with_rtf = [r for r in results if r["rtf"] is not None]

    if not with_rtf:
        print("No RTF data available yet — run evaluate_checkpoint first")
        # Fall back to trainable_pct vs UA
        plot_trainable_vs_ua(results, output_dir)
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Color by audio_len
    for r in with_rtf:
        color = COLORS["audio_len"].get(r["audio_len"], "#95a5a6")
        ax.scatter(r["rtf"], r["ua"], color=color, s=120, zorder=5,
                   edgecolors="black", linewidths=0.5)
        ax.annotate(r["name"].replace("ser-", "").replace("-cremad", ""),
                    (r["rtf"], r["ua"]), textcoords="offset points",
                    xytext=(6, 4), fontsize=7, color="#333333")

    # Draw Pareto frontier
    pareto = compute_pareto(with_rtf, "rtf", "ua")
    if len(pareto) >= 2:
        px = [p["rtf"] for p in pareto]
        py = [p["ua"]  for p in pareto]
        ax.plot(px, py, "k--", linewidth=1.5, label="Pareto frontier", zorder=4)
        ax.scatter(px, py, color="gold", s=200, zorder=6,
                   edgecolors="black", linewidths=1.5, marker="*", label="Pareto optimal")

    # Legend for audio_len colors
    patches = [mpatches.Patch(color=c, label=f"audio={k}s")
               for k, c in COLORS["audio_len"].items()]
    ax.legend(handles=patches + ax.get_legend_handles_labels()[0][-2:],
              loc="lower right", fontsize=9)

    ax.set_xlabel("Real-Time Factor (RTF) — lower is faster", fontsize=12)
    ax.set_ylabel("Unweighted Accuracy (UA %) — higher is better", fontsize=12)
    ax.set_title("Compute-Performance Pareto Frontier\nCREMA-D Emotion Recognition", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # lower RTF = faster = better = right side

    plt.tight_layout()
    path = os.path.join(output_dir, "pareto_frontier.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_trainable_vs_ua(results, output_dir):
    """Trainable params vs UA — useful even without RTF."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ── Plot 1: Audio duration axis ─────────────────────────────────────────
    ax = axes[0]
    audio_runs = [r for r in results if r["model"] == "wav2vec2-large-robust"
                  and r["mode"] == "lora" and r["lora_r"] == 16 and r["unfreeze"] == 4]
    if audio_runs:
        audio_runs = sorted(audio_runs, key=lambda r: r["audio_len"])
        xs = [r["audio_len"] for r in audio_runs]
        ys = [r["ua"] for r in audio_runs]
        ax.plot(xs, ys, "o-", color="#2d4059", linewidth=2, markersize=10)
        for r in audio_runs:
            ax.annotate(f"{r['ua']:.1f}%", (r["audio_len"], r["ua"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel("Max Audio Length (seconds)", fontsize=11)
    ax.set_ylabel("UA (%)", fontsize=11)
    ax.set_title("xT Axis: Audio Duration vs UA", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([2, 4, 6])

    # ── Plot 2: LoRA rank axis ───────────────────────────────────────────────
    ax = axes[1]
    rank_runs = [r for r in results if r["model"] == "wav2vec2-large-robust"
                 and r["mode"] == "lora" and r["audio_len"] == 4.0 and r["unfreeze"] == 4]
    if rank_runs:
        rank_runs = sorted(rank_runs, key=lambda r: r["lora_r"] or 0)
        xs = [r["lora_r"] for r in rank_runs if r["lora_r"]]
        ys = [r["ua"] for r in rank_runs if r["lora_r"]]
        ax.plot(xs, ys, "s-", color="#e74c3c", linewidth=2, markersize=10)
        for r in rank_runs:
            if r["lora_r"]:
                ax.annotate(f"{r['ua']:.1f}%", (r["lora_r"], r["ua"]),
                            textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel("LoRA Rank (r)", fontsize=11)
    ax.set_ylabel("UA (%)", fontsize=11)
    ax.set_title("LoRA Rank vs UA", fontsize=12)
    ax.grid(True, alpha=0.3)

    # ── Plot 3: Unfreeze layers axis ─────────────────────────────────────────
    ax = axes[2]
    unfreeze_runs = [r for r in results if r["model"] == "wav2vec2-large-robust"
                     and r["mode"] == "lora" and r["audio_len"] == 4.0 and r["lora_r"] == 16]
    if unfreeze_runs:
        unfreeze_runs = sorted(unfreeze_runs, key=lambda r: r["unfreeze"])
        xs = [r["unfreeze"] for r in unfreeze_runs]
        ys = [r["ua"] for r in unfreeze_runs]
        ax.plot(xs, ys, "^-", color="#27ae60", linewidth=2, markersize=10)
        for r in unfreeze_runs:
            ax.annotate(f"{r['ua']:.1f}%", (r["unfreeze"], r["ua"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel("Unfrozen Top Encoder Layers", fontsize=11)
    ax.set_ylabel("UA (%)", fontsize=11)
    ax.set_title("DAMA: Unfreeze Layers vs UA", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.suptitle("CREMA-D Emotion Recognition — Axis Sweeps", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "axis_sweeps.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_model_comparison(results, output_dir):
    """Bar chart comparing model sizes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    model_runs = {}
    for r in results:
        key = f"{r['model']}\n({r['mode']}, r={r['lora_r']}, {r['audio_len']}s)"
        if key not in model_runs or r["ua"] > model_runs[key]["ua"]:
            model_runs[key] = r

    keys   = list(model_runs.keys())
    uas    = [model_runs[k]["ua"] for k in keys]
    params = [model_runs[k]["trainable_M"] for k in keys]
    colors = ["#2ecc71" if "large-robust" in k else
              "#e74c3c" if "base" in k else
              "#3498db" if "whisper" in k else "#9b59b6" for k in keys]

    bars = ax.bar(range(len(keys)), uas, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, fontsize=8)
    ax.set_ylabel("Unweighted Accuracy (UA %)", fontsize=12)
    ax.set_title("Model Comparison on CREMA-D", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 80)

    # Annotate bars with UA and trainable params
    for i, (bar, key) in enumerate(zip(bars, keys)):
        r = model_runs[key]
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{r['ua']:.1f}%\n{r['trainable_M']:.0f}M params",
                ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def print_summary_table(results, output_dir):
    """Print and save a summary table of all results."""
    print("\n" + "="*100)
    print(f"{'Run Name':<55} {'UA%':>6} {'Acc%':>6} {'TrainM':>8} {'Trt%':>6} {'Hrs':>5} {'RTF':>7}")
    print("="*100)

    rows = []
    for r in sorted(results, key=lambda x: -x["ua"]):
        rtf_str = f"{r['rtf']:.4f}" if r["rtf"] else "N/A"
        name = r["name"][:54]
        print(f"{name:<55} {r['ua']:>6.1f} {r['accuracy']:>6.1f} "
              f"{r['trainable_M']:>8.1f} {r['trainable_pct']:>6.2f} "
              f"{r['train_hrs']:>5.2f} {rtf_str:>7}")
        rows.append(r)
    print("="*100)

    # Save as JSON
    out = os.path.join(output_dir, "all_results.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nAll results saved to {out}")

    # Print Pareto optimal (by trainable params vs UA, since RTF may not be available)
    pareto = compute_pareto(results, "trainable_M", "ua")
    print(f"\nPareto optimal (minimize trainable params, maximize UA):")
    for r in pareto:
        print(f"  {r['name']:<55} UA={r['ua']:.1f}%  trainable={r['trainable_M']:.1f}M")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading results from {args.checkpoints_dir}...")
    results = load_results(args.checkpoints_dir)
    print(f"\nLoaded {len(results)} runs\n")

    if not results:
        print("No results found. Make sure checkpoints have run_meta.json files.")
        return

    print_summary_table(results, args.output_dir)
    plot_trainable_vs_ua(results, args.output_dir)
    plot_model_comparison(results, args.output_dir)

    with_rtf = [r for r in results if r["rtf"]]
    if with_rtf:
        plot_pareto_frontier(results, args.output_dir)
    else:
        print("\nNo RTF data yet — run evaluate_checkpoint on each run first.")
        print("Skipping RTF Pareto plot, generating axis sweep plots instead.")


if __name__ == "__main__":
    main()