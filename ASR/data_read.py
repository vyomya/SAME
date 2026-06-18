"""
analyze_results.py — Read eval JSONs from a folder and print a results table.
Usage: python analyze_results.py --results_dir /path/to/eval_results
"""

import os
import json
import glob
import argparse


def load_rows(results_dir):
    rows = []
    for fpath in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(fpath) as f:
                d = json.load(f)
            if "run_info" not in d or "splits" not in d:
                continue

            run   = d["run_info"]
            comp  = d.get("compression", {})
            flops = d.get("flops", {})
            split = (d["splits"].get("test_other") or
                     next(iter(d["splits"].values()), {}))

            rows.append({
                "Model":      run.get("model_size", "—"),
                "LoRA_r":     run.get("lorar", "—"),
                "Mode":       run.get("mode", "—"),
                "Frames":     run.get("total_frames", "—"),
                "Frm/Tok":    run.get("tokens_per_frame", "—"),
                "Quant":      comp.get("quantization", "fp16"),
                "Prune":      f"{comp.get('pruning_ratio', 0):.0%}",
                "WER %":      split.get("wer_pct"),
                "Size MB":    comp.get("model_size_mb"),
                "FLOPs G":    flops.get("flops_total_G"),
                "EffFLOPs G": flops.get("flops_total_eff_G"),
            })
        except Exception as e:
            print(f"Skipping {os.path.basename(fpath)}: {e}")
    return rows


def print_table(rows):
    cols = [
        ("Model",      "Model",      "<12"),
        ("LoRA_r",     "LoRA_r",     "<7"),
        ("Mode",       "Mode",       "<8"),
        ("Frames",     "Frames",     ">7"),
        ("Frm/Tok",    "Frm/Tok",    ">8"),
        ("Quant",      "Quant",      "<6"),
        ("Prune",      "Prune",      ">6"),
        ("WER %",      "WER %",      ">8"),
        ("Size MB",    "Size MB",    ">9"),
        ("FLOPs G",    "FLOPs G",    ">9"),
        ("EffFLOPs G", "EffFLOPs G", ">11"),
    ]

    def fmt(val, spec):
        if val is None:
            return format("—", spec)
        if isinstance(val, float):
            return format(f"{val:.3f}", spec)
        return format(str(val), spec)

    header = "  ".join(format(label, spec) for label, _, spec in cols)
    sep    = "  ".join("-" * int(spec.strip("<>")) for _, _, spec in cols)

    print("=" * len(sep))
    print(header)
    print(sep)
    for row in rows:
        print("  ".join(fmt(row[key], spec) for _, key, spec in cols))
    print("=" * len(sep))
    print(f"{len(rows)} result(s)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True,
                   help="Folder containing eval JSON files")
    p.add_argument("--sort_by", default="WER %",
                   choices=["WER %", "FLOPs G", "EffFLOPs G", "Size MB", "Model"],
                   help="Column to sort by")
    args = p.parse_args()

    rows = load_rows(args.results_dir)
    if not rows:
        print("No valid JSON files found.")
        return

    rows.sort(key=lambda r: (r[args.sort_by] is None, r[args.sort_by]))
    print_table(rows)


if __name__ == "__main__":
    main()