"""
run_emotion_experiment.py — Experiment launcher for Speech Emotion Recognition
===============================================================================

This script owns the high-level experiment axes:
  --benchmark_dataset   benchmark to train/eval on (iemocap | cremad | ravdess | msp_improv)
  --llm_name            encoder model name or shorthand key (e.g. wav2vec2-large-robust)
  --max_audio_len       max clip length in seconds (overrides per-benchmark default)

Everything else is passed through as raw vargs after `--` and forwarded
directly to emotion_finetune.py's argument parser.

Design rationale
----------------
Same separation-of-concerns as run_experiment.py:
  - run_emotion_experiment.py  → what experiment (axes, dataset, model choice)
  - emotion_finetune.py        → how to run it (training loop, LoRA, eval)
  - vargs let you override ANY emotion_finetune arg without touching this file

Model shorthands (--llm_name):
  wav2vec2-large-robust   → facebook/wav2vec2-large-robust   [default, best SER]
  hubert-large            → facebook/hubert-large-ls960-ft
  wav2vec2-base           → facebook/wav2vec2-base
  wavlm-large             → microsoft/wavlm-large            [best SUPERB]
  wav2vec2-large          → facebook/wav2vec2-large-960h

  You can also pass a full HuggingFace path directly (e.g. superb/wav2vec2-large-superb-er).

Usage
-----
  # Basic: LoRA on IEMOCAP with wav2vec2-large-robust
  python run_emotion_experiment.py \\
      --benchmark_dataset iemocap \\
      --llm_name wav2vec2-large-robust

  # Full finetune on CREMA-D with WavLM-large
  python run_emotion_experiment.py \\
      --benchmark_dataset cremad \\
      --llm_name wavlm-large \\
      -- --mode full --batch_size 8 --max_steps 2000

  # Eval only from a saved checkpoint
  python run_emotion_experiment.py \\
      --benchmark_dataset iemocap \\
      --llm_name wav2vec2-large-robust \\
      -- --eval_only --checkpoint ./checkpoints/ser-wav2vec2-large-robust-lora-iemocap-r32

  # Sweep over all models × modes
  python run_emotion_experiment.py \\
      --benchmark_dataset iemocap \\
      --llm_name wav2vec2-large-robust \\
      -- --sweep \\
         --sweep_models facebook/wav2vec2-large-robust,microsoft/wavlm-large \\
         --sweep_modes full,lora

  # Quick smoke test (2000 samples, streaming)
  python run_emotion_experiment.py \\
      --benchmark_dataset cremad \\
      --llm_name wav2vec2-base \\
      -- --streaming --max_train_samples 2000 --max_eval_samples 500 \\
         --mode lora --max_steps 300
"""

import sys
import argparse

# Import emotion_finetune as a module (avoids subprocess, shares process memory)
import Emotion.emotion_finetune as ef


# ─────────────────────────────────────────────────────────────────────────────
# VALID VALUES
# ─────────────────────────────────────────────────────────────────────────────

VALID_BENCHMARKS = list(ef.BENCHMARK_REGISTRY.keys())
MODEL_SHORTHANDS = ef.MODEL_REGISTRY   # key → full HF path


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_launcher_args():
    p = argparse.ArgumentParser(
        description=(
            "Experiment launcher for Speech Emotion Recognition.\n"
            "Pass emotion_finetune.py overrides after --"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_emotion_experiment.py --benchmark_dataset iemocap \\\n"
            "      --llm_name wav2vec2-large-robust\n\n"
            "  python run_emotion_experiment.py --benchmark_dataset cremad \\\n"
            "      --llm_name wavlm-large -- --mode full --batch_size 8\n"
        ),
    )

    p.add_argument(
        "--benchmark_dataset",
        type=str,
        required=True,
        choices=VALID_BENCHMARKS,
        help=(
            "Benchmark dataset to train and evaluate on.\n"
            f"  Choices: {VALID_BENCHMARKS}\n"
            "  Label sets:\n"
            "    iemocap    : 4-class (neutral, happy, sad, angry)\n"
            "    cremad     : 6-class (+ fear, disgust)\n"
            "    ravdess    : 4-class (neutral/calm merged, fearful/disgust/surprised dropped)\n"
            "    msp_improv : 4-class (neutral, happy, sad, angry)"
        ),
    )
    p.add_argument(
        "--llm_name",
        type=str,
        required=True,
        help=(
            "Encoder model: shorthand key OR full HuggingFace path.\n"
            "  Shorthands:\n"
            + "\n".join(
                f"    {k:28s} → {v}" for k, v in MODEL_SHORTHANDS.items()
            )
            + "\n  Or pass any HF model path directly."
        ),
    )
    p.add_argument(
        "--max_audio_len",
        type=float,
        default=None,
        help=(
            "Override max audio clip length in seconds.\n"
            "  Per-benchmark defaults:\n"
            "    iemocap=6s  cremad=6s  ravdess=5s  msp_improv=8s\n"
            "  Longer clips → more context but more VRAM/time."
        ),
    )

    launcher_args, vargs = p.parse_known_args()
    return launcher_args, vargs


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def resolve_model_name(llm_name: str) -> str:
    """
    Resolve a shorthand key to a full HF model path.
    If not a known shorthand, return as-is (assume full HF path or local path).
    """
    return MODEL_SHORTHANDS.get(llm_name, llm_name)


def build_finetune_argv(launcher_args, vargs: list) -> list:
    """
    Merge launcher args into argv list for emotion_finetune.parse_args().
    vargs (after --) are appended as-is and override launcher defaults.
    """
    model_name = resolve_model_name(launcher_args.llm_name)

    base = [
        "--benchmark_dataset", launcher_args.benchmark_dataset,
        "--model_name",        model_name,
    ]
    if launcher_args.max_audio_len is not None:
        base += ["--max_audio_len", str(launcher_args.max_audio_len)]

    vargs_clean = [a for a in vargs if a != "--"]
    return base + vargs_clean


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    launcher_args, vargs = parse_launcher_args()

    model_name  = resolve_model_name(launcher_args.llm_name)
    bench_info  = ef.BENCHMARK_REGISTRY[launcher_args.benchmark_dataset]

    # ── Print experiment summary ──────────────────────────────────────────────
    print("=" * 65)
    print("  EMOTION RECOGNITION EXPERIMENT")
    print("=" * 65)
    print(f"  benchmark        : {launcher_args.benchmark_dataset}")
    print(f"  num_labels       : {bench_info['num_labels']}  "
          f"({', '.join(bench_info['label_names'])})")
    print(f"  model            : {model_name}")
    print(f"  max_audio_len    : "
          f"{launcher_args.max_audio_len or bench_info['max_audio_len']}s")
    if vargs:
        print(f"  vargs (overrides): {' '.join(vargs)}")
    else:
        print("  vargs            : (none — using emotion_finetune defaults)")
    print("=" * 65 + "\n")

    # ── Build argv and parse with emotion_finetune's parser ───────────────────
    finetune_argv = build_finetune_argv(launcher_args, vargs)
    print(f"Resolved emotion_finetune argv:\n  {' '.join(finetune_argv)}\n")

    ft_args = ef.parse_args(finetune_argv)

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if ft_args.sweep:
        ef.run_sweep(ft_args)
    elif ft_args.eval_only:
        ef.evaluate_checkpoint(ft_args)
    else:
        ef.train(ft_args)


if __name__ == "__main__":
    main()
