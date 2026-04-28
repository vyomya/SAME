

import sys
import argparse

# Import emotion_finetune as a module (avoids subprocess, shares process memory)
import Emotion.emotion_finetune as ef


VALID_BENCHMARKS = list(ef.BENCHMARK_REGISTRY.keys())
MODEL_SHORTHANDS = ef.MODEL_REGISTRY   # key → full HF path


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
