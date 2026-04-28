

import sys
import argparse

# Import whisper_finetune as a module (avoids subprocess, shares process memory)
import ASR.finetune as wf

VALID_TASKS      = wf.SUPPORTED_TASKS
VALID_BENCHMARKS = list(wf.BENCHMARK_REGISTRY.keys())
VALID_SIZES      = list(wf.WHISPER_SIZES.keys())


def parse_launcher_args():
    p = argparse.ArgumentParser(
        description=(
            "Experiment launcher for Whisper finetuning.\n"
            "Pass whisper_finetune.py overrides after --"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_experiment.py --task asr --benchmark_dataset librispeech \\\n"
            "      --llm_size small --tokens_per_frame 1 --total_frames 1500\n\n"
            "  python run_experiment.py --task asr --benchmark_dataset common_voice \\\n"
            "      --llm_size medium --tokens_per_frame 2 --total_frames 1500 \\\n"
            "      -- --mode full --batch_size 8 --max_steps 2000\n"
        ),
    )

    # ── Required experiment axes ──────────────────────────────────────────────
    p.add_argument(
        "--task",
        type=str,
        required=True,
        choices=VALID_TASKS,
        help=(
            "Downstream task to finetune for.\n"
            f"  Choices: {VALID_TASKS}"
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
            "  Each benchmark has a default train/eval/test split registered\n"
            "  in whisper_finetune.BENCHMARK_REGISTRY."
        ),
    )
    p.add_argument(
        "--llm_size",
        type=str,
        required=True,
        choices=VALID_SIZES,
        help=(
            "Whisper model size (maps to --model_size in whisper_finetune.py).\n"
            f"  Choices: {VALID_SIZES}\n"
            "  Trainable param counts (approx):\n"
            "    tiny=39M  base=74M  small=244M  medium=769M  large-v3=1.54B"
        ),
    )
    p.add_argument(
        "--tokens_per_frame",
        type=int,
        required=True,
        help=(
            "Expected LM tokens produced per encoder output frame.\n"
            "  Used to derive max_label_len = total_frames * tokens_per_frame.\n"
            "  Typical values:\n"
            "    1   — English ASR (roughly 1 BPE token per 10ms frame)\n"
            "    2-3 — morphologically rich / agglutinative languages\n"
            "    3-5 — character-level or fine-grained tokenizers"
        ),
    )
    p.add_argument(
        "--total_frames",
        type=int,
        required=True,
        help=(
            "Encoder output frames for a max-length audio clip.\n"
            "  Whisper processes audio in 30s chunks:\n"
            "    30s × 16kHz = 480000 samples\n"
            "    → STFT with hop=160 → 3000 mel frames\n"
            "    → 2× CNN stride  → 1500 encoder frames  ← use 1500\n"
            "  Use a smaller value if your dataset has shorter clips."
        ),
    )

    # parse_known_args so that anything after -- falls into vargs
    launcher_args, vargs = p.parse_known_args()
    return launcher_args, vargs


def tokens_per_frame_explanation(tokens_per_frame: int, total_frames: int) -> str:
    """Return a human-readable explanation of the derived label budget."""
    derived = tokens_per_frame * total_frames
    return (
        f"  tokens_per_frame = {tokens_per_frame}\n"
        f"  total_frames     = {total_frames}\n"
        f"  → derived max_label_len = {tokens_per_frame} × {total_frames} = {derived} tokens\n"
        f"    (further capped by --max_label_len if set in vargs)"
    )


def build_finetune_argv(launcher_args, vargs: list) -> list:
    """
    Merge launcher args into a single argv list for whisper_finetune.parse_args().

    Launcher args are injected as named flags. vargs (after --) are appended
    as-is, so they can override anything including the launcher-injected values.

    Precedence (right wins in argparse):
        launcher defaults  <  launcher args  <  vargs
    """
    # Build the base argv from launcher args
    base = [
        "--task",              launcher_args.task,
        "--benchmark_dataset", launcher_args.benchmark_dataset,
        "--model_size",        launcher_args.llm_size,          # llm_size → model_size
        "--tokens_per_frame",  str(launcher_args.tokens_per_frame),
        "--total_frames",      str(launcher_args.total_frames),
    ]

    # Strip the '--' separator if it somehow ends up in vargs
    vargs_clean = [a for a in vargs if a != "--"]

    return base + vargs_clean


def main():
    launcher_args, vargs = parse_launcher_args()

    # ── Print experiment summary ──────────────────────────────────────────────
    print("=" * 65)
    print("  EXPERIMENT")
    print("=" * 65)
    print(f"  task             : {launcher_args.task}")
    print(f"  benchmark        : {launcher_args.benchmark_dataset}")
    print(f"  model size       : {launcher_args.llm_size}  "
          f"({wf.WHISPER_SIZES[launcher_args.llm_size]})")
    print(tokens_per_frame_explanation(
        launcher_args.tokens_per_frame, launcher_args.total_frames
    ))
    if vargs:
        print(f"  vargs (overrides): {' '.join(vargs)}")
    else:
        print("  vargs            : (none — using whisper_finetune defaults)")
    print("=" * 65 + "\n")

    # ── Build argv and parse with whisper_finetune's parser ───────────────────
    finetune_argv = build_finetune_argv(launcher_args, vargs)
    print(f"Resolved whisper_finetune argv:\n  {' '.join(finetune_argv)}\n")

    ft_args = wf.parse_args(finetune_argv)

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if ft_args.sweep:
        wf.run_sweep(ft_args)
    elif ft_args.eval_only:
        wf.evaluate_checkpoint(ft_args)
    else:
        wf.train(ft_args)


if __name__ == "__main__":
    main()