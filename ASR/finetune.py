"""
Whisper Finetuning on ASR / downstream tasks
==============================================
Two modes:
  mode=full  — finetune entire Whisper (enc+dec), all weights
  mode=lora  — LoRA on encoder self-attention + decoder cross-attention

Can be used standalone OR imported/called by run_experiment.py.

Key audio framing args (set by run_experiment.py):
  --tokens_per_frame  : LM tokens produced per encoder output frame
  --total_frames      : total encoder output frames for a max-length audio clip
  --task              : downstream task name (asr, translation, ...)
  --benchmark_dataset : which benchmark dataset to use (librispeech, common_voice, ...)

Usage (standalone):
  python whisper_finetune.py --model_size small --mode full

Usage (called from run_experiment.py):
  python run_experiment.py --task asr --benchmark_dataset librispeech \\
      --llm_size small --tokens_per_frame 2 --total_frames 1500 \\
      -- --mode lora --lora_r 64 --batch_size 8
"""

import os
import re
import time
import json
import argparse
import numpy as np
import torch
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
)
import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

WHISPER_SIZES = {
    "tiny":     "openai/whisper-tiny",
    "base":     "openai/whisper-base",
    "small":    "openai/whisper-small",
    "medium":   "openai/whisper-medium",
    "large-v3": "openai/whisper-large-v3",
}

# ── Supported tasks ───────────────────────────────────────────────────────────
SUPPORTED_TASKS = ["asr", "translation"]

# ── Benchmark dataset registry ────────────────────────────────────────────────
# Each entry defines how to load the dataset and which column holds the label.
# train_split / eval_split can be overridden from CLI.
BENCHMARK_REGISTRY = {
    "librispeech": {
        "hf_path":       "librispeech_asr",   # ← use this, not openslr/librispeech_asr
        "hf_config":     "clean",
        "text_column":   "text",
        "language":      "English",
        "default_train": "train.clean.100",
        "default_eval":  "validation.clean",
        "default_test":  ["test.clean", "test.other"],
        "task":          "transcribe",
    },
    "common_voice": {
        "hf_path":       "mozilla-foundation/common_voice_13_0",
        "hf_config":     "en",
        "text_column":   "sentence",
        "language":      "English",
        "default_train": "train",
        "default_eval":  "validation",
        "default_test":  ["test"],
        "task":          "transcribe",
    },
    "fleurs": {
        "hf_path":       "google/fleurs",
        "hf_config":     "en_us",
        "text_column":   "transcription",
        "language":      "English",
        "default_train": "train",
        "default_eval":  "validation",
        "default_test":  ["test"],
        "task":          "transcribe",
    },
    "voxpopuli": {
        "hf_path":       "facebook/voxpopuli",
        "hf_config":     "en",
        "text_column":   "normalized_text",
        "language":      "English",
        "default_train": "train",
        "default_eval":  "validation",
        "default_test":  ["test"],
        "task":          "transcribe",
    },
}

# LoRA target modules per Whisper component
# Encoder self-attention:  q_proj, v_proj  (in encoder layers)
# Decoder self-attention:  q_proj, v_proj  (in decoder layers)
# Decoder cross-attention: q_proj, v_proj  (encoder_attn in decoder layers)
LORA_TARGET_MODULES = [
    # encoder self-attention
    "encoder.layers.*.self_attn.q_proj",
    "encoder.layers.*.self_attn.v_proj",
    # decoder self-attention
    "decoder.layers.*.self_attn.q_proj",
    "decoder.layers.*.self_attn.v_proj",
    # decoder cross-attention  ← key for seq2seq
    "decoder.layers.*.encoder_attn.q_proj",
    "decoder.layers.*.encoder_attn.v_proj",
]

# Simplified module names (PEFT matches by suffix)
LORA_TARGET_MODULES_SIMPLE = [
    "q_proj",
    "v_proj",
]


# ─────────────────────────────────────────────────────────────────────────────
# DATA COLLATOR
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WhisperDataCollator:
    """
    Pads input features and label token ids.
    Labels are padded with -100 so padding positions are
    ignored in the cross-entropy loss.
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # ── Audio features ────────────────────────────────────────────────────
        # Each feature is already (128, 3000) from WhisperFeatureExtractor
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # ── Labels (token ids) ────────────────────────────────────────────────
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Replace tokenizer pad_token_id with -100 (ignored in CE loss)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Strip BOS token prepended by tokenizer if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

def load_benchmark_dataset(benchmark, split, sampling_rate=16_000):
    info = BENCHMARK_REGISTRY[benchmark]

    if benchmark == "librispeech":
        config = "clean" if "clean" in split else "other"
        # LibriSpeech split names use dot notation: "train.clean.100"
        # HF datasets expects them as-is
        dataset = load_dataset(
            "librispeech_asr",
            config,
            split=split,
            trust_remote_code=True,
        )
    else:
        dataset = load_dataset(
            info["hf_path"],
            info["hf_config"],
            split=split,
            trust_remote_code=True,
        )

    return dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

def prepare_dataset(
    batch,
    processor,
    text_column:      str = "text",
    max_label_len:    int = 448,
    tokens_per_frame: int = 1,
    total_frames:     int = 1500,
):
    """
    Map function: converts raw audio + text into model inputs.

    Args:
        text_column      : column name for transcript (varies by benchmark)
        max_label_len    : max decoder token sequence length
        tokens_per_frame : expected LM tokens per encoder output frame.
                           Used to derive a dynamic max_label_len if the
                           caller hasn't set one explicitly:
                             max_label_len = total_frames * tokens_per_frame
                           This ensures the label budget matches the audio
                           capacity of the encoder.
        total_frames     : encoder output frames for a max-length audio clip.
                           Whisper's CNN compresses 3000 mel frames → 1500
                           encoder frames for 30s audio.

    Whisper STFT config (fixed, built into WhisperFeatureExtractor):
      n_fft=400  (25ms window @ 16kHz)
      hop=160    (10ms hop    @ 16kHz)
      n_mels=128
      → CNN 2× stride → 1500 encoder frames per 30s clip
    """
    audio = batch["audio"]

    # Derive label length from audio framing if not explicitly overridden
    derived_max = total_frames * tokens_per_frame
    effective_max = min(max_label_len, derived_max)

    # Log-mel spectrogram: (128, 3000)
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
    ).input_features[0]

    # Tokenize transcript — lowercase for English ASR convention
    transcript = batch[text_column]
    if isinstance(transcript, str):
        transcript = transcript.lower()

    batch["labels"] = processor.tokenizer(
        transcript,
        max_length=effective_max,
        truncation=True,
    ).input_ids

    return batch


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def make_compute_metrics(processor):
    """
    Returns a compute_metrics function compatible with Seq2SeqTrainer.
    Applies simple normalization before WER computation.
    """
    wer_metric = evaluate.load("wer")

    def normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s\']", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def compute_metrics(pred):
        pred_ids   = pred.predictions
        label_ids  = pred.label_ids

        # Replace -100 back to pad_token_id for decoding
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_str  = [normalize(p) for p in pred_str]
        label_str = [normalize(r) for r in label_str]

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": round(100 * wer, 4)}

    return compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# MODEL SETUP
# ─────────────────────────────────────────────────────────────────────────────

def build_model_full(model_name: str) -> WhisperForConditionalGeneration:
    """
    Load full Whisper model — all weights trainable.
    All encoder + decoder parameters will be updated during training.
    """
    print(f"Loading {model_name} (full finetune mode)...")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Disable Whisper's built-in forced_decoder_ids during training
    # (these force specific language/task tokens; we handle this via processor)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # required for gradient checkpointing

    total = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total:,} | All trainable")
    return model


def build_model_lora(
    model_name: str,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
) -> WhisperForConditionalGeneration:
    """
    Load Whisper with LoRA adapters on:
      - Encoder self-attention: q_proj, v_proj
      - Decoder self-attention: q_proj, v_proj
      - Decoder cross-attention: q_proj, v_proj  (encoder_attn)

    Why these layers?
      - q_proj and v_proj carry the most task-specific information
      - Cross-attention layers are critical for seq2seq — they control
        how the decoder attends to audio representations
      - k_proj and out_proj contribute less and can stay frozen

    Why LoRA over full finetuning?
      - Reduces trainable params from ~1.5B (large-v3) to ~15M (~1%)
      - Prevents catastrophic forgetting of general speech knowledge
      - Fits large models on smaller GPUs (24GB instead of 80GB)
      - Near-identical WER to full finetuning on in-domain data
    """
    print(f"Loading {model_name} (LoRA mode, r={lora_r}, alpha={lora_alpha})...")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        # PEFT matches these by suffix against all Linear layer names
        target_modules=LORA_TARGET_MODULES_SIMPLE,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    # Resolve benchmark info
    benchmark = getattr(args, "benchmark_dataset", "librispeech")
    task      = getattr(args, "task",              "asr")
    bench_info = BENCHMARK_REGISTRY.get(benchmark, BENCHMARK_REGISTRY["librispeech"])

    # Fall back to benchmark defaults if splits not explicitly provided
    train_split = getattr(args, "train_split", None) or bench_info["default_train"]
    eval_split  = getattr(args, "eval_split",  None) or bench_info["default_eval"]
    text_column = bench_info["text_column"]
    language    = bench_info["language"]
    whisper_task = bench_info["task"]

    # Audio framing params
    tokens_per_frame = getattr(args, "tokens_per_frame", 1)
    total_frames     = getattr(args, "total_frames",     1500)

    model_name = WHISPER_SIZES[args.model_size]
    run_name   = f"whisper-{args.model_size}-{args.mode}-{benchmark}-{task}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # ── Processor ─────────────────────────────────────────────────────────────
    processor = WhisperProcessor.from_pretrained(
        model_name, language=language, task=whisper_task
    )

    # ── Dataset ────────────────────────────────────────────────────────────────
    train_dataset = load_benchmark_dataset(benchmark, train_split)
    eval_dataset  = load_benchmark_dataset(benchmark, eval_split)

    print("Preprocessing datasets (extracting features + tokenizing)...")

    map_fn = lambda b: prepare_dataset(
        b,
        processor,
        text_column=text_column,
        max_label_len=args.max_label_len,
        tokens_per_frame=tokens_per_frame,
        total_frames=total_frames,
    )

    train_dataset = train_dataset.map(
        map_fn,
        remove_columns=train_dataset.column_names,
        num_proc=args.num_proc,
        desc="Train",
    )
    eval_dataset = eval_dataset.map(
        map_fn,
        remove_columns=eval_dataset.column_names,
        num_proc=args.num_proc,
        desc="Eval",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.mode == "full":
        model = build_model_full(model_name)
    elif args.mode == "lora":
        model = build_model_lora(
            model_name,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}. Choose 'full' or 'lora'.")

    # ── Data collator ─────────────────────────────────────────────────────────
    collator = WhisperDataCollator(processor=processor)

    # ── Training arguments ────────────────────────────────────────────────────
    #
    # Key decisions:
    #   - predict_with_generate=True: use model.generate() at eval time
    #     (greedy decoding) instead of teacher-forced logits — gives real WER
    #   - generation_max_length: cap decode length
    #   - gradient_checkpointing: trade compute for memory (essential for large)
    #   - fp16: half precision on GPU
    #   - eval_steps / save_steps: evaluate every N steps, save best checkpoint
    #
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,

        # Batch size & accumulation
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,

        # Learning rate
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="linear",

        # Training length
        max_steps=args.max_steps,          # use max_steps OR num_train_epochs
        # num_train_epochs=args.num_epochs, # uncomment to use epochs instead

        # Precision & memory
        fp16=args.fp16 and torch.cuda.is_available(),
        gradient_checkpointing=True,

        # Evaluation
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        predict_with_generate=True,
        generation_max_length=args.max_label_len,

        # Saving
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,

        # Logging
        logging_steps=args.log_steps,
        report_to=["tensorboard"],
        run_name=run_name,

        # Misc
        dataloader_num_workers=args.num_proc,
        remove_unused_columns=False,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=make_compute_metrics(processor),
        tokenizer=processor.feature_extractor,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Run: {run_name}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Eval  samples: {len(eval_dataset)}")
    print(f"{'='*60}\n")

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/3600:.2f}h")

    # ── Save final model ──────────────────────────────────────────────────────
    if args.mode == "lora":
        # Save only LoRA adapter weights (small — ~50MB for large-v3)
        model.save_pretrained(output_dir)
    else:
        trainer.save_model(output_dir)

    processor.save_pretrained(output_dir)

    # ── Save run metadata for sweep analysis ─────────────────────────────────
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    meta = {
        "model_size":        args.model_size,
        "mode":              args.mode,
        "task":              getattr(args, "task", "asr"),
        "benchmark_dataset": getattr(args, "benchmark_dataset", "librispeech"),
        "tokens_per_frame":  getattr(args, "tokens_per_frame", 1),
        "total_frames":      getattr(args, "total_frames", 1500),
        "lora_r":            args.lora_r if args.mode == "lora" else None,
        "train_split":       train_split,
        "trainable_params":  trainable,
        "total_params":      total,
        "trainable_pct":     round(100 * trainable / total, 4),
        "training_hours":    round(elapsed / 3600, 3),
    }
    with open(os.path.join(output_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))

    return output_dir


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION (standalone — load checkpoint and run full test set)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_checkpoint(args):
    """
    Load a saved checkpoint and run evaluation on LibriSpeech test sets.
    Logs WER, RTF (real-time factor), and parameter counts.

    Real-Time Factor (RTF) = inference_time / audio_duration
      RTF < 1.0 means faster than real-time (good for production)
    """
    checkpoint_dir = args.checkpoint
    assert checkpoint_dir, "--checkpoint required for eval_only mode"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = WHISPER_SIZES[args.model_size]

    # ── Load processor ────────────────────────────────────────────────────────
    processor = WhisperProcessor.from_pretrained(
        checkpoint_dir if os.path.exists(os.path.join(checkpoint_dir, "preprocessor_config.json"))
        else model_name,
        language="English",
        task="transcribe",
    )

    # ── Load model ────────────────────────────────────────────────────────────
    if args.mode == "lora":
        base = WhisperForConditionalGeneration.from_pretrained(model_name)
        base.config.forced_decoder_ids = None
        model = PeftModel.from_pretrained(base, checkpoint_dir)
        model = model.merge_and_unload()   # merge LoRA into base weights for fast inference
    else:
        model = WhisperForConditionalGeneration.from_pretrained(checkpoint_dir)

    model.config.forced_decoder_ids = None
    model = model.to(device).eval()

    # ── Eval datasets — use benchmark registry ────────────────────────────────
    benchmark  = getattr(args, "benchmark_dataset", "librispeech")
    bench_info = BENCHMARK_REGISTRY.get(benchmark, BENCHMARK_REGISTRY["librispeech"])
    test_splits = bench_info["default_test"]

    eval_splits = {
        split: load_benchmark_dataset(benchmark, split)
        for split in test_splits
    }

    wer_metric = evaluate.load("wer")
    results = {}

    for split_name, dataset in eval_splits.items():
        print(f"\nEvaluating on {split_name} ({len(dataset)} samples)...")

        all_preds, all_refs = [], []
        total_audio_s = 0.0
        total_infer_s = 0.0

        for sample in tqdm.tqdm(dataset):
            audio_array = sample["audio"]["array"].astype(np.float32)
            ref_text    = sample["text"].lower().strip()
            audio_dur_s = len(audio_array) / 16_000

            # Feature extraction
            features = processor.feature_extractor(
                audio_array, sampling_rate=16_000, return_tensors="pt"
            ).input_features.to(device)

            # Timed inference
            t0 = time.time()
            with torch.no_grad():
                pred_ids = model.generate(
                    features,
                    language="en",
                    task="transcribe",
                    num_beams=1,        # greedy — use num_beams=5 for best WER
                    max_new_tokens=256,
                )
            infer_time = time.time() - t0

            pred_text = processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True)

            # Normalize
            pred_text = normalize_text(pred_text)
            ref_text  = normalize_text(ref_text)

            all_preds.append(pred_text)
            all_refs.append(ref_text)
            total_audio_s += audio_dur_s
            total_infer_s += infer_time

        wer = wer_metric.compute(predictions=all_preds, references=all_refs)
        rtf = total_infer_s / total_audio_s

        results[split_name] = {
            "wer":           round(100 * wer, 3),
            "rtf":           round(rtf, 4),
            "total_audio_h": round(total_audio_s / 3600, 2),
            "total_infer_h": round(total_infer_s / 3600, 2),
        }
        print(f"  WER: {results[split_name]['wer']:.3f}% | RTF: {rtf:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters())
    summary = {
        "model_size":   args.model_size,
        "mode":         args.mode,
        "total_params": total_params,
        "results":      results,
    }
    print("\n" + "="*60)
    print(json.dumps(summary, indent=2))
    print("="*60)

    out_path = os.path.join(checkpoint_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {out_path}")
    return summary


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s\']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# SWEEP RUNNER  (compute vs WER across model sizes and modes)
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(args):
    """
    Run a full grid sweep over model sizes × modes and collect results.
    Saves a combined sweep_results.json for plotting compute vs WER curves.

    Compute proxy metrics logged per run:
      - trainable_params     : params updated during training
      - total_params         : total model params (proxy for inference VRAM)
      - training_hours       : wall-clock training time
      - rtf (test.clean)     : real-time factor (inference speed)
      - wer (test.clean/other): accuracy
    """
    sizes = args.sweep_sizes.split(",")   # e.g. "small,medium,large-v3"
    modes = args.sweep_modes.split(",")   # e.g. "full,lora"

    all_results = []

    for size in sizes:
        for mode in modes:
            print(f"\n{'#'*60}")
            print(f"  SWEEP: whisper-{size} | mode={mode}")
            print(f"{'#'*60}")

            # Temporarily override args for this run
            args.model_size = size
            args.mode       = mode

            checkpoint_dir = train(args)
            eval_summary   = evaluate_checkpoint(
                argparse.Namespace(
                    model_size=size,
                    mode=mode,
                    checkpoint=checkpoint_dir,
                )
            )

            # Load run metadata
            meta_path = os.path.join(checkpoint_dir, "run_meta.json")
            with open(meta_path) as f:
                meta = json.load(f)

            combined = {**meta, **eval_summary["results"]}
            all_results.append(combined)

            # Save incrementally
            sweep_path = os.path.join(args.output_dir, "sweep_results.json")
            with open(sweep_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nSweep results so far saved to {sweep_path}")

    print("\n" + "="*60)
    print("SWEEP COMPLETE")
    print("="*60)
    for r in all_results:
        clean_wer = r.get("test.clean", {}).get("wer", "N/A")
        other_wer = r.get("test.other", {}).get("wer", "N/A")
        print(
            f"  whisper-{r['model_size']:10s} | {r['mode']:5s} | "
            f"params={r['trainable_params']:>12,} | "
            f"test.clean WER={clean_wer}% | test.other WER={other_wer}%"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Whisper Finetuning: full vs LoRA, multi-benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Experiment identity (set by run_experiment.py or manually) ────────────
    p.add_argument("--task", type=str, default="asr",
                   choices=SUPPORTED_TASKS,
                   help="Downstream task")
    p.add_argument("--benchmark_dataset", type=str, default="librispeech",
                   choices=list(BENCHMARK_REGISTRY.keys()),
                   help="Benchmark dataset to train and evaluate on")
    p.add_argument("--tokens_per_frame", type=int, default=1,
                   help=(
                       "Expected LM tokens per encoder output frame. "
                       "Used to derive max_label_len = total_frames * tokens_per_frame. "
                       "Typical: 1 for English ASR (1 token ≈ 1 frame), "
                       "2-3 for morphologically rich languages."
                   ))
    p.add_argument("--total_frames", type=int, default=1500,
                   help=(
                       "Encoder output frames for max-length audio. "
                       "Whisper: 3000 mel frames → 1500 after 2× CNN stride. "
                       "Reduce if training on shorter audio clips."
                   ))

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument("--model_size", type=str, default="small",
                   choices=list(WHISPER_SIZES.keys()),
                   help="Whisper model size")
    p.add_argument("--mode", type=str, default="lora",
                   choices=["full", "lora"],
                   help="Finetuning mode: full weights or LoRA adapters")

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument("--train_split", type=str, default=None,
                   help="Override training split (defaults to benchmark registry value)")
    p.add_argument("--eval_split", type=str, default=None,
                   help="Override eval split (defaults to benchmark registry value)")
    p.add_argument("--max_label_len", type=int, default=448,
                   help="Hard cap on label token length (further bounded by total_frames*tokens_per_frame)")
    p.add_argument("--num_proc", type=int, default=4,
                   help="Workers for dataset preprocessing")

    # ── LoRA ──────────────────────────────────────────────────────────────────
    p.add_argument("--lora_r",       type=int,   default=32)
    p.add_argument("--lora_alpha",   type=int,   default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--eval_batch_size", type=int,   default=8)
    p.add_argument("--grad_accum",      type=int,   default=2)
    p.add_argument("--learning_rate",   type=float, default=1e-5)
    p.add_argument("--warmup_steps",    type=int,   default=500)
    p.add_argument("--max_steps",       type=int,   default=4000,
                   help="Total training steps. -1 to use num_epochs instead.")
    p.add_argument("--num_epochs",      type=int,   default=3)
    p.add_argument("--eval_steps",      type=int,   default=500)
    p.add_argument("--log_steps",       type=int,   default=25)
    p.add_argument("--fp16",            action="store_true", default=True)

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("--output_dir", type=str, default="./checkpoints")

    # ── Eval only ─────────────────────────────────────────────────────────────
    p.add_argument("--eval_only",  action="store_true",
                   help="Skip training, run evaluation from checkpoint")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Checkpoint path for eval_only mode")

    # ── Sweep ─────────────────────────────────────────────────────────────────
    p.add_argument("--sweep",       action="store_true")
    p.add_argument("--sweep_sizes", type=str, default="small,medium,large-v3")
    p.add_argument("--sweep_modes", type=str, default="full,lora")

    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()   # parses sys.argv by default

    if args.sweep:
        run_sweep(args)
    elif args.eval_only:
        evaluate_checkpoint(args)
    else:
        train(args)