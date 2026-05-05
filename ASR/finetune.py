"""
Whisper Finetuning on ASR / downstream tasks
==============================================
Two modes:
  mode=full  — finetune entire Whisper (enc+dec), all weights
  mode=lora  — LoRA on encoder self-attention + decoder cross-attention

Streaming support:
  --streaming           stream audio on the fly, zero disk cache
  --max_train_samples   cap training samples (works with or without streaming)
  --max_eval_samples    cap eval samples

Axes of compute reduction (aligned with Wang et al.):
  xN → --model_size   : model capacity (tiny/small/medium/large-v3)
  xT → --total_frames : audio duration truncation (fewer encoder frames)
  xV → --tokens_per_frame : encoder output subsampling before decoder cross-attn

Can be used standalone OR imported/called by run_experiment.py.

Usage (standalone):
  python whisper_finetune.py --model_size small --mode lora --streaming
  python whisper_finetune.py --model_size small --mode lora --max_train_samples 2000

3-way experiment:
  # Run A — baseline
  python whisper_finetune.py --total_frames 1500 --tokens_per_frame 1
  # Run B — audio clipping
  python whisper_finetune.py --total_frames 750  --tokens_per_frame 1
  # Run C — encoder subsampling
  python whisper_finetune.py --total_frames 1500 --tokens_per_frame 2
"""

# ─────────────────────────────────────────────────────────────────────────────
# CACHE SETUP — must be before ALL other imports
# ─────────────────────────────────────────────────────────────────────────────
import os
CACHE_DIR = "/scratch/zt1/project/msml604/user/vyomwal5/anaconda3/envs/asr/hf_cache"
os.environ["LD_LIBRARY_PATH"] = (
    "/scratch/zt1/project/msml604/user/vyomwal5/anaconda3/envs/same/lib:"
    "/scratch/zt1/project/msml604/user/vyomwal5/anaconda3/envs/same/lib/python3.11/site-packages/torch/lib:"
    "/scratch/zt1/project/msml604/user/vyomwal5/anaconda3/envs/same/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:"
    "/scratch/zt1/project/msml604/user/vyomwal5/anaconda3/envs/same/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:"
    "/scratch/zt1/project/msml604/user/vyomwal5/anaconda3/envs/same/lib/python3.11/site-packages/nvidia/npp/lib:"
    + os.environ.get("LD_LIBRARY_PATH", "")
)
local_path = {
    "small":    "/scratch/zt1/project/msml604/user/vyomwal5/anaconda3/envs/asr/hf_cache/models/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d",
    "medium":   "/scratch/zt1/project/msml604/user/vyomwal5/anaconda3/envs/asr/hf_cache/models/models--openai--whisper-medium/snapshots/abdf7c39ab9d0397620ccaea8974cc764cd0953e",
    "tiny":     "/scratch/zt1/project/msml604/user/vyomwal5/anaconda3/envs/asr/hf_cache/models/models--openai--whisper-tiny/snapshots/169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
    "large-v3": "/scratch/zt1/project/msml604/user/vyomwal5/anaconda3/envs/asr/hf_cache/models/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1",
}
os.environ["HF_HOME"]               = CACHE_DIR
os.environ["HF_DATASETS_CACHE"]     = f"{CACHE_DIR}/datasets"
os.environ["TRANSFORMERS_CACHE"]    = f"{CACHE_DIR}/models"
os.environ["HUGGINGFACE_HUB_CACHE"] = f"{CACHE_DIR}/hub"
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import re
import torchaudio
import time
import json
import argparse
import numpy as np
import torch
import evaluate
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Union
from jiwer import wer as jiwer_wer
from datasets import ( load_dataset, Audio, IterableDataset, Dataset,
)
from transformers import ( WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
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

SUPPORTED_TASKS = ["asr", "translation"]

BENCHMARK_REGISTRY = {
    "librispeech": {
        "hf_path":       "librispeech_asr",
        "text_column":   "text",
        "language":      "English",
        "default_train": "train.100",
        "default_eval":  "validation",
        "default_test":  [
            ("clean", "test"),
            ("other", "test"),
        ],
        "task": "transcribe",
    },
    "common_voice": {
        "hf_path":       "mozilla-foundation/common_voice_13_0",
        "hf_config":     "en",
        "text_column":   "sentence",
        "language":      "English",
        "default_train": "train",
        "default_eval":  "validation",
        "default_test":  [("en", "test")],
        "task":          "transcribe",
    },
    "fleurs": {
        "hf_path":       "google/fleurs",
        "hf_config":     "en_us",
        "text_column":   "transcription",
        "language":      "English",
        "default_train": "train",
        "default_eval":  "validation",
        "default_test":  [("en_us", "test")],
        "task":          "transcribe",
    },
    "voxpopuli": {
        "hf_path":       "facebook/voxpopuli",
        "hf_config":     "en",
        "text_column":   "normalized_text",
        "language":      "English",
        "default_train": "train",
        "default_eval":  "validation",
        "default_test":  [("en", "test")],
        "task":          "transcribe",
    },
}

LORA_TARGET_MODULES = ["q_proj", "v_proj"]


# ─────────────────────────────────────────────────────────────────────────────
# SUBSAMPLING WRAPPER  (xV axis — Wang et al.)
# ─────────────────────────────────────────────────────────────────────────────

class WhisperWithTokenSubsampling(torch.nn.Module):
    """
    Wraps WhisperForConditionalGeneration (or a PeftModel around it) to
    subsample encoder hidden states BEFORE they are passed to the decoder's
    cross-attention layers.

    This is the xV axis from Wang et al.:
      tokens_per_frame=1 → all 1500 encoder frames reach the decoder  (baseline)
      tokens_per_frame=2 → every 2nd frame → 750 frames
      tokens_per_frame=4 → every 4th frame → 375 frames

    Cross-attention FLOPs scale as:
      FLOPs ∝ L_decoder × L_encoder × d_model
    so halving L_encoder halves decoder cross-attention cost.

    The encoder still processes the full (128, 3000) spectrogram — only the
    hidden states passed to the decoder are thinned.  To reduce encoder cost
    too, use --total_frames (xT axis), which clips the audio before encoding.

    Saving / loading
    ----------------
    Only the *inner* model's weights are saved (LoRA adapters or full weights).
    The wrapper itself has no learnable parameters and is re-applied at eval
    time by passing --tokens_per_frame to evaluate_checkpoint.
    """

    def __init__(self, base_model: torch.nn.Module, tokens_per_frame: int = 1):
        super().__init__()
        self.model            = base_model
        self.tokens_per_frame = tokens_per_frame

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _subsample(self, hidden: torch.Tensor) -> torch.Tensor:
        """Stride-subsample encoder hidden states along the time axis."""
        if self.tokens_per_frame > 1:
            hidden = hidden[:, :: self.tokens_per_frame, :]
        return hidden

    def _encode_and_subsample(self, input_features: torch.Tensor):
        """Run encoder, subsample, return patched BaseModelOutput."""
        # Access the underlying WhisperModel regardless of PEFT wrapping
        base = self.model
        # PeftModel wraps the model under .base_model.model; plain
        # WhisperForConditionalGeneration exposes .model directly.
        whisper_model = getattr(base, "model", base)           # WhisperModel
        encoder       = getattr(whisper_model, "model", whisper_model).encoder \
                        if hasattr(whisper_model, "model") \
                        else whisper_model.encoder

        encoder_out = encoder(input_features)
        encoder_out.last_hidden_state = self._subsample(
            encoder_out.last_hidden_state
        )
        return encoder_out

    # ------------------------------------------------------------------
    # forward  (used during training by Seq2SeqTrainer)
    # ------------------------------------------------------------------
    def forward(self, input_features, labels=None, **kwargs):
        encoder_out = self._encode_and_subsample(input_features)
        return self.model(
            encoder_outputs=encoder_out,
            labels=labels,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # generate  (used during inference / evaluation)
    # ------------------------------------------------------------------
    def generate(self, input_features, **kwargs):
        encoder_out = self._encode_and_subsample(input_features)
        return self.model.generate(
            encoder_outputs=encoder_out,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Delegate attribute access to inner model so that Trainer, PEFT,
    # processor helpers, etc. all continue to work transparently.
    # ------------------------------------------------------------------
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _librispeech_config_for_split(split: str) -> str:
    return "other" if "500" in split else "clean"


def load_benchmark_dataset(
    benchmark:       str,
    split:           str,
    sampling_rate:   int  = 16_000,
    streaming:       bool = False,
    max_samples:     Optional[int] = None,
    config_override: Optional[str] = None,
) -> Union[Dataset, IterableDataset]:
    if benchmark not in BENCHMARK_REGISTRY:
        raise ValueError(
            f"Unknown benchmark '{benchmark}'. "
            f"Choose from: {list(BENCHMARK_REGISTRY.keys())}"
        )
    info = BENCHMARK_REGISTRY[benchmark]

    if config_override is not None:
        config = config_override
    elif benchmark == "librispeech":
        config = _librispeech_config_for_split(split)
    else:
        config = info["hf_config"]

    print(
        f"Loading {benchmark} [{config}] split={split} "
        f"{'(streaming)' if streaming else '(cached)'}"
        + (f" max_samples={max_samples}" if max_samples else "")
    )

    dataset = load_dataset(
        info["hf_path"],
        config,
        split=split,
        streaming=streaming,
        trust_remote_code=True,
    )

    if max_samples is not None:
        if streaming:
            dataset = dataset.take(max_samples)
        else:
            n = min(max_samples, len(dataset))
            dataset = dataset.select(range(n))

    if not streaming:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    return dataset


def prepare_dataset(
    batch,
    processor,
    text_column:      str = "text",
    max_label_len:    int = 448,
    tokens_per_frame: int = 1,
    total_frames:     int = 1500,
    sampling_rate:    int = 16_000,
):
    """
    Converts raw audio + transcript into model inputs.

    Axis mapping (Wang et al.):
      xN → model_size      — handled at model build time
      xT → total_frames    — clips audio HERE (fewer active encoder frames)
      xV → tokens_per_frame — NOT applied here; applied in model.forward()
                              via WhisperWithTokenSubsampling

    Feature extractor always pads/truncates to (128, 3000) so Whisper's
    Conv1d layers and positional embeddings receive a valid fixed-size input
    even when audio has been clipped.
    """
    audio       = batch["audio"]
    audio_array = np.array(audio["array"], dtype=np.float32)
    audio_sr    = audio["sampling_rate"]

    # Resample if needed (critical for streaming mode)
    if audio_sr != sampling_rate:
        waveform    = torch.tensor(audio_array).unsqueeze(0)
        resampler   = torchaudio.transforms.Resample(audio_sr, sampling_rate)
        audio_array = resampler(waveform).squeeze(0).numpy().astype(np.float32)

    # xT axis: clip audio to control active duration sent to encoder
    #   total_frames=1500 → 30 s  (full window)
    #   total_frames=750  → 15 s
    #   total_frames=375  →  7.5 s
    max_audio_samples = int((total_frames / 1500) * 30 * sampling_rate)
    audio_array = audio_array[:max_audio_samples]

    # Feature extraction — always outputs (128, 3000)
    batch["input_features"] = processor.feature_extractor(
        audio_array,
        sampling_rate=sampling_rate,
    ).input_features[0]

    # Label tokenisation
    transcript = batch[text_column]
    if isinstance(transcript, str):
        transcript = transcript.lower()

    batch["labels"] = processor.tokenizer(
        transcript,
        max_length=max_label_len,
        truncation=True,
    ).input_ids

    return batch


def apply_preprocessing(
    dataset:          Union[Dataset, IterableDataset],
    processor,
    text_column:      str,
    max_label_len:    int,
    tokens_per_frame: int,
    total_frames:     int,
    num_proc:         int = 1,
    sampling_rate:    int = 16_000,
) -> Union[Dataset, IterableDataset]:
    map_fn = partial(
        prepare_dataset,
        processor=processor,
        text_column=text_column,
        max_label_len=max_label_len,
        tokens_per_frame=tokens_per_frame,
        total_frames=total_frames,
        sampling_rate=sampling_rate,
    )

    if isinstance(dataset, IterableDataset):
        dataset = dataset.map(
            map_fn,
            remove_columns=["file", "audio", text_column,
                             "speaker_id", "chapter_id", "id"],
            load_from_cache_file=False,
        )
    else:
        dataset = dataset.map(
            map_fn,
            remove_columns=dataset.column_names,
            num_proc=1,
            writer_batch_size=50,
            desc="Preprocessing",
            load_from_cache_file=False,
        )

    return dataset



@dataclass
class WhisperDataCollator:
    """
    Pads input_features and labels into batches.
    Labels padded with -100 so padding is ignored in cross-entropy loss.
    """
    processor: Any
    fp16: bool = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        if self.fp16:
            batch["input_features"] = batch["input_features"].half()

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def make_compute_metrics(processor):

    def normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s\']", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_str  = [normalize(p) for p in pred_str]
        label_str = [normalize(r) for r in label_str]
        wer = jiwer_wer(label_str, pred_str)
        return {"wer": round(100 * wer, 4)}

    return compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM TRAINER — fixes shared-tensor safetensors crash with PEFT/LoRA
# ─────────────────────────────────────────────────────────────────────────────

class PeftSafeSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Seq2SeqTrainer subclass that saves LoRA checkpoints correctly.

    Root cause of the crash
    -----------------------
    Whisper ties proj_out.weight ↔ embed_tokens.weight.  When a PeftModel
    (or a WhisperWithTokenSubsampling wrapper around one) is passed to the
    standard Trainer, its internal _save() calls safetensors.save_file(),
    which raises RuntimeError on shared-memory tensors.

    Fix
    ---
    Override _save() to:
      1. Unwrap WhisperWithTokenSubsampling if present.
      2. If the inner model is a PeftModel, use peft's save_pretrained()
         which handles tied weights safely.
      3. Otherwise fall back to the standard Trainer._save().
    """

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        model = self.model

        # Unwrap subsampling wrapper — it has no weights of its own
        if isinstance(model, WhisperWithTokenSubsampling):
            model = model.model

        if isinstance(model, PeftModel):
            # PEFT's save_pretrained handles tied weights correctly
            model.save_pretrained(output_dir)
            # Also save the tokenizer / processor config if the trainer holds it
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)
        else:
            # Full fine-tune: delegate to the standard HF save path
            super()._save(output_dir, state_dict)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL SETUP
# ─────────────────────────────────────────────────────────────────────────────

def _wrap_if_subsampling(
    model: torch.nn.Module,
    tokens_per_frame: int,
) -> torch.nn.Module:
    """
    Conditionally wrap model with WhisperWithTokenSubsampling.
    No-op when tokens_per_frame == 1 (baseline).
    """
    if tokens_per_frame > 1:
        model = WhisperWithTokenSubsampling(model, tokens_per_frame)
        n_out = 1500 // tokens_per_frame
        print(
            f"  [xV] Encoder output subsampled: stride={tokens_per_frame} "
            f"→ {n_out} tokens reach the decoder  "
            f"(cross-attn FLOPs ≈ {tokens_per_frame}× reduction)"
        )
    else:
        print("  [xV] No encoder subsampling (tokens_per_frame=1, baseline)")
    return model


def build_model_full(
    model_name:       str,
    fp16:             bool = False,
    tokens_per_frame: int  = 1,
) -> torch.nn.Module:
    """All weights trainable, optional xV subsampling wrapper."""
    dtype = torch.float16 if fp16 else torch.float32
    print(f"Loading {model_name} (full finetune)...")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype
    )
    model.config.forced_decoder_ids        = None
    model.generation_config.suppress_tokens = []
    model.config.use_cache                 = False

    total = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total:,} | All trainable")

    return _wrap_if_subsampling(model, tokens_per_frame)


def build_model_lora(
    model_name:       str,
    lora_r:           int   = 32,
    lora_alpha:       int   = 64,
    lora_dropout:     float = 0.05,
    fp16:             bool  = False,
    tokens_per_frame: int   = 1,
) -> torch.nn.Module:
    """
    LoRA on q_proj + v_proj across encoder self-attn, decoder self-attn,
    and decoder cross-attn.  Optional xV subsampling wrapper applied on top.
    """
    dtype = torch.float16 if fp16 else torch.float32
    print(f"Loading {model_name} (LoRA r={lora_r}, alpha={lora_alpha})...")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype
    )
    model.config.forced_decoder_ids        = None
    model.generation_config.suppress_tokens = []
    model.config.use_cache                 = False

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return _wrap_if_subsampling(model, tokens_per_frame)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    benchmark  = getattr(args, "benchmark_dataset", "librispeech")
    task       = getattr(args, "task",              "asr")
    bench_info = BENCHMARK_REGISTRY.get(benchmark, BENCHMARK_REGISTRY["librispeech"])

    train_split      = getattr(args, "train_split", None) or bench_info["default_train"]
    eval_split       = getattr(args, "eval_split",  None) or bench_info["default_eval"]
    text_column      = bench_info["text_column"]
    language         = bench_info["language"]
    whisper_task     = bench_info["task"]
    tokens_per_frame = getattr(args, "tokens_per_frame", 1)
    total_frames     = getattr(args, "total_frames",     1500)

    streaming         = getattr(args, "streaming",         False)
    max_train_samples = getattr(args, "max_train_samples", None)
    max_eval_samples  = getattr(args, "max_eval_samples",  None)

    model_name = local_path[args.model_size]
    run_name   = (
        f"whisper-{args.model_size}-{args.mode}-{benchmark}-{task}"
        f"-tpf{tokens_per_frame}-tf{total_frames}-v100"
    )
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nRun             : {run_name}")
    print(f"Cache directory : {CACHE_DIR}")
    print(f"Streaming mode  : {streaming}")
    print(f"[xT] total_frames     = {total_frames}  "
          f"(active audio = {total_frames/1500*30:.1f}s)")
    print(f"[xV] tokens_per_frame = {tokens_per_frame}  "
          f"(decoder sees {1500//tokens_per_frame} encoder tokens)")
    if max_train_samples:
        print(f"Max train samples : {max_train_samples}")
    if max_eval_samples:
        print(f"Max eval  samples : {max_eval_samples}")

    processor = WhisperProcessor.from_pretrained(
        model_name, language=language, task=whisper_task
    )

    train_dataset = load_benchmark_dataset(
        benchmark, train_split,
        streaming=streaming,
        max_samples=max_train_samples,
    )
    eval_dataset = load_benchmark_dataset(
        benchmark, eval_split,
        streaming=streaming,
        max_samples=max_eval_samples,
    )

    print("Preprocessing datasets...")
    train_dataset = apply_preprocessing(
        train_dataset, processor,
        text_column=text_column,
        max_label_len=args.max_label_len,
        tokens_per_frame=tokens_per_frame,
        total_frames=total_frames,
        num_proc=args.num_proc if not streaming else 1,
    )
    eval_dataset = apply_preprocessing(
        eval_dataset, processor,
        text_column=text_column,
        max_label_len=args.max_label_len,
        tokens_per_frame=tokens_per_frame,
        total_frames=total_frames,
        num_proc=args.num_proc if not streaming else 1,
    )

    # ── Build model (with subsampling wrapper if tokens_per_frame > 1) ────────
    if args.mode == "full":
        model = build_model_full(
            model_name,
            fp16=args.fp16,
            tokens_per_frame=tokens_per_frame,
        )
    elif args.mode == "lora":
        model = build_model_lora(
            model_name,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            fp16=args.fp16,
            tokens_per_frame=tokens_per_frame,
        )
    else:
        raise ValueError(f"Unknown mode '{args.mode}'. Choose 'full' or 'lora'.")

    collator = WhisperDataCollator(processor=processor, fp16=args.fp16)

    is_streaming = isinstance(train_dataset, IterableDataset)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,

        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="linear",

        max_steps=args.max_steps,

        fp16=args.fp16 and torch.cuda.is_available(),
        gradient_checkpointing=True,

        eval_strategy="steps",
        eval_steps=args.eval_steps,
        predict_with_generate=True,
        generation_max_length=args.max_label_len,

        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=(not is_streaming and args.mode == "full"),  # False for LoRA: PEFT saves adapter_model.safetensors, not pytorch_model.bin
        metric_for_best_model="wer",
        greater_is_better=False,

        logging_steps=args.log_steps,
        report_to=["tensorboard"],
        run_name=run_name,

        dataloader_num_workers=0 if is_streaming else args.num_proc,
        remove_unused_columns=False,
    )

    trainer = PeftSafeSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=make_compute_metrics(processor),
        processing_class=processor.feature_extractor,
    )

    print(f"\n{'='*60}")
    print(f"  Run     : {run_name}")
    print(f"  Mode    : {args.mode} | Streaming: {is_streaming}")
    if not is_streaming:
        print(f"  Train   : {len(train_dataset):,} samples")
        print(f"  Eval    : {len(eval_dataset):,} samples")
    print(f"{'='*60}\n")

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/3600:.2f}h")

    # ── Save ──────────────────────────────────────────────────────────────────
    # For LoRA: load_best_model_at_end is disabled (PEFT/safetensors conflict),
    # so we manually find the best checkpoint from trainer state and copy it
    # to output_dir.  For full fine-tune, trainer.save_model() is sufficient.
    if args.mode == "lora":
        best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
        if best_ckpt and os.path.isdir(best_ckpt):
            print(f"  Best checkpoint: {best_ckpt}  (copying to {output_dir})")
            import shutil
            for fname in os.listdir(best_ckpt):
                src = os.path.join(best_ckpt, fname)
                dst = os.path.join(output_dir, fname)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
        else:
            # Fallback: save current model weights (last checkpoint)
            print("  No best_model_checkpoint recorded; saving current weights.")
            trainer.save_model(output_dir)
    else:
        trainer.save_model(output_dir)

    processor.save_pretrained(output_dir)

    # Save tokens_per_frame and total_frames into the checkpoint directory
    # so evaluate_checkpoint can reload them automatically
    experiment_cfg = {
        "tokens_per_frame": tokens_per_frame,
        "total_frames":     total_frames,
        "model_size":       args.model_size,
        "mode":             args.mode,
    }
    with open(os.path.join(output_dir, "experiment_cfg.json"), "w") as f:
        json.dump(experiment_cfg, f, indent=2)

    _inner    = model.model if isinstance(model, WhisperWithTokenSubsampling) else model
    trainable = sum(p.numel() for p in _inner.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in _inner.parameters())
    meta = {
        "model_size":        args.model_size,
        "mode":              args.mode,
        "task":              task,
        "benchmark_dataset": benchmark,
        "tokens_per_frame":  tokens_per_frame,
        "total_frames":      total_frames,
        "streaming":         is_streaming,
        "max_train_samples": max_train_samples,
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


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s\']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def evaluate_checkpoint(args):
    """
    Load a saved checkpoint and run WER + RTF evaluation on test splits.

    Automatically reads tokens_per_frame and total_frames from the checkpoint's
    experiment_cfg.json (written by train()).  Command-line values override
    if explicitly provided.

    Key fix vs original: total_frames truncation is NOW applied during eval
    (previously eval always used full 30 s audio regardless of total_frames).
    """
    checkpoint_dir = args.checkpoint
    assert checkpoint_dir, "--checkpoint required for eval_only mode"

    # ── Load experiment config saved by train() ───────────────────────────────
    cfg_path = os.path.join(checkpoint_dir, "experiment_cfg.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            saved_cfg = json.load(f)
        print(f"Loaded experiment config from checkpoint: {saved_cfg}")
    else:
        saved_cfg = {}
        print("No experiment_cfg.json found; using args values.")

    # Command-line args override saved config (so you can test cross-settings)
    tokens_per_frame = getattr(args, "tokens_per_frame", None) \
                       or saved_cfg.get("tokens_per_frame", 1)
    total_frames     = getattr(args, "total_frames", None) \
                       or saved_cfg.get("total_frames", 1500)

    print(f"[xT] total_frames     = {total_frames}  "
          f"(eval audio clipped to {total_frames/1500*30:.1f}s)")
    print(f"[xV] tokens_per_frame = {tokens_per_frame}  "
          f"(decoder sees {1500//tokens_per_frame} encoder tokens)")

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = local_path[args.model_size]
    benchmark  = getattr(args, "benchmark_dataset", "librispeech")
    bench_info = BENCHMARK_REGISTRY.get(benchmark, BENCHMARK_REGISTRY["librispeech"])
    text_column = bench_info["text_column"]

    proc_path = (
        checkpoint_dir
        if os.path.exists(os.path.join(checkpoint_dir, "preprocessor_config.json"))
        else model_name
    )
    processor = WhisperProcessor.from_pretrained(
        proc_path, language="English", task="transcribe"
    )

    # ── Load inner model (no wrapper yet) ────────────────────────────────────
    dtype = torch.float16 if getattr(args, "fp16", True) else torch.float32

    if args.mode == "lora":
        base = WhisperForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype
        )
        base.config.forced_decoder_ids = None
        model = PeftModel.from_pretrained(base, checkpoint_dir)
        model = model.merge_and_unload()   # fuse LoRA → plain WhisperForConditionalGeneration
    else:
        model = WhisperForConditionalGeneration.from_pretrained(
            checkpoint_dir, torch_dtype=dtype
        )

    model.config.forced_decoder_ids = None

    # ── Re-apply subsampling wrapper if needed (xV axis) ─────────────────────
    # This MUST happen after merge_and_unload() so the wrapper sees a plain
    # WhisperForConditionalGeneration, not a PeftModel.
    if tokens_per_frame > 1:
        model = WhisperWithTokenSubsampling(model, tokens_per_frame)
        print(f"  Subsampling wrapper applied: stride={tokens_per_frame}")

    model = model.to(device).eval()

    # ── Load test splits ──────────────────────────────────────────────────────
    if benchmark == "librispeech":
        eval_splits = {
            "test_clean": load_benchmark_dataset(
                benchmark, "test", streaming=True, config_override="clean",
                max_samples=getattr(args, "max_eval_samples", None),
            ),
            "test_other": load_benchmark_dataset(
                benchmark, "test", streaming=True, config_override="other",
                max_samples=getattr(args, "max_eval_samples", None),
            ),
        }
    else:
        eval_splits = {}
        for config, split in bench_info["default_test"]:
            key = f"{config}_{split}"
            eval_splits[key] = load_benchmark_dataset(
                benchmark, split, streaming=True,
                max_samples=getattr(args, "max_eval_samples", None),
            )

    results = {}
    fp16    = getattr(args, "fp16", True)

    # Pre-compute audio clip length in samples (xT axis applied at eval too)
    max_audio_samples = int((total_frames / 1500) * 30 * 16_000)

    for split_name, dataset in eval_splits.items():
        print(f"\nEvaluating on {split_name} ...")
        all_preds, all_refs = [], []
        total_audio_s = 0.0
        total_infer_s = 0.0

        for sample in tqdm.tqdm(dataset):
            audio_array = sample["audio"]["array"].astype(np.float32)
            ref_text    = sample[text_column].lower().strip()

            # xT: clip audio to match training condition
            audio_array   = audio_array[:max_audio_samples]
            audio_dur_s   = len(audio_array) / 16_000

            features = processor.feature_extractor(
                audio_array, sampling_rate=16_000, return_tensors="pt"
            ).input_features.to(device)

            if fp16:
                features = features.half()

            t0 = time.time()
            with torch.no_grad():
                pred_ids = model.generate(
                    features,
                    language="en",
                    task="transcribe",
                    num_beams=1,
                    max_new_tokens=256,
                )
            infer_time = time.time() - t0

            pred_text = processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True)
            all_preds.append(normalize_text(pred_text))
            all_refs.append(normalize_text(ref_text))
            total_audio_s += audio_dur_s
            total_infer_s += infer_time

        wer = jiwer_wer(all_refs, all_preds)
        rtf = total_infer_s / total_audio_s
        results[split_name] = {
            "wer":           round(100 * wer, 3),
            "rtf":           round(rtf, 4),
            "n_samples":     len(all_preds),
            "total_audio_h": round(total_audio_s / 3600, 2),
        }
        print(f"  WER: {results[split_name]['wer']:.3f}% | RTF: {rtf:.4f}")

    total_params = sum(
        p.numel() for p in
        (model.model.parameters() if isinstance(model, WhisperWithTokenSubsampling)
         else model.parameters())
    )
    summary = {
        "model_size":        args.model_size,
        "mode":              args.mode,
        "tokens_per_frame":  tokens_per_frame,
        "total_frames":      total_frames,
        "total_params":      total_params,
        "results":           results,
    }
    print("\n" + "=" * 60)
    print(json.dumps(summary, indent=2))

    out_path = os.path.join(checkpoint_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {out_path}")
    return summary


def run_sweep(args):
    sizes = args.sweep_sizes.split(",")
    modes = args.sweep_modes.split(",")
    all_results = []

    for size in sizes:
        for mode in modes:
            print(f"\n{'#'*60}\n  SWEEP: whisper-{size} | mode={mode}\n{'#'*60}")
            args.model_size = size
            args.mode       = mode

            checkpoint_dir = train(args)
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            eval_summary = evaluate_checkpoint(
                argparse.Namespace(
                    model_size=size,
                    mode=mode,
                    checkpoint=checkpoint_dir,
                    benchmark_dataset=getattr(args, "benchmark_dataset", "librispeech"),
                    max_eval_samples=getattr(args, "max_eval_samples", None),
                    fp16=getattr(args, "fp16", True),
                    # tokens_per_frame / total_frames loaded from experiment_cfg.json
                    tokens_per_frame=None,
                    total_frames=None,
                )
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            meta_path = os.path.join(checkpoint_dir, "run_meta.json")
            with open(meta_path) as f:
                meta = json.load(f)

            all_results.append({**meta, **eval_summary["results"]})

            sweep_path = os.path.join(args.output_dir, "sweep_results.json")
            with open(sweep_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"Sweep results saved to {sweep_path}")

    print("\n" + "=" * 60 + "\nSWEEP COMPLETE\n" + "=" * 60)
    for r in all_results:
        print(
            f"  whisper-{r['model_size']:10s} | {r['mode']:5s} | "
            f"tpf={r['tokens_per_frame']} tf={r['total_frames']} | "
            f"trainable={r['trainable_params']:>12,} | "
            f"WER test_clean={r.get('test_clean', {}).get('wer', 'N/A')}%"
        )



def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Whisper Finetuning: full vs LoRA, multi-benchmark, streaming",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Experiment identity
    p.add_argument("--task",              type=str, default="asr",         choices=SUPPORTED_TASKS)
    p.add_argument("--benchmark_dataset", type=str, default="librispeech", choices=list(BENCHMARK_REGISTRY.keys()))
    p.add_argument("--tokens_per_frame",  type=int, default=1,
                   help="xV axis: subsample encoder hidden states by this stride before decoder. "
                        "1=baseline (1500 tokens), 2=750 tokens, 4=375 tokens")
    p.add_argument("--total_frames",      type=int, default=1500,
                   help="xT axis: clip audio to (total_frames/1500)*30s before encoding. "
                        "1500=30s full, 750=15s, 375=7.5s")

    # Model
    p.add_argument("--model_size", type=str, default="small", choices=list(WHISPER_SIZES.keys()))
    p.add_argument("--mode",       type=str, default="lora",  choices=["full", "lora"])

    # Data
    p.add_argument("--train_split",       type=str,  default=None)
    p.add_argument("--eval_split",        type=str,  default=None)
    p.add_argument("--max_label_len",     type=int,  default=448)
    p.add_argument("--num_proc",          type=int,  default=1)
    p.add_argument("--streaming",         action="store_true")
    p.add_argument("--max_train_samples", type=int,  default=None)
    p.add_argument("--max_eval_samples",  type=int,  default=None)

    # LoRA
    p.add_argument("--lora_r",       type=int,   default=32)
    p.add_argument("--lora_alpha",   type=int,   default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Training
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--eval_batch_size", type=int,   default=8)
    p.add_argument("--grad_accum",      type=int,   default=2)
    p.add_argument("--learning_rate",   type=float, default=1e-5)
    p.add_argument("--warmup_steps",    type=int,   default=500)
    p.add_argument("--max_steps",       type=int,   default=4000)
    p.add_argument("--num_epochs",      type=int,   default=3)
    p.add_argument("--eval_steps",      type=int,   default=500)
    p.add_argument("--log_steps",       type=int,   default=25)
    p.add_argument("--fp16",            action="store_true", default=True)

    # Output
    p.add_argument("--output_dir", type=str, default="/home/vyomwal5/SAME/checkpoints")

    # Eval only
    p.add_argument("--eval_only",  action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)

    # Sweep
    p.add_argument("--sweep",       action="store_true")
    p.add_argument("--sweep_sizes", type=str, default="small,medium,large-v3")
    p.add_argument("--sweep_modes", type=str, default="full,lora")

    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.sweep:
        run_sweep(args)
    elif args.eval_only:
        evaluate_checkpoint(args)
    else:
        train(args)