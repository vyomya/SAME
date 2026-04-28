import os

BASE = "/scratch/zt1/project/msml604/user/mokshdag/miniconda3/envs/same"
_lib_paths = [
    f"{BASE}/lib/python3.11/site-packages/nvidia/nccl/lib",
    f"{BASE}/lib",
    f"{BASE}/lib/python3.11/site-packages/torch/lib",
    f"{BASE}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib",
    f"{BASE}/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib",
    f"{BASE}/lib/python3.11/site-packages/nvidia/npp/lib",
]
existing = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = ":".join(_lib_paths) + (":" + existing if existing else "")

CACHE_DIR = "/scratch/zt1/project/msml604/user/mokshdag/hf_cache"

os.environ["HF_HOME"]                = CACHE_DIR
os.environ["HF_DATASETS_CACHE"]      = f"{CACHE_DIR}/datasets"
os.environ["TRANSFORMERS_CACHE"]     = f"{CACHE_DIR}/models"
os.environ["HUGGINGFACE_HUB_CACHE"]  = f"{CACHE_DIR}/models"
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"

# Local paths
LOCAL_MODEL_PATH = {
    "facebook/wav2vec2-large-robust": f"{CACHE_DIR}/models/wav2vec2-large-robust",
}

CREMAD_AUDIO_DIR = f"{CACHE_DIR}/datasets/crema-d/data/data/AudioWAV"

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import glob
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
import torchaudio
import tqdm
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

MODEL_REGISTRY = {
    "wav2vec2-large-robust": "facebook/wav2vec2-large-robust",
    "hubert-large":          "facebook/hubert-large-ls960-ft",
    "wav2vec2-base":         "facebook/wav2vec2-base",
    "wavlm-large":           "microsoft/wavlm-large",
    "wav2vec2-large":        "facebook/wav2vec2-large-960h",
}

SUPPORTED_TASKS = ["emotion_recognition"]

BENCHMARK_REGISTRY = {
    "cremad": {
        "local_audio_dir": CREMAD_AUDIO_DIR,
        "label_column":    "emotion",
        "label_map": {
            "NEU": 0,
            "HAP": 1,
            "SAD": 2,
            "ANG": 3,
            "FEA": 4,
            "DIS": 5,
        },
        "num_labels":    6,
        "label_names":   ["neutral", "happy", "sad", "angry", "fear", "disgust"],
        "default_train": "train",
        "default_eval":  "validation",
        "default_test":  "test",
        "max_audio_len": 6.0,
    },
}

LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "out_proj"]


# -----------------------------------------------------------------------------
# LAZY PYTORCH DATASET
# -----------------------------------------------------------------------------

class CremaDataset(torch.utils.data.Dataset):


    def __init__(
        self,
        audio_dir:     str,
        split:         str,
        feature_extractor,
        label_map:     Dict,
        max_audio_len: float = 6.0,
        sampling_rate: int   = 16_000,
        max_samples:   Optional[int] = None,
        seed:          int   = 42,
    ):
        self.feature_extractor = feature_extractor
        self.label_map         = label_map
        self.max_len           = int(max_audio_len * sampling_rate)
        self.sampling_rate     = sampling_rate

        # Build file list and labels (paths only, no audio loaded yet)
        all_wavs = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
        if not all_wavs:
            raise FileNotFoundError(f"No wav files found in {audio_dir}")

        rng     = np.random.default_rng(seed)
        indices = rng.permutation(len(all_wavs))
        n       = len(all_wavs)
        n_train = int(0.8 * n)
        n_val   = int(0.1 * n)

        if split == "train":
            chosen = indices[:n_train]
        elif split == "validation":
            chosen = indices[n_train:n_train + n_val]
        elif split == "test":
            chosen = indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split '{split}'")

        if max_samples is not None:
            chosen = chosen[:max_samples]

        self.samples = []
        for idx in chosen:
            path    = all_wavs[idx]
            fname   = os.path.basename(path)
            emotion = fname.replace(".wav", "").split("_")[2]
            label   = label_map.get(emotion, -1)
            if label == -1:
                continue
            self.samples.append((path, label))

        print(f"  CremaDataset [{split}]: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # Read audio from disk lazily here (not at init time)
        audio_array, sr = sf.read(path, dtype="float32")

        if sr != self.sampling_rate:
            waveform    = torch.tensor(audio_array).unsqueeze(0)
            resampler   = torchaudio.transforms.Resample(sr, self.sampling_rate)
            audio_array = resampler(waveform).squeeze(0).numpy()

        if len(audio_array) > self.max_len:
            audio_array = audio_array[:self.max_len]

        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=self.sampling_rate,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )

        item = {"input_values": inputs.input_values[0], "label": label}
        if "attention_mask" in inputs:
            item["attention_mask"] = inputs.attention_mask[0]
        return item

@dataclass
class EmotionDataCollator:
    """
    Stacks pre-padded input_values and labels into batches.
    Since CremaDataset already pads to max_length, collation is just stacking.
    """
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {
            "input_values": torch.tensor(
                np.stack([f["input_values"] for f in features]), dtype=torch.float32
            ),
            "labels": torch.tensor(
                [f["label"] for f in features], dtype=torch.long
            ),
        }
        if "attention_mask" in features[0]:
            batch["attention_mask"] = torch.tensor(
                np.stack([f["attention_mask"] for f in features]), dtype=torch.long
            )
        return batch


def make_compute_metrics(label_names: List[str]):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        ua  = f1_score(labels, preds, average="macro",    zero_division=0)
        wa  = f1_score(labels, preds, average="weighted", zero_division=0)
        return {
            "accuracy":    round(acc, 4),
            "ua":          round(ua,  4),
            "weighted_f1": round(wa,  4),
        }
    return compute_metrics


# -----------------------------------------------------------------------------
# MODEL SETUP
# -----------------------------------------------------------------------------

def build_model_full(model_name, num_labels, label_names):
    print(f"Loading {model_name} (full finetune, {num_labels} classes)...")
    model = AutoModelForAudioClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={i: n for i, n in enumerate(label_names)},
        label2id={n: i for i, n in enumerate(label_names)},
        ignore_mismatched_sizes=True,
    )
    for param in model.parameters():
        param.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total:,} | All trainable")
    return model


def build_model_lora(model_name, num_labels, label_names,
                     lora_r=16, lora_alpha=32, lora_dropout=0.05,
                     unfreeze_top_layers=4):
    
    print(f"Loading {model_name} (LoRA r={lora_r}, alpha={lora_alpha}, "
          f"unfreeze_top={unfreeze_top_layers}, {num_labels} classes)...")
    model = AutoModelForAudioClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={i: n for i, n in enumerate(label_names)},
        label2id={n: i for i, n in enumerate(label_names)},
        ignore_mismatched_sizes=True,
    )

    for param in model.parameters():
        param.requires_grad = False

    try:
        task_type = TaskType.AUDIO_CLASSIFICATION
    except AttributeError:
        task_type = TaskType.FEATURE_EXTRACTION

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        task_type=task_type,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    encoder_layers = model.base_model.model.wav2vec2.encoder.layers
    total_layers   = len(encoder_layers)
    unfreeze_from  = max(0, total_layers - unfreeze_top_layers)
    print(f"  Encoder has {total_layers} layers, unfreezing top {unfreeze_top_layers} "
          f"(layers {unfreeze_from}-{total_layers-1})")
    for layer in encoder_layers[unfreeze_from:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Step 4: always unfreeze classifier head + projector
    for name, param in model.named_parameters():
        if "classifier" in name or "projector" in name:
            param.requires_grad = True

    model.print_trainable_parameters()
    return model



def train(args):
    benchmark   = getattr(args, "benchmark_dataset", "cremad")
    bench_info  = BENCHMARK_REGISTRY[benchmark]
    model_name  = LOCAL_MODEL_PATH.get(args.model_name, args.model_name)

    train_split   = getattr(args, "train_split", None) or bench_info["default_train"]
    eval_split    = getattr(args, "eval_split",  None) or bench_info["default_eval"]
    num_labels    = bench_info["num_labels"]
    label_names   = bench_info["label_names"]
    max_audio_len = getattr(args, "max_audio_len", None) or bench_info["max_audio_len"]

    max_train_samples = getattr(args, "max_train_samples", None)
    max_eval_samples  = getattr(args, "max_eval_samples",  None)

    model_slug = model_name.replace("/", "-").replace("_", "-")
    run_name   = (
        f"ser-{model_slug}-{args.mode}-{benchmark}-"
        f"r{args.lora_r if args.mode == 'lora' else 'full'}-"
        f"audio{max_audio_len}s"
    )
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nCache directory : {CACHE_DIR}")
    print(f"Model           : {model_name}")
    print(f"Max audio len   : {max_audio_len}s")

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    # Lazy datasets - NO map(), NO RAM spike
    print("Building lazy datasets (audio read on-the-fly from disk)...")
    train_dataset = CremaDataset(
        audio_dir=bench_info["local_audio_dir"],
        split=train_split,
        feature_extractor=feature_extractor,
        label_map=bench_info["label_map"],
        max_audio_len=max_audio_len,
        max_samples=max_train_samples,
    )
    eval_dataset = CremaDataset(
        audio_dir=bench_info["local_audio_dir"],
        split=eval_split,
        feature_extractor=feature_extractor,
        label_map=bench_info["label_map"],
        max_audio_len=max_audio_len,
        max_samples=max_eval_samples,
    )

    if args.mode == "full":
        model = build_model_full(model_name, num_labels, label_names)
    elif args.mode == "lora":
        model = build_model_lora(
            model_name, num_labels, label_names,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            unfreeze_top_layers=args.unfreeze_top_layers,
        )
    else:
        raise ValueError(f"Unknown mode '{args.mode}'")

    collator = EmotionDataCollator()

    training_args = TrainingArguments(
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

        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="ua",
        greater_is_better=True,

        logging_steps=args.log_steps,
        report_to=[],
        run_name=run_name,

        dataloader_num_workers=1,
        remove_unused_columns=False,
    )


    if args.mode == "lora":
        
        head_params    = [(n, p) for n, p in model.named_parameters()
                          if ("classifier" in n or "projector" in n) and p.requires_grad]
        encoder_params = [(n, p) for n, p in model.named_parameters()
                          if "encoder.layers" in n
                          and "lora_" not in n
                          and "classifier" not in n
                          and "projector" not in n
                          and p.requires_grad]
        lora_params    = [(n, p) for n, p in model.named_parameters()
                          if "lora_" in n and p.requires_grad]

        print(f"  Three-group optimizer:")
        print(f"    head    : {sum(p.numel() for _,p in head_params):>10,} params  lr=1e-3")
        print(f"    encoder : {sum(p.numel() for _,p in encoder_params):>10,} params  lr=5e-5")
        print(f"    lora    : {sum(p.numel() for _,p in lora_params):>10,} params  lr={args.learning_rate}")

        optimizer = torch.optim.AdamW([
            {"params": [p for _,p in head_params],    "lr": 1e-3},
            {"params": [p for _,p in encoder_params], "lr": 5e-5},
            {"params": [p for _,p in lora_params],    "lr": args.learning_rate},
        ], weight_decay=0.01)

        # Linear warmup then linear decay — same schedule Trainer uses internally
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.max_steps,
        )
        optimizers = (optimizer, scheduler)
    else:
        optimizers = (None, None)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=make_compute_metrics(label_names),
        processing_class=feature_extractor,
        optimizers=optimizers,
    )

    print(f"\n{'='*60}")
    print(f"  Run    : {run_name}")
    print(f"  Mode   : {args.mode}")
    print(f"  Labels : {label_names}")
    print(f"  Train  : {len(train_dataset):,} samples")
    print(f"  Eval   : {len(eval_dataset):,} samples")
    print(f"{'='*60}\n")

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/3600:.2f}h")

    if args.mode == "lora":
        model.save_pretrained(output_dir)
    else:
        trainer.save_model(output_dir)
    feature_extractor.save_pretrained(output_dir)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    meta = {
        "model_name":        model_name,
        "mode":              args.mode,
        "benchmark_dataset": benchmark,
        "num_labels":        num_labels,
        "label_names":       label_names,
        "max_audio_len":     max_audio_len,
        "lora_r":            args.lora_r if args.mode == "lora" else None,
        "trainable_params":  trainable,
        "total_params":      total,
        "trainable_pct":     round(100 * trainable / total, 4),
        "training_hours":    round(elapsed / 3600, 3),
    }
    with open(os.path.join(output_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))

    return output_dir


def evaluate_checkpoint(args):
    checkpoint_dir = args.checkpoint
    assert checkpoint_dir, "--checkpoint required for eval_only mode"

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name  = LOCAL_MODEL_PATH.get(args.model_name, args.model_name)
    benchmark   = getattr(args, "benchmark_dataset", "cremad")
    bench_info  = BENCHMARK_REGISTRY[benchmark]
    num_labels  = bench_info["num_labels"]
    label_names = bench_info["label_names"]
    max_audio_len = getattr(args, "max_audio_len", None) or bench_info["max_audio_len"]

    fe_path = (
        checkpoint_dir
        if os.path.exists(os.path.join(checkpoint_dir, "preprocessor_config.json"))
        else model_name
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(fe_path)

    if args.mode == "lora":
        base = AutoModelForAudioClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label={i: n for i, n in enumerate(label_names)},
            label2id={n: i for i, n in enumerate(label_names)},
            ignore_mismatched_sizes=True,
        )
        model = PeftModel.from_pretrained(base, checkpoint_dir)
        model = model.merge_and_unload()
    else:
        model = AutoModelForAudioClassification.from_pretrained(
            checkpoint_dir,
            num_labels=num_labels,
            id2label={i: n for i, n in enumerate(label_names)},
            label2id={n: i for i, n in enumerate(label_names)},
            ignore_mismatched_sizes=True,
        )

    model = model.to(device).eval()

    test_dataset = CremaDataset(
        audio_dir=bench_info["local_audio_dir"],
        split=bench_info["default_test"],
        feature_extractor=feature_extractor,
        label_map=bench_info["label_map"],
        max_audio_len=max_audio_len,
        max_samples=getattr(args, "max_eval_samples", None),
    )

    all_preds, all_labels = [], []
    total_audio_s = 0.0
    total_infer_s = 0.0

    print(f"\nEvaluating on {benchmark} test split ({len(test_dataset)} samples)...")
    for item in tqdm.tqdm(test_dataset):
        input_values = torch.tensor(item["input_values"]).unsqueeze(0).to(device)
        label        = item["label"]
        audio_dur_s  = len(item["input_values"]) / 16_000

        t0 = time.time()
        with torch.no_grad():
            logits = model(input_values=input_values).logits
        total_infer_s += time.time() - t0

        all_preds.append(logits.argmax(dim=-1).item())
        all_labels.append(label)
        total_audio_s += audio_dur_s

    acc = accuracy_score(all_labels, all_preds)
    ua  = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    wa  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    rtf = total_infer_s / max(total_audio_s, 1e-6)
    report = classification_report(all_labels, all_preds,
                                   target_names=label_names, zero_division=0)

    print(f"\nAccuracy : {100*acc:.2f}%")
    print(f"UA       : {100*ua:.2f}%")
    print(f"WA       : {100*wa:.2f}%")
    print(f"RTF      : {rtf:.4f}")
    print("\n" + report)

    summary = {
        "model_name":            model_name,
        "mode":                  args.mode,
        "benchmark":             benchmark,
        "n_samples":             len(all_preds),
        "accuracy":              round(100 * acc, 3),
        "ua":                    round(100 * ua,  3),
        "weighted_f1":           round(100 * wa,  3),
        "rtf":                   round(rtf, 4),
        "confusion_matrix":      confusion_matrix(all_labels, all_preds).tolist(),
        "classification_report": report,
    }
    out_path = os.path.join(checkpoint_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump({k: v for k, v in summary.items()
                   if k != "classification_report"}, f, indent=2)
    print(f"Results saved to {out_path}")
    return summary


def run_sweep(args):
    models = args.sweep_models.split(",")
    modes  = args.sweep_modes.split(",")
    all_results = []

    for model_name in models:
        for mode in modes:
            print(f"\n{'#'*60}\n  SWEEP: {model_name} | mode={mode}\n{'#'*60}")
            args.model_name = model_name
            args.mode       = mode
            checkpoint_dir  = train(args)
            eval_summary    = evaluate_checkpoint(
                argparse.Namespace(
                    model_name=model_name,
                    mode=mode,
                    checkpoint=checkpoint_dir,
                    benchmark_dataset=getattr(args, "benchmark_dataset", "cremad"),
                    max_eval_samples=getattr(args, "max_eval_samples", None),
                    max_audio_len=getattr(args, "max_audio_len", None),
                )
            )
            meta_path = os.path.join(checkpoint_dir, "run_meta.json")
            with open(meta_path) as f:
                meta = json.load(f)
            all_results.append({**meta, "ua": eval_summary["ua"],
                                 "wa": eval_summary["weighted_f1"]})
            sweep_path = os.path.join(args.output_dir, "sweep_results.json")
            with open(sweep_path, "w") as f:
                json.dump(all_results, f, indent=2)

    print("\n" + "="*60 + "\nSWEEP COMPLETE\n" + "="*60)
    for r in all_results:
        print(f"  {r['model_name']:40s} | {r['mode']:5s} | UA={r.get('ua','N/A')}%")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Speech Emotion Recognition: wav2vec2/HuBERT/WavLM, full vs LoRA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--benchmark_dataset", type=str, default="cremad",
                   choices=list(BENCHMARK_REGISTRY.keys()))
    p.add_argument("--model_name",        type=str,
                   default="facebook/wav2vec2-large-robust")
    p.add_argument("--mode",              type=str, default="lora",
                   choices=["full", "lora"])
    p.add_argument("--max_audio_len",     type=float, default=None)

    p.add_argument("--train_split",       type=str,  default=None)
    p.add_argument("--eval_split",        type=str,  default=None)
    p.add_argument("--num_proc",          type=int,  default=1)
    p.add_argument("--streaming",         action="store_true")
    p.add_argument("--max_train_samples", type=int,  default=None)
    p.add_argument("--max_eval_samples",  type=int,  default=None)

    p.add_argument("--lora_r",       type=int,   default=16)   # reduced from 32
    p.add_argument("--lora_alpha",   type=int,   default=32)   # alpha/r ratio = 2
    p.add_argument("--lora_dropout",         type=float, default=0.05)
    p.add_argument("--unfreeze_top_layers",  type=int,   default=4,
                   help="Number of top encoder transformer layers to unfreeze (0=LoRA only, 4=recommended)")

    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--eval_batch_size", type=int,   default=8)
    p.add_argument("--grad_accum",      type=int,   default=2)
    p.add_argument("--learning_rate",   type=float, default=1e-4)
    p.add_argument("--warmup_steps",    type=int,   default=100)   # reduced: head needs to move early
    p.add_argument("--max_steps",       type=int,   default=3000)
    p.add_argument("--eval_steps",      type=int,   default=300)
    p.add_argument("--log_steps",       type=int,   default=25)
    p.add_argument("--fp16",            action="store_true", default=True)

    p.add_argument("--output_dir", type=str,
                   default="/home/mokshdag/checkpoints/emotion_finetune")

    p.add_argument("--eval_only",  action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)

    p.add_argument("--sweep",        action="store_true")
    p.add_argument("--sweep_models", type=str,
                   default="facebook/wav2vec2-large-robust,microsoft/wavlm-large")
    p.add_argument("--sweep_modes",  type=str, default="full,lora")

    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.sweep:
        run_sweep(args)
    elif args.eval_only:
        evaluate_checkpoint(args)
    else:
        train(args)