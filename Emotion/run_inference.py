"""
run_inference.py
================
Runs inference on all checkpoints in a folder, computes:
  - UA, Accuracy, Weighted F1
  - RTF (Real-Time Factor)
  - FLOPs (via thop or manual calculation)
  - Trainable/Total params
  - GPU name and memory
  - Training hours

Saves results to:
  - JSON (for notebook/tensorboard)
  - TensorBoard event files (one scalar per metric per run)
  - CSV summary table

Usage:
  python run_inference.py \
      --checkpoints_dir /scratch/zt1/project/msml605/user/mokshdag/checkpoints/checkpoints/ \
      --audio_dir /scratch/zt1/project/msml604/user/mokshdag/hf_cache/datasets/crema-d/data/data/AudioWAV \
      --output_dir /scratch/zt1/project/msml605/user/mokshdag/inference_results/

  # Single checkpoint
  python run_inference.py \
      --checkpoint /path/to/checkpoint \
      --model_name facebook/wav2vec2-large-robust \
      --mode lora \
      --audio_dir ...
"""

# ── Cache setup ───────────────────────────────────────────────────────────────
import os
BASE = "/scratch/zt1/project/msml604/user/mokshdag/miniconda3/envs/same"
_lib_paths = [
    f"{BASE}/lib/python3.11/site-packages/nvidia/nccl/lib",
    f"{BASE}/lib",
    f"{BASE}/lib/python3.11/site-packages/torch/lib",
    f"{BASE}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib",
    f"{BASE}/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib",
]
existing = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = ":".join(_lib_paths) + (":" + existing if existing else "")

CACHE_DIR  = "/scratch/zt1/project/msml604/user/mokshdag/hf_cache"
VYOM_CACHE = "/scratch/zt1/project/msml604/user/dvyomwal5/anaconda3/envs/asr/hf_cache"

LOCAL_MODEL_PATH = {
    "facebook/wav2vec2-large-robust": f"{CACHE_DIR}/models/wav2vec2-large-robust",
    "facebook/wav2vec2-base":         f"{CACHE_DIR}/models/wav2vec2-base",
    "openai/whisper-small":           f"{VYOM_CACHE}/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d",
}

# ── Imports ───────────────────────────────────────────────────────────────────
import glob
import time
import json
import csv
import argparse
import numpy as np
import torch
import soundfile as sf
import torchaudio
import tqdm
from pathlib import Path
from typing import Optional, Dict, List

from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter
        HAS_TB = True
    except ImportError:
        HAS_TB = False
        print("Warning: TensorBoard not available — JSON/CSV only")

# FLOPs calculation
try:
    from thop import profile as thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("Warning: thop not installed — FLOPs estimated analytically")

# ── Constants ─────────────────────────────────────────────────────────────────
CREMAD_LABEL_MAP = {"NEU": 0, "HAP": 1, "SAD": 2, "ANG": 3, "FEA": 4, "DIS": 5}
LABEL_NAMES      = ["neutral", "happy", "sad", "angry", "fear", "disgust"]
SAMPLING_RATE    = 16_000


# ── GPU Info ──────────────────────────────────────────────────────────────────

def get_gpu_info() -> Dict:
    if not torch.cuda.is_available():
        return {"name": "CPU", "memory_gb": 0, "compute_capability": "N/A"}
    idx  = torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(idx)
    return {
        "name":                prop.name,
        "memory_gb":           round(prop.total_memory / 1e9, 1),
        "compute_capability":  f"{prop.major}.{prop.minor}",
        "multi_processor_count": prop.multi_processor_count,
    }


# ── FLOPs calculation ─────────────────────────────────────────────────────────

def estimate_flops_analytical(model, audio_len_s: float, sampling_rate: int = 16_000) -> float:
    """
    Analytical FLOPs estimate for wav2vec2-style encoder + classifier.
    FLOPs ∝ L * T^2 * d  (transformer self-attention)
    where T = sequence length after CNN downsampling.
    """
    total_params = sum(p.numel() for p in model.parameters())

    # wav2vec2: CNN stride=320 → T ≈ audio_len_s * 50
    T = int(audio_len_s * 50)

    # Rough: 2 * params * T flops for a forward pass
    flops = 2 * total_params * T
    return flops / 1e9  # GFLOPs


def compute_flops(model, feature_extractor, audio_len_s: float,
                  device: torch.device, sampling_rate: int = 16_000) -> float:
    """Compute FLOPs using thop if available, else analytical estimate."""
    if HAS_THOP:
        try:
            # Create dummy input
            dummy_audio = np.zeros(int(audio_len_s * sampling_rate), dtype=np.float32)
            inputs = feature_extractor(
                dummy_audio, sampling_rate=sampling_rate,
                max_length=int(audio_len_s * sampling_rate),
                truncation=True, padding="max_length", return_tensors="pt"
            )
            input_values = inputs.input_values.to(device)

            # Merge LoRA if needed
            eval_model = model
            if hasattr(model, "merge_and_unload"):
                eval_model = model.merge_and_unload()

            macs, _ = thop_profile(eval_model, inputs=(input_values,), verbose=False)
            return round(2 * macs / 1e9, 2)  # FLOPs = 2 * MACs
        except Exception as e:
            print(f"  thop failed ({e}), using analytical estimate")

    return round(estimate_flops_analytical(model, audio_len_s), 2)


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_cremad_test(audio_dir: str, seed: int = 42, max_samples: Optional[int] = None):
    """Load CREMA-D test split (last 10% deterministically)."""
    all_wavs = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    rng      = np.random.default_rng(seed)
    indices  = rng.permutation(len(all_wavs))
    n        = len(all_wavs)
    test_idx = indices[int(0.9 * n):]

    if max_samples:
        test_idx = test_idx[:max_samples]

    samples = []
    for idx in test_idx:
        path    = all_wavs[idx]
        fname   = os.path.basename(path)
        emotion = fname.replace(".wav", "").split("_")[2]
        label   = CREMAD_LABEL_MAP.get(emotion, -1)
        if label == -1:
            continue
        samples.append((path, label))

    print(f"  Test split: {len(samples)} samples")
    return samples


# ── Inference ────────────────────────────────────────────────────────────────

def run_inference(
    model,
    feature_extractor,
    samples: List,
    max_audio_len: float,
    device: torch.device,
) -> Dict:
    """Run inference and collect metrics."""
    model.eval()
    max_len = int(max_audio_len * SAMPLING_RATE)

    all_preds, all_labels = [], []
    total_audio_s = 0.0
    total_infer_s = 0.0
    gpu_mem_peak  = 0.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for path, label in tqdm.tqdm(samples, desc="Inference"):
        audio_array, sr = sf.read(path, dtype="float32")

        if sr != SAMPLING_RATE:
            waveform    = torch.tensor(audio_array).unsqueeze(0)
            resampler   = torchaudio.transforms.Resample(sr, SAMPLING_RATE)
            audio_array = resampler(waveform).squeeze(0).numpy()

        audio_dur_s = len(audio_array) / SAMPLING_RATE
        if len(audio_array) > max_len:
            audio_array = audio_array[:max_len]

        inputs = feature_extractor(
            audio_array, sampling_rate=SAMPLING_RATE,
            max_length=max_len, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        input_values = inputs.input_values.to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(input_values=input_values).logits
        total_infer_s += time.perf_counter() - t0

        all_preds.append(logits.argmax(dim=-1).item())
        all_labels.append(label)
        total_audio_s += audio_dur_s

    if torch.cuda.is_available():
        gpu_mem_peak = torch.cuda.max_memory_allocated() / 1e9

    acc = accuracy_score(all_labels, all_preds)
    ua  = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    wa  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    rtf = total_infer_s / max(total_audio_s, 1e-6)
    cm  = confusion_matrix(all_labels, all_preds).tolist()
    report = classification_report(
        all_labels, all_preds, target_names=LABEL_NAMES, zero_division=0
    )

    return {
        "accuracy":              round(100 * acc, 3),
        "ua":                    round(100 * ua,  3),
        "weighted_f1":           round(100 * wa,  3),
        "rtf":                   round(rtf, 5),
        "total_audio_h":         round(total_audio_s / 3600, 4),
        "total_infer_s":         round(total_infer_s, 3),
        "n_samples":             len(all_preds),
        "gpu_peak_memory_gb":    round(gpu_mem_peak, 3),
        "confusion_matrix":      cm,
        "classification_report": report,
    }


# ── Parse checkpoint metadata ─────────────────────────────────────────────────

def parse_checkpoint_meta(checkpoint_dir: Path) -> Dict:
    """Extract config from run_meta.json and trainer_state.json."""
    meta = {}
    meta_path = checkpoint_dir / "run_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    # Best eval UA from training
    best_train_ua = 0.0
    for ts_path in checkpoint_dir.rglob("trainer_state.json"):
        try:
            with open(ts_path) as f:
                state = json.load(f)
            uas = [x.get("eval_ua", 0) for x in state["log_history"] if "eval_ua" in x]
            if uas:
                best_train_ua = max(best_train_ua, max(uas))
        except:
            pass

    meta["best_train_ua"] = round(best_train_ua * 100, 3)

    # Parse axes from folder name
    name = checkpoint_dir.name
    if "audio" in name:
        try:
            audio_part = name.split("audio")[1].split("s")[0]
            meta["audio_len_parsed"] = float(audio_part)
        except:
            pass
    if "unfreeze" in name:
        try:
            meta["unfreeze_parsed"] = int(name.split("unfreeze")[-1])
        except:
            pass

    return meta


# ── Load model from checkpoint ────────────────────────────────────────────────

def load_model_from_checkpoint(
    checkpoint_dir: Path,
    meta: Dict,
    device: torch.device,
):
    """Load model + feature extractor from checkpoint."""
    model_name_hf = meta.get("model_name", "facebook/wav2vec2-large-robust")
    # Resolve to local path
    model_name = LOCAL_MODEL_PATH.get(model_name_hf.split("/")[-2] + "/" + model_name_hf.split("/")[-1],
                  LOCAL_MODEL_PATH.get(model_name_hf, model_name_hf))
    # Simpler: check key directly
    for k, v in LOCAL_MODEL_PATH.items():
        if k in model_name_hf or model_name_hf in k:
            model_name = v
            break

    mode       = meta.get("mode", "lora")
    num_labels = meta.get("num_labels", 6)
    label_names = meta.get("label_names", LABEL_NAMES)

    # Feature extractor — prefer from checkpoint, fall back to base model
    fe_path = (str(checkpoint_dir)
               if (checkpoint_dir / "preprocessor_config.json").exists()
               else model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(fe_path)

    # Model
    id2label = {i: n for i, n in enumerate(label_names)}
    label2id = {n: i for i, n in enumerate(label_names)}

    if mode == "lora":
        # Find the actual LoRA adapter files
        # They could be in checkpoint_dir directly or in a checkpoint-XXXX subfolder
        adapter_dir = checkpoint_dir
        if not (checkpoint_dir / "adapter_config.json").exists():
            # Look in best checkpoint subfolder
            for sub in sorted(checkpoint_dir.iterdir()):
                if sub.is_dir() and "checkpoint" in sub.name:
                    if (sub / "adapter_config.json").exists():
                        adapter_dir = sub
                        break

        base = AutoModelForAudioClassification.from_pretrained(
            model_name, num_labels=num_labels,
            id2label=id2label, label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        model = PeftModel.from_pretrained(base, str(adapter_dir))
        model = model.merge_and_unload()
    else:
        # Full finetune — load directly from checkpoint
        model_load_path = checkpoint_dir
        if not (checkpoint_dir / "config.json").exists():
            for sub in sorted(checkpoint_dir.iterdir()):
                if sub.is_dir() and "checkpoint" in sub.name:
                    if (sub / "config.json").exists():
                        model_load_path = sub
                        break
        model = AutoModelForAudioClassification.from_pretrained(
            str(model_load_path), num_labels=num_labels,
            id2label=id2label, label2id=label2id,
            ignore_mismatched_sizes=True,
        )

    model = model.to(device).eval()
    return model, feature_extractor, model_name


# ── Save results ──────────────────────────────────────────────────────────────

def save_results(all_results: List[Dict], output_dir: str):
    """Save to JSON, CSV, and TensorBoard."""
    os.makedirs(output_dir, exist_ok=True)

    # ── JSON ──────────────────────────────────────────────────────────────────
    json_path = os.path.join(output_dir, "all_inference_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"JSON saved: {json_path}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "summary_table.csv")
    fieldnames = [
        "run_name", "model", "mode", "audio_len", "lora_r", "unfreeze_layers",
        "adaptation", "flops_b", "trainable_params_m", "total_params_m",
        "ua_pct", "accuracy_pct", "weighted_f1_pct", "rtf",
        "train_hrs", "gpu_name", "gpu_memory_gb", "gpu_peak_memory_gb",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"CSV saved: {csv_path}")

    # ── TensorBoard ───────────────────────────────────────────────────────────
    if HAS_TB:
        tb_dir = os.path.join(output_dir, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)

        for i, r in enumerate(all_results):
            run_name = r.get("run_name", f"run_{i}")
            writer   = SummaryWriter(log_dir=os.path.join(tb_dir, run_name))

            # Log all scalar metrics
            for metric, val in {
                "inference/ua":                  r.get("ua_pct", 0),
                "inference/accuracy":            r.get("accuracy_pct", 0),
                "inference/weighted_f1":         r.get("weighted_f1_pct", 0),
                "inference/rtf":                 r.get("rtf", 0),
                "compute/flops_b":               r.get("flops_b", 0),
                "compute/trainable_params_m":    r.get("trainable_params_m", 0),
                "compute/total_params_m":        r.get("total_params_m", 0),
                "compute/train_hrs":             r.get("train_hrs", 0),
                "compute/gpu_peak_memory_gb":    r.get("gpu_peak_memory_gb", 0),
            }.items():
                writer.add_scalar(metric, float(val or 0), global_step=0)

            writer.close()

        print(f"TensorBoard logs saved: {tb_dir}")
        print(f"  View with: tensorboard --logdir {tb_dir} --port 6006")

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "="*120)
    print(f"{'Model':<30} {'Mode':<6} {'Audio':>5} {'Rank':>5} {'Adapt':<12} "
          f"{'FLOPs(B)':>9} {'Params(M)':>10} {'UA%':>7} {'Acc%':>7} {'RTF':>8} {'GPU':<20}")
    print("="*120)
    for r in sorted(all_results, key=lambda x: -x.get("ua_pct", 0)):
        print(
            f"{str(r.get('model',''))[:30]:<30} "
            f"{r.get('mode',''):<6} "
            f"{r.get('audio_len', '?'):>5} "
            f"{str(r.get('lora_r','—')):>5} "
            f"{r.get('adaptation','uniform'):<12} "
            f"{r.get('flops_b', 0):>9.1f} "
            f"{r.get('trainable_params_m', 0):>10.1f} "
            f"{r.get('ua_pct', 0):>7.2f} "
            f"{r.get('accuracy_pct', 0):>7.2f} "
            f"{r.get('rtf', 0):>8.5f} "
            f"{str(r.get('gpu_name',''))[:20]:<20}"
        )
    print("="*120)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints_dir", type=str, default=None,
                   help="Directory containing multiple checkpoint folders")
    p.add_argument("--checkpoint",      type=str, default=None,
                   help="Single checkpoint folder to evaluate")
    p.add_argument("--model_name",      type=str, default=None,
                   help="Override model name (for single checkpoint mode)")
    p.add_argument("--mode",            type=str, default=None,
                   help="Override mode: lora or full (for single checkpoint mode)")
    p.add_argument("--audio_dir",       type=str,
                   default="/scratch/zt1/project/msml604/user/mokshdag/hf_cache/datasets/crema-d/data/data/AudioWAV")
    p.add_argument("--output_dir",      type=str,
                   default="/scratch/zt1/project/msml605/user/mokshdag/inference_results/")
    p.add_argument("--max_samples",     type=int, default=None,
                   help="Cap test samples for quick testing")
    return p.parse_args()


def process_checkpoint(checkpoint_dir: Path, args, samples, device, gpu_info) -> Optional[Dict]:
    """Process a single checkpoint and return results dict."""
    print(f"\n{'='*70}")
    print(f"  Checkpoint: {checkpoint_dir.name}")
    print(f"{'='*70}")

    meta = parse_checkpoint_meta(checkpoint_dir)

    # Override from args if provided
    if args.model_name:
        meta["model_name"] = args.model_name
    if args.mode:
        meta["mode"] = args.mode

    if not meta.get("model_name"):
        print("  SKIP: no model_name in meta and none provided via --model_name")
        return None

    try:
        model, feature_extractor, model_name_resolved = load_model_from_checkpoint(
            checkpoint_dir, meta, device
        )
    except Exception as e:
        print(f"  SKIP: failed to load model — {e}")
        return None

    # Count params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params     = sum(p.numel() for p in model.parameters())

    # Audio len
    max_audio_len = meta.get("max_audio_len") or meta.get("audio_len_parsed", 4.0)
    if max_audio_len is None:
        max_audio_len = 4.0

    # FLOPs
    flops_b = compute_flops(model, feature_extractor, float(max_audio_len), device)

    # Inference
    infer_results = run_inference(model, feature_extractor, samples, float(max_audio_len), device)

    # Adaptation type
    unfreeze = meta.get("unfreeze_parsed", meta.get("unfreeze_top_layers", 4))
    if meta.get("mode") == "full":
        adaptation = "Full"
    elif unfreeze == 0:
        adaptation = "LoRA-only"
    elif unfreeze == 4:
        adaptation = "LoRA+top4"
    elif unfreeze == 8:
        adaptation = "LoRA+top8"
    else:
        adaptation = f"LoRA+top{unfreeze}"

    result = {
        # Identity
        "run_name":           checkpoint_dir.name,
        "model":              meta.get("model_name", "").split("/")[-1],
        "model_full":         meta.get("model_name", ""),
        "mode":               meta.get("mode", "lora"),
        "adaptation":         adaptation,
        # Axes
        "audio_len":          float(max_audio_len),
        "lora_r":             meta.get("lora_r"),
        "unfreeze_layers":    unfreeze,
        # Compute
        "flops_b":            flops_b,
        "trainable_params_m": round(trainable_params / 1e6, 2),
        "total_params_m":     round(total_params / 1e6, 2),
        "trainable_pct":      round(100 * trainable_params / max(total_params, 1), 3),
        "train_hrs":          meta.get("training_hours", 0),
        # GPU
        "gpu_name":           gpu_info["name"],
        "gpu_memory_gb":      gpu_info["memory_gb"],
        "gpu_peak_memory_gb": infer_results["gpu_peak_memory_gb"],
        # Performance
        "ua_pct":             infer_results["ua"],
        "accuracy_pct":       infer_results["accuracy"],
        "weighted_f1_pct":    infer_results["weighted_f1"],
        "rtf":                infer_results["rtf"],
        "n_samples":          infer_results["n_samples"],
        "total_audio_h":      infer_results["total_audio_h"],
        # Training UA (from trainer_state)
        "best_train_ua":      meta.get("best_train_ua", 0),
        # Per-class report
        "classification_report": infer_results["classification_report"],
        "confusion_matrix":      infer_results["confusion_matrix"],
    }

    print(f"\n  UA={result['ua_pct']:.2f}%  Acc={result['accuracy_pct']:.2f}%  "
          f"RTF={result['rtf']:.5f}  FLOPs={flops_b:.1f}B  "
          f"Params={result['trainable_params_m']:.1f}M  GPU={gpu_info['name']}")

    return result


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_info = get_gpu_info()

    print(f"Device  : {device}")
    print(f"GPU     : {gpu_info['name']} ({gpu_info['memory_gb']}GB)")
    print(f"Audio   : {args.audio_dir}")

    # Load test set once
    print("\nLoading CREMA-D test split...")
    samples = load_cremad_test(args.audio_dir, max_samples=args.max_samples)

    all_results = []

    if args.checkpoints_dir:
        # Process all checkpoints in directory
        checkpoint_dirs = sorted(Path(args.checkpoints_dir).iterdir())
        checkpoint_dirs = [d for d in checkpoint_dirs if d.is_dir()]
        print(f"\nFound {len(checkpoint_dirs)} checkpoint folders")

        for ckpt_dir in checkpoint_dirs:
            result = process_checkpoint(ckpt_dir, args, samples, device, gpu_info)
            if result:
                all_results.append(result)
                # Save incrementally after each run
                save_results(all_results, args.output_dir)

    elif args.checkpoint:
        result = process_checkpoint(Path(args.checkpoint), args, samples, device, gpu_info)
        if result:
            all_results.append(result)
            save_results(all_results, args.output_dir)
    else:
        print("ERROR: provide --checkpoints_dir or --checkpoint")
        return

    print(f"\n\nDone! Processed {len(all_results)} checkpoints.")
    print(f"Results saved to: {args.output_dir}")
    if HAS_TB:
        print(f"TensorBoard: tensorboard --logdir {args.output_dir}/tensorboard --port 6006")


if __name__ == "__main__":
    main()
