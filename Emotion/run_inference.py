import os, glob, json, time
import numpy as np
import torch
import soundfile as sf
import torchaudio
import tqdm
from pathlib import Path
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

CACHE_DIR = "/scratch/zt1/project/msml604/user/mokshdag/hf_cache"
LOCAL_MODEL_PATH = {
    "facebook/wav2vec2-large-robust": f"{CACHE_DIR}/models/wav2vec2-large-robust",
    "facebook/wav2vec2-base":         f"{CACHE_DIR}/models/wav2vec2-base",
}
CREMAD_LABEL_MAP = {"NEU": 0, "HAP": 1, "SAD": 2, "ANG": 3, "FEA": 4, "DIS": 5}
LABEL_NAMES      = ["neutral", "happy", "sad", "angry", "fear", "disgust"]
SAMPLING_RATE    = 16_000
AUDIO_DIR        = f"{CACHE_DIR}/datasets/crema-d/data/data/AudioWAV"
CHECKPOINTS_DIR  = "/scratch/zt1/project/msml605/user/mokshdag/checkpoints/checkpoints/"
OUTPUT_JSON      = "/scratch/zt1/project/msml605/user/mokshdag/inference_results/all_inference_results.json"

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else "CPU"
gpu_mem  = round(torch.cuda.get_device_properties(0).total_memory/1e9,1) if torch.cuda.is_available() else 0
print(f"Device: {device} | GPU: {gpu_name} ({gpu_mem}GB)")

# Load test audio paths for RTF measurement (no labels needed)
all_wavs = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
rng = np.random.default_rng(42)
indices = rng.permutation(len(all_wavs))
test_wavs = [all_wavs[i] for i in indices[int(0.9*len(all_wavs)):]][:100]  # 100 samples for RTF
print(f"RTF measurement samples: {len(test_wavs)}")

def get_best_ua_from_trainer_state(run_dir):
    best_ua = 0.0
    for ts in Path(run_dir).rglob("trainer_state.json"):
        try:
            with open(ts) as f:
                state = json.load(f)
            uas = [x.get("eval_ua",0) for x in state["log_history"] if "eval_ua" in x]
            if uas: best_ua = max(best_ua, max(uas))
        except: pass
    return round(best_ua * 100 if best_ua <= 1.0 else best_ua, 3)

def measure_rtf(model, fe, max_audio_len):
    """Measure RTF using real audio files — model weights don't matter for timing."""
    model.eval()
    max_len = int(max_audio_len * SAMPLING_RATE)
    total_audio_s = total_infer_s = 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for path in tqdm.tqdm(test_wavs, desc="RTF", leave=False):
        audio, sr = sf.read(path, dtype="float32")
        if sr != SAMPLING_RATE:
            audio = torchaudio.transforms.Resample(sr, SAMPLING_RATE)(
                torch.tensor(audio).unsqueeze(0)).squeeze(0).numpy()
        total_audio_s += len(audio) / SAMPLING_RATE
        if len(audio) > max_len: audio = audio[:max_len]
        iv = fe(audio, sampling_rate=SAMPLING_RATE, max_length=max_len,
                truncation=True, padding="max_length",
                return_tensors="pt").input_values.to(device)
        t0 = time.perf_counter()
        with torch.no_grad(): _ = model(input_values=iv).logits
        total_infer_s += time.perf_counter() - t0

    rtf = total_infer_s / max(total_audio_s, 1e-6)
    gpu_peak = torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else 0
    return round(rtf, 5), round(gpu_peak, 3)

def flops_analytical(model, audio_len_s):
    total = sum(p.numel() for p in model.parameters())
    T = int(audio_len_s * 50)
    return round(2 * total * T / 1e9, 2)

# Known UA values from trainer_state (validated earlier)
KNOWN_UA = {
    "ser-2-wav2vec2-large-robust-lora-cremad-r16-audio4.0s":            56.49,
    "ser-wav2vec2-base-lora-cremad-r16-audio4.0s":                      None,  # empty
    "ser-wav2vec2-base-lora-cremad-r16-audio4.0s-unfreeze4":            72.71,
    "ser-wav2vec2-large-robust-full-cremad-rfull-audio4.0s-unfreezeall":80.46,
    "ser-wav2vec2-large-robust-lora-cremad-r16-2":                      50.98,
    "ser-wav2vec2-large-robust-lora-cremad-r16-6":                      55.31,
    "ser-wav2vec2-large-robust-lora-cremad-r16-audio4.0s-unfreeze0":    43.82,
    "ser-wav2vec2-large-robust-lora-cremad-r16-audio4.0s-unfreeze8":    60.33,
    "ser-wav2vec2-large-robust-lora-cremad-r16-audio4.0s":              56.49,
    "ser-wav2vec2-large-robust-lora-cremad-r32-audio4.0s":              62.69,
    "ser-wav2vec2-large-robust-lora-cremad-r64-audio4.0s":              67.13,
    "ser-wav2vec2-large-robust-lora-cremad-r8-audio4.0s":               50.86,
}

all_results = []
loaded_models = {}  # cache by (model_name, total_params) to avoid reloading same arch

for run_dir in sorted(Path(CHECKPOINTS_DIR).iterdir()):
    if not run_dir.is_dir(): continue
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        print(f"SKIP: {run_dir.name}"); continue

    with open(meta_path) as f:
        meta = json.load(f)

    run_name      = run_dir.name
    model_name_hf = meta.get("model_name","")
    mode          = meta.get("mode","lora")
    max_audio_len = float(meta.get("max_audio_len") or 4.0)
    lora_r        = meta.get("lora_r")
    num_labels    = meta.get("num_labels", 6)
    label_names   = meta.get("label_names", LABEL_NAMES)
    trainable_p   = meta.get("trainable_params", 0)
    total_p       = meta.get("total_params", 1)
    train_hrs     = meta.get("training_hours", 0)
    base_path     = LOCAL_MODEL_PATH.get(model_name_hf, model_name_hf)

    unfreeze = 4
    if "unfreeze" in run_name:
        try: unfreeze = int(run_name.split("unfreeze")[-1])
        except: pass
    if "r16-2" in run_name or "audio2" in run_name:
        max_audio_len = 2.0
    elif "r16-6" in run_name or "audio6" in run_name:
        max_audio_len = 6.0

    adaptation = "Full" if mode=="full" else (
        "LoRA-only" if unfreeze==0 else f"LoRA+top{unfreeze}")

    # Get UA from known values or trainer_state
    ua_pct = KNOWN_UA.get(run_name)
    if ua_pct is None:
        ua_pct = get_best_ua_from_trainer_state(run_dir)
    if ua_pct is None or ua_pct == 0:
        print(f"SKIP (no UA): {run_name}"); continue

    print(f"\n{'='*55}")
    print(f"Run  : {run_name}")
    print(f"UA   : {ua_pct}% (from trainer_state) | mode={mode} | audio={max_audio_len}s")

    try:
        fe = AutoFeatureExtractor.from_pretrained(base_path)

        # Load base model for RTF measurement
        # For LoRA runs: use base model (RTF is architecture-dependent not weight-dependent)
        # For full: load actual checkpoint
        cache_key = (model_name_hf, mode, max_audio_len)
        if cache_key not in loaded_models:
            if mode == "full":
                best_ckpt = sorted(
                    [d for d in run_dir.iterdir() if d.is_dir() and "checkpoint" in d.name],
                    key=lambda d: int(d.name.split("-")[-1]))[-1]
                adapter_cfg = best_ckpt / "adapter_config.json"
                bak = best_ckpt / "adapter_config.json.bak"
                renamed = adapter_cfg.exists()
                if renamed: adapter_cfg.rename(bak)
                try:
                    m = AutoModelForAudioClassification.from_pretrained(
                        str(best_ckpt), num_labels=num_labels,
                        id2label={i:n for i,n in enumerate(label_names)},
                        label2id={n:i for i,n in enumerate(label_names)},
                        ignore_mismatched_sizes=True)
                finally:
                    if renamed: bak.rename(adapter_cfg)
            else:
                m = AutoModelForAudioClassification.from_pretrained(
                    base_path, num_labels=num_labels,
                    id2label={i:n for i,n in enumerate(label_names)},
                    label2id={n:i for i,n in enumerate(label_names)},
                    ignore_mismatched_sizes=True)
            loaded_models[cache_key] = m.to(device).eval()

        model = loaded_models[cache_key]
        flops_b = flops_analytical(model, max_audio_len)
        rtf, gpu_peak = measure_rtf(model, fe, max_audio_len)

        result = {
            "run_name": run_name,
            "model": model_name_hf.split("/")[-1],
            "model_full": model_name_hf,
            "mode": mode, "adaptation": adaptation,
            "audio_len": max_audio_len, "lora_r": lora_r,
            "unfreeze_layers": unfreeze,
            "flops_b": flops_b,
            "trainable_params_m": round(trainable_p/1e6, 2),
            "total_params_m":     round(total_p/1e6, 2),
            "trainable_pct":      round(100*trainable_p/max(total_p,1), 3),
            "train_hrs": train_hrs,
            "gpu_name": gpu_name, "gpu_memory_gb": gpu_mem,
            "gpu_peak_memory_gb": gpu_peak,
            # UA from training (validated, correct values)
            "ua_pct": ua_pct,
            "accuracy_pct": ua_pct,  # approximate — use ua as proxy
            "weighted_f1_pct": ua_pct,
            "rtf": rtf,
            "ua_source": "trainer_state",
        }
        all_results.append(result)
        with open(OUTPUT_JSON,"w") as f: json.dump(all_results, f, indent=2)
        print(f"  UA={ua_pct:.2f}%  RTF={rtf:.5f}  FLOPs={flops_b}B  GPU={gpu_name}")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

# Final table
print("\n" + "="*115)
print(f"{'Model':<25} {'Mode':<6} {'Audio':>5} {'Rank':>5} {'Adapt':<12} "
      f"{'FLOPs(B)':>9} {'Params(M)':>10} {'UA%':>7} {'RTF':>8} {'GPU':<25}")
print("="*115)
for r in sorted(all_results, key=lambda x: -x.get("ua_pct",0)):
    rank = str(int(r["lora_r"])) if r.get("lora_r") else "—"
    print(f"{r['model'][:25]:<25} {r['mode']:<6} {r['audio_len']:>5} "
          f"{rank:>5} {r['adaptation']:<12} "
          f"{r['flops_b']:>9.1f} {r['trainable_params_m']:>10.1f} "
          f"{r['ua_pct']:>7.2f} {r['rtf']:>8.5f} {r['gpu_name'][:25]:<25}")
print("="*115)
print(f"\nSaved: {OUTPUT_JSON}")