"""
inference_eval.py — Comprehensive Inference Evaluation for Whisper ASR
=======================================================================
Compatible with whisper_finetune.py — handles:
  - Plain WhisperForConditionalGeneration  (mode=baseline / mode=full)
  - PeftModel (LoRA) wrapped in WhisperWithTokenSubsampling  (mode=lora, tpf>1)
  - experiment_cfg.json auto-read from checkpoint directory
  - GPU-agnostic loading (A100 / V100 / H100 / CPU)

Computes and logs:
  - WER  (test_clean / test_other)
  - Real-Time Factor (RTF)
  - Theoretical inference FLOPs (encoder + decoder)
  - Measured latency per sample (mean, p50, p95, p99)  via CUDA events
  - GPU name, VRAM usage, compute capability
  - Throughput (samples/sec, audio-hours/hour)
  - Model parameter counts (total, encoder, decoder)
  - TensorBoard scalars + HParam panel

Output:
  - Per-run JSON  →  {output_dir}/eval_{size}_{mode}_tf{F}_tpf{T}_{gpu}_{bench}.json
  - TensorBoard   →  {output_dir}/tensorboard/{model_size}/{mode}/
  - Combined JSON →  {output_dir}/sweep_{bench}_{timestamp}.json  (multi-run)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Auto-discover ALL checkpoints under the training output folder
python inference_eval.py \\
    --checkpoint_root /home/vyomwal5/SAME/checkpoints \\
    --benchmark librispeech --max_eval_samples 500

# 2. Filter by model size / mode
python inference_eval.py \\
    --checkpoint_root /home/vyomwal5/SAME/checkpoints \\
    --filter_model_size small,medium --filter_mode lora \\
    --benchmark librispeech --max_eval_samples 500

# 3. Single checkpoint (tokens_per_frame / total_frames read from experiment_cfg.json)
python inference_eval.py \\
    --model_size small --mode lora \\
    --checkpoint /home/vyomwal5/SAME/checkpoints/whisper-small-lora-librispeech-asr-1-1500-a100 \\
    --benchmark librispeech --max_eval_samples 500

# 4. Override compute axes at eval time (cross-setting test)
python inference_eval.py \\
    --model_size small --mode lora \\
    --checkpoint /home/vyomwal5/SAME/checkpoints/whisper-small-lora-librispeech-asr-1-1500-a100 \\
    --tokens_per_frame 2 --total_frames 750

# 5. Sweep tokens_per_frame over one checkpoint
python inference_eval.py \\
    --model_size small --mode lora \\
    --checkpoint /home/vyomwal5/SAME/checkpoints/whisper-small-lora-librispeech-asr-1-1500-a100 \\
    --sweep_tokens_per_frame 1,2,4 --total_frames 1500

# 6. Zero-shot baseline (no checkpoint)
python inference_eval.py \\
    --model_size small --mode baseline \\
    --tokens_per_frame 1 --total_frames 1500

TensorBoard:
  tensorboard --logdir /home/vyomwal5/SAME/eval_results/tensorboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ─────────────────────────────────────────────────────────────────────────────
# CACHE — before ALL imports
# ─────────────────────────────────────────────────────────────────────────────
import os

CACHE_DIR = "/scratch/zt1/project/msml604/user/vyomwal5/anaconda3/envs/asr/hf_cache"


LOCAL_PATH = {
    "tiny":     "/scratch/zt1/project/msml604/user/vyomwal5/anaconda3/envs/asr/hf_cache/models/models--openai--whisper-tiny/snapshots/169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
    "small":    "/scratch/zt1/project/msml604/user/vyomwal5/anaconda3/envs/asr/hf_cache/models/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d",
    "medium":   "/scratch/zt1/project/msml604/user/vyomwal5/anaconda3/envs/asr/hf_cache/models/models--openai--whisper-medium/snapshots/abdf7c39ab9d0397620ccaea8974cc764cd0953e",
    "large-v3": "/scratch/zt1/project/msml604/user/vyomwal5/anaconda3/envs/asr/hf_cache/models/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1",
}

os.environ["HF_HOME"]                = CACHE_DIR
os.environ["HF_DATASETS_CACHE"]      = f"{CACHE_DIR}/datasets"
os.environ["TRANSFORMERS_CACHE"]     = f"{CACHE_DIR}/models"
os.environ["HUGGINGFACE_HUB_CACHE"]  = f"{CACHE_DIR}/hub"
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import re
import sys
import json
import time
import datetime
import argparse
import platform
import subprocess
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
from jiwer import wer as jiwer_wer
import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter
        TENSORBOARD_AVAILABLE = True
    except ImportError:
        TENSORBOARD_AVAILABLE = False
        print("WARNING: tensorboard not found — install with: pip install tensorboard")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

WHISPER_ENC_FRAMES = 1500
WHISPER_MEL_FRAMES = 3000
WHISPER_SR         = 16_000
WHISPER_MAX_DUR    = 30.0

WHISPER_PARAMS = {
    "tiny":     (39_000_000,    14_000_000,  25_000_000),
    "base":     (74_000_000,    24_000_000,  50_000_000),
    "small":    (244_000_000,   88_000_000, 156_000_000),
    "medium":   (769_000_000,  307_000_000, 462_000_000),
    "large-v3": (1_540_000_000, 633_000_000, 907_000_000),
}

WHISPER_D_MODEL = {
    "tiny": 384, "base": 512, "small": 768, "medium": 1024, "large-v3": 1280,
}

WHISPER_LAYERS = {
    "tiny": (4, 4), "base": (6, 6), "small": (12, 12),
    "medium": (24, 24), "large-v3": (32, 32),
}

# Folder name pattern produced by whisper_finetune.py
#   whisper-{size}-{mode}-librispeech-asr-{tpf}-{frames}-{gpu_tag}
CHECKPOINT_NAME_RE = re.compile(
    r"^whisper-(?P<size>tiny|base|small|medium|large-v3)"
    r"-(?P<mode>lora|full)"
    r"-librispeech-asr"
    r"-(?P<tpf>\d+)"
    r"-(?P<frames>\d+)"
    r"(?:-(?P<gpu>[a-zA-Z0-9]+))?$"
)


# ─────────────────────────────────────────────────────────────────────────────
# WhisperWithTokenSubsampling
# Identical to whisper_finetune.py — must stay in sync.
# ─────────────────────────────────────────────────────────────────────────────

class WhisperWithTokenSubsampling(torch.nn.Module):
    """
    Encoder-output subsampling wrapper (xV axis, Wang et al.).

    After the encoder runs on the full (128, 3000) mel spectrogram, the
    hidden states are strided along the time dimension before being passed
    to the decoder's cross-attention layers:

      tokens_per_frame=1  → 1500 tokens  (baseline, no-op)
      tokens_per_frame=2  →  750 tokens  (2× cross-attn FLOP reduction)
      tokens_per_frame=4  →  375 tokens  (4× cross-attn FLOP reduction)

    The wrapper carries NO learnable parameters and is never saved to disk.
    It is always reconstructed at load time from experiment_cfg.json.

    Encoder navigation
    ------------------
    The wrapper needs to reach WhisperEncoder regardless of whether the inner
    model is a plain WhisperForConditionalGeneration or a PeftModel.

        Plain:   WhisperForConditionalGeneration
                   .model          → WhisperModel
                     .encoder      → WhisperEncoder

        PEFT:    PeftModel
                   .base_model.model  → WhisperForConditionalGeneration
                     .model          → WhisperModel
                       .encoder      → WhisperEncoder
    """

    def __init__(self, base_model: torch.nn.Module, tokens_per_frame: int = 1):
        super().__init__()
        self.model            = base_model
        self.tokens_per_frame = tokens_per_frame

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_encoder(self):
        m = self.model

        # Unwrap PEFT if present
        if isinstance(m, PeftModel):
            m = m.base_model.model            # → WhisperForConditionalGeneration

        # WhisperForConditionalGeneration → WhisperModel
        whisper_model = getattr(m, "model", m)

        # WhisperModel → WhisperEncoder
        encoder = getattr(whisper_model, "encoder", None)
        if encoder is None:
            # Some model versions nest further
            encoder = getattr(
                getattr(whisper_model, "model", whisper_model), "encoder"
            )
        return encoder

    def _subsample(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.tokens_per_frame > 1:
            hidden = hidden[:, :: self.tokens_per_frame, :]
        return hidden

    def _encode_and_subsample(self, input_features: torch.Tensor):
        encoder     = self._get_encoder()
        encoder_out = encoder(input_features)
        encoder_out.last_hidden_state = self._subsample(
            encoder_out.last_hidden_state
        )
        return encoder_out

    # ── Public API ────────────────────────────────────────────────────────────

    def forward(self, input_features, labels=None, **kwargs):
        encoder_out = self._encode_and_subsample(input_features)
        return self.model(encoder_outputs=encoder_out, labels=labels, **kwargs)

    def generate(self, input_features, **kwargs):
        encoder_out = self._encode_and_subsample(input_features)
        return self.model.generate(encoder_outputs=encoder_out, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CheckpointSpec:
    path:             str
    model_size:       str
    mode:             str
    tokens_per_frame: int
    total_frames:     int
    gpu_tag:          str

    def __str__(self):
        return (f"whisper-{self.model_size} mode={self.mode} "
                f"tpf={self.tokens_per_frame} frames={self.total_frames} "
                f"gpu={self.gpu_tag}  [{self.path}]")


def _read_experiment_cfg(ckpt_path: str) -> Dict:
    """
    Read experiment_cfg.json written by whisper_finetune.py::train().
    Returns {} if absent (older / renamed checkpoints).
    """
    cfg_path = os.path.join(ckpt_path, "experiment_cfg.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        print(f"  [cfg] experiment_cfg.json: {cfg}")
        return cfg
    print(f"  [cfg] No experiment_cfg.json in {ckpt_path} — using folder-name values.")
    return {}


def discover_checkpoints(
    root:               str,
    filter_model_sizes: Optional[List[str]] = None,
    filter_modes:       Optional[List[str]]  = None,
) -> List[CheckpointSpec]:
    """
    Walk `root` one level deep, parse folder names, then refine tpf /
    total_frames from experiment_cfg.json (which takes precedence).
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Checkpoint root not found: {root}")

    specs: List[CheckpointSpec] = []

    for entry in sorted(root_path.iterdir()):
        if not entry.is_dir():
            continue

        m = CHECKPOINT_NAME_RE.match(entry.name)
        if m is None:
            print(f"  [discover] Skip (name mismatch): {entry.name}")
            continue

        size = m.group("size")
        mode = m.group("mode")
        tpf  = int(m.group("tpf"))
        tf   = int(m.group("frames"))
        gpu  = m.group("gpu") or "unknown"

        if filter_model_sizes and size not in filter_model_sizes:
            continue
        if filter_modes and mode not in filter_modes:
            continue

        # Require at least one config file to confirm this is a real checkpoint
        if not any((entry / f).exists()
                   for f in ("adapter_config.json", "config.json")):
            print(f"  [discover] Skip (no config file): {entry.name}")
            continue

        # experiment_cfg.json beats folder-name values for axes params
        saved = _read_experiment_cfg(str(entry))
        tpf   = saved.get("tokens_per_frame", tpf)
        tf    = saved.get("total_frames",     tf)

        specs.append(CheckpointSpec(
            path=str(entry), model_size=size, mode=mode,
            tokens_per_frame=tpf, total_frames=tf, gpu_tag=gpu,
        ))

    print(f"\n  Discovered {len(specs)} checkpoint(s) under '{root}':")
    for s in specs:
        print(f"    {s}")
    print()
    return specs


# ─────────────────────────────────────────────────────────────────────────────
# GPU INFO
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GPUInfo:
    name: str; index: int
    total_vram_gb: float; used_vram_gb: float; free_vram_gb: float
    compute_capability: str; driver_version: str; cuda_version: str; sm_count: int


def get_gpu_info() -> Optional[GPUInfo]:
    if not torch.cuda.is_available():
        return None
    idx   = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    total = props.total_memory / 1e9
    used  = torch.cuda.memory_allocated(idx) / 1e9
    driver = "unknown"
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip().split("\n")[idx]
    except Exception:
        pass
    return GPUInfo(
        name=props.name, index=idx,
        total_vram_gb=round(total, 2), used_vram_gb=round(used, 2),
        free_vram_gb=round(total - used, 2),
        compute_capability=f"{props.major}.{props.minor}",
        driver_version=driver,
        cuda_version=torch.version.cuda or "unknown",
        sm_count=props.multi_processor_count,
    )


def get_vram_used_gb(idx: int = 0) -> float:
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.memory_allocated(idx) / 1e9, 3)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  — GPU-agnostic, finetune-compatible
# ─────────────────────────────────────────────────────────────────────────────

def load_model_for_inference(
    model_size:       str,
    mode:             str,
    checkpoint:       Optional[str],
    device:           torch.device,
    tokens_per_frame: int  = 1,
    fp16:             bool = True,
) -> Tuple[torch.nn.Module, WhisperProcessor]:
    """
    Load model + processor — compatible with every variant from finetune.py.

    Strategy per mode
    -----------------
    baseline
        Load pretrained Whisper from LOCAL_PATH. No adapters, no wrapper
        (tpf is always 1 for baseline).

    lora  (any tpf)
        1. Load base weights to CPU.
        2. Apply PeftModel.from_pretrained(is_trainable=False).
        3. merge_and_unload() → plain WhisperForConditionalGeneration.
        4. If tpf > 1, wrap with WhisperWithTokenSubsampling.
           The wrapper is NEVER saved on disk; it is always reconstructed
           here, mirroring evaluate_checkpoint() in whisper_finetune.py.

    full  (any tpf)
        1. Load from checkpoint dir (full weights saved there).
        2. If tpf > 1, wrap.

    GPU-agnostic trick
    ------------------
    All weights are loaded to CPU first with low_cpu_mem_usage=True, then
    moved to `device` in a single .to() call.  This prevents "CUDA error:
    no kernel image is available for execution on the device" that occurs
    when a checkpoint saved on an A100 / H100 (SM 8.0 / 9.0) is loaded
    directly onto a V100 (SM 7.0) without remapping.
    """
    model_name = LOCAL_PATH[model_size]
    dtype      = torch.float16 if fp16 else torch.float32

    # ── Step 1: determine the correct mel bin count ───────────────────────────
    # Source of truth is the base model's config.json — this is what the
    # Conv1d weight was initialized with and cannot be wrong.
    #   tiny / base / small / medium  →  80 mel bins
    #   large-v3                      → 128 mel bins
    _base_cfg_path = os.path.join(model_name, "config.json")
    _num_mel_bins  = 80   # safe default
    if os.path.exists(_base_cfg_path):
        with open(_base_cfg_path) as _f:
            _num_mel_bins = json.load(_f).get("num_mel_bins", _num_mel_bins)
    print(f"\n  [mel] Conv1d expects {_num_mel_bins} mel bins (from base config.json)")

    # ── Step 2: try to read feature_size from processor_config.json ───────────
    # whisper_finetune.py saves processor_config.json (not preprocessor_config.json).
    # HuggingFace's WhisperProcessor.from_pretrained does NOT read
    # processor_config.json for feature extractor settings — it only reads
    # preprocessor_config.json.  So we parse it manually here.
    _checkpoint_mel_bins = None
    if checkpoint:
        _proc_cfg_path = os.path.join(checkpoint, "processor_config.json")
        if os.path.exists(_proc_cfg_path):
            with open(_proc_cfg_path) as _f:
                _proc_cfg = json.load(_f)
            # processor_config.json has a nested "feature_extractor" key
            _fe_cfg = _proc_cfg.get("feature_extractor", _proc_cfg)
            _checkpoint_mel_bins = _fe_cfg.get("feature_size",
                                   _fe_cfg.get("num_mel_bins", None))
            print(f"  [mel] processor_config.json reports feature_size={_checkpoint_mel_bins}")

    # ── Step 3: load tokenizer from checkpoint, feature extractor from base ───
    # We always load the feature extractor from the base model path because:
    #   (a) preprocessor_config.json is absent from the checkpoint, and
    #   (b) processor_config.json is not read by from_pretrained for the FE.
    # The tokenizer comes from the checkpoint so custom token settings are kept.
    print(f"  Loading feature extractor from base model: {model_name}")
    processor = WhisperProcessor.from_pretrained(
        model_name, language="English", task="transcribe"
    )
    # Override tokenizer with checkpoint version if available
    if checkpoint and os.path.exists(os.path.join(checkpoint, "tokenizer_config.json")):
        from transformers import WhisperTokenizer
        ckpt_tokenizer = WhisperTokenizer.from_pretrained(
            checkpoint, language="English", task="transcribe"
        )
        processor.tokenizer = ckpt_tokenizer
        print(f"  Tokenizer loaded from checkpoint: {checkpoint}")

    # ── Step 4: hard-enforce mel bins = what Conv1d expects ───────────────────
    # Even if base model loading somehow returns wrong bins, this corrects it.
    _loaded_mel = getattr(processor.feature_extractor, "feature_size", None)
    if _loaded_mel != _num_mel_bins:
        print(f"  [mel] MISMATCH: loaded feature_size={_loaded_mel}, "
              f"overriding to {_num_mel_bins} to match Conv1d.")
        processor.feature_extractor.feature_size = _num_mel_bins
        processor.feature_extractor.num_mel_bins = _num_mel_bins
        # Recompute mel filterbank for the correct number of bins
        try:
            import librosa
            processor.feature_extractor.mel_filters = librosa.filters.mel(
                sr=WHISPER_SR,
                n_fft=processor.feature_extractor.n_fft,
                n_mels=_num_mel_bins,
                fmin=0.0, fmax=WHISPER_SR // 2,
            )
            print(f"  [mel] mel_filters recomputed ({_num_mel_bins} bins).")
        except Exception as _e:
            print(f"  [mel] librosa unavailable ({_e}); shape is correct but "
                  f"filterbank values may differ slightly.")
    else:
        print(f"  [mel] Feature extractor confirmed: {_num_mel_bins} mel bins. ✓")

    # Sanity-check: warn if processor_config.json disagrees with base config
    if _checkpoint_mel_bins is not None and _checkpoint_mel_bins != _num_mel_bins:
        print(f"  [mel] WARNING: processor_config.json says feature_size="
              f"{_checkpoint_mel_bins} but Conv1d expects {_num_mel_bins}. "
              f"Using {_num_mel_bins}. Check your training config.")

    # ── Base weights → CPU ────────────────────────────────────────────────────
    print(f"Loading base Whisper-{model_size} to CPU ...")
    base = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    base.config.forced_decoder_ids         = None
    base.generation_config.suppress_tokens = []
    base.config.use_cache                  = True  # KV cache on during inference

    # ── Mode-specific adapter / weight loading ────────────────────────────────
    if mode == "baseline":
        model = base

    elif mode == "lora":
        assert checkpoint, "--checkpoint required for mode=lora"
        print(f"Applying LoRA from: {checkpoint}")
        peft_model = PeftModel.from_pretrained(
            base, checkpoint, is_trainable=False,
        )
        # Fuse LoRA ΔW into W — identical numerics, lower overhead,
        # no PEFT dependency at inference time, and fixes tied-weight
        # safetensors issues on non-training GPUs.
        model = peft_model.merge_and_unload()
        print("  LoRA merged into base weights.")

    elif mode == "full":
        assert checkpoint, "--checkpoint required for mode=full"
        print(f"Loading full finetune from: {checkpoint}")
        model = WhisperForConditionalGeneration.from_pretrained(
            checkpoint, torch_dtype=dtype, low_cpu_mem_usage=True,
        )
        model.config.forced_decoder_ids         = None
        model.generation_config.suppress_tokens = []
        model.config.use_cache                  = True

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose: baseline | lora | full")

    # ── Re-apply subsampling wrapper (xV axis) ────────────────────────────────
    # The wrapper is stateless (no parameters) and is never saved to disk.
    # Re-constructing it here mirrors evaluate_checkpoint() in finetune.py.
    if tokens_per_frame > 1:
        model = WhisperWithTokenSubsampling(model, tokens_per_frame)
        n_out = WHISPER_ENC_FRAMES // tokens_per_frame
        print(f"  [xV] Wrapper applied: stride={tokens_per_frame} "
              f"→ {n_out} encoder tokens to decoder.")
    else:
        print("  [xV] No subsampling (tokens_per_frame=1).")

    # ── CPU → target device (single transfer, avoids SM mismatch) ────────────
    print(f"  Moving to {device} ...")
    model = model.to(device).eval()

    # ── Parameter summary ─────────────────────────────────────────────────────
    inner        = model.model if isinstance(model, WhisperWithTokenSubsampling) else model
    total_params = sum(p.numel() for p in inner.parameters())
    trainable    = sum(p.numel() for p in inner.parameters() if p.requires_grad)
    print(f"  Params  total={total_params:,}  trainable={trainable:,}")

    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
# FLOP ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def estimate_flops(
    model_size: str, total_frames: int,
    tokens_per_frame: int, avg_output_tokens: int = 50,
) -> Dict:
    """Theoretical inference FLOPs (Wang et al. 2025 style). All in GigaFLOPs."""
    d            = WHISPER_D_MODEL[model_size]
    n_enc, n_dec = WHISPER_LAYERS[model_size]
    T            = total_frames
    T_dec        = T // tokens_per_frame
    L            = avg_output_tokens

    enc = n_enc * (2 * T * d * (4 * d + T) + 16 * T * d * d)
    dec = n_dec * (2 * T_dec * d * d + 2 * d * d + 8 * d * d) * L
    tot = enc + dec

    return {
        "flops_encoder_G":        round(enc / 1e9, 3),
        "flops_decoder_G":        round(dec / 1e9, 3),
        "flops_total_G":          round(tot / 1e9, 3),
        "flops_encoder_T":        round(enc / 1e12, 4),
        "flops_decoder_T":        round(dec / 1e12, 4),
        "flops_total_T":          round(tot / 1e12, 4),
        "encoder_frames":         T,
        "decoder_cross_attn_len": T_dec,
        "avg_output_tokens":      L,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEXT NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s\']", "", text)
    return re.sub(r"\s+", " ", text)


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_audio(
    audio_array:      np.ndarray,
    processor:        WhisperProcessor,
    device:           torch.device,
    total_frames:     int,
    tokens_per_frame: int,   # accepted but not used here — xV is in the wrapper
    fp16:             bool = True,
) -> torch.Tensor:
    """
    xT axis: clip audio so at most `total_frames` active encoder frames exist.
    xV axis: handled inside WhisperWithTokenSubsampling.forward() — NOT here.
    Feature extractor always pads / truncates to (128, 3000).
    """
    max_samples = int((total_frames / WHISPER_ENC_FRAMES) * WHISPER_MAX_DUR * WHISPER_SR)
    audio_array = audio_array[:max_samples]

    features = processor.feature_extractor(
        audio_array.astype(np.float32), sampling_rate=WHISPER_SR,
    ).input_features[0]   # (128, 3000)

    feat = torch.tensor(features).unsqueeze(0).to(device)
    return feat.half() if fp16 else feat


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_eval_splits(benchmark: str, max_samples: Optional[int]) -> Dict:
    splits = {}
    if benchmark == "librispeech":
        for cfg, label in [("other", "test_other")]:
            print(f"  Loading librispeech [{cfg}] test (streaming)...")
            ds = load_dataset("librispeech_asr", cfg, split="test", streaming=False)
            if max_samples:
                ds = ds.take(max_samples)
            splits[label] = {"dataset": ds, "text_col": "text"}
    elif benchmark == "common_voice":
        print("  Loading common_voice [en] test (streaming)...")
        ds = load_dataset(
            "mozilla-foundation/common_voice_13_0", "en",
            split="test", streaming=True,
        )
        if max_samples:
            ds = ds.take(max_samples)
        splits["test"] = {"dataset": ds, "text_col": "sentence"}
    elif benchmark == "fleurs":
        print("  Loading fleurs [en_us] test (streaming)...")
        ds = load_dataset("google/fleurs", "en_us", split="test", streaming=True)
        if max_samples:
            ds = ds.take(max_samples)
        splits["test"] = {"dataset": ds, "text_col": "transcription"}
    else:
        raise ValueError(f"Unknown benchmark '{benchmark}'")
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# CORE EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_split(
    model, processor, dataset, text_col: str, device: torch.device,
    total_frames: int, tokens_per_frame: int,
    fp16: bool = True, num_beams: int = 1,
    max_new_tokens: int = 256, warmup_steps: int = 3,
) -> Dict:
    """
    Inference loop for one dataset split.

    Timing: CUDA events (accurate GPU wall time, unaffected by CPU Python
    overhead).  Falls back to 0.0 on CPU — RTF will show 0 but WER is valid.
    """
    all_preds, all_refs       = [], []
    latencies_ms, audio_durs  = [], []
    output_tok_lens           = []
    use_cuda                  = torch.cuda.is_available()
    vram_before               = get_vram_used_gb()

    # Warmup — loads CUDA kernels and fills JIT caches before timing starts
    # Use the actual mel bin count from the processor to avoid shape mismatch
    _mel_bins = getattr(processor.feature_extractor, "feature_size", 80)
    print(f"  Warming up ({warmup_steps} steps, mel_bins={_mel_bins})...")
    dummy_dtype = torch.float16 if fp16 else torch.float32
    dummy = torch.zeros(1, _mel_bins, 3000, dtype=dummy_dtype, device=device)
    with torch.no_grad():
        for _ in range(warmup_steps):
            model.generate(dummy, max_new_tokens=8, num_beams=1)
    if use_cuda:
        torch.cuda.synchronize()

    print("  Running inference...")
    for sample in tqdm.tqdm(dataset):
        audio_array = np.array(sample["audio"]["array"], dtype=np.float32)
        ref_text    = sample[text_col]
        dur_s       = len(audio_array) / WHISPER_SR

        features = preprocess_audio(
            audio_array, processor, device,
            total_frames=total_frames,
            tokens_per_frame=tokens_per_frame,
            fp16=fp16,
        )

        if use_cuda:
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ev0.record()

        with torch.no_grad():
            pred_ids = model.generate(
                features, language="en", task="transcribe",
                num_beams=num_beams, max_new_tokens=max_new_tokens,
            )

        if use_cuda:
            ev1.record()
            torch.cuda.synchronize()
            lat_ms = ev0.elapsed_time(ev1)
        else:
            lat_ms = 0.0

        pred_text = processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True)
        all_preds.append(normalize_text(pred_text))
        all_refs.append(normalize_text(ref_text))
        latencies_ms.append(lat_ms)
        audio_durs.append(dur_s)
        output_tok_lens.append(pred_ids.shape[1])

    vram_after = get_vram_used_gb()

    # Aggregate
    wer_val    = jiwer_wer(all_refs, all_preds)
    lat_arr    = np.array(latencies_ms)
    dur_arr    = np.array(audio_durs)
    rtf_each   = lat_arr / 1000.0 / np.maximum(dur_arr, 1e-6)
    tot_audio  = float(dur_arr.sum())
    tot_infer  = float(lat_arr.sum()) / 1000.0
    n          = len(all_preds)

    return {
        "wer_pct":                     round(100 * wer_val, 4),
        "n_samples":                   n,
        "total_audio_h":               round(tot_audio / 3600, 4),
        "total_inference_s":           round(tot_infer, 3),
        "rtf_overall":                 round(tot_infer / max(tot_audio, 1e-6), 5),
        "rtf_mean_per_sample":         round(float(rtf_each.mean()), 5),
        "latency_mean_ms":             round(float(lat_arr.mean()), 2),
        "latency_std_ms":              round(float(lat_arr.std()),  2),
        "latency_p50_ms":              round(float(np.percentile(lat_arr, 50)), 2),
        "latency_p95_ms":              round(float(np.percentile(lat_arr, 95)), 2),
        "latency_p99_ms":              round(float(np.percentile(lat_arr, 99)), 2),
        "latency_min_ms":              round(float(lat_arr.min()), 2),
        "latency_max_ms":              round(float(lat_arr.max()), 2),
        "throughput_samples_per_sec":  round(n / max(tot_infer, 1e-6), 3),
        "throughput_audio_hrs_per_hr": round(
            (tot_audio / 3600) / max(tot_infer / 3600, 1e-6), 3
        ),
        "avg_output_tokens":           round(float(np.mean(output_tok_lens)), 1),
        "max_output_tokens":           int(np.max(output_tok_lens)),
        "vram_before_gb":              vram_before,
        "vram_after_gb":               vram_after,
        "vram_delta_gb":               round(vram_after - vram_before, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TENSORBOARD MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class TBWriterManager:
    """
    One SummaryWriter per (model_size, mode) group.

    Tag layout (X-axis = total_frames):
      {split}/tpf{N}/{gpu_tag}/wer_pct
      {split}/tpf{N}/{gpu_tag}/rtf_overall
      {split}/tpf{N}/{gpu_tag}/latency_{p50|p95|p99}_ms
      {split}/tpf{N}/{gpu_tag}/throughput_audio_hrs_per_hr
      flops/tpf{N}/{gpu_tag}/{total_G | encoder_G | decoder_G}
      model/{total|encoder|decoder}_params_M
    """

    def __init__(self, tb_root: str):
        self.tb_root  = Path(tb_root)
        self._writers: Dict[str, object] = {}

    def _get_writer(self, model_size: str, mode: str):
        if not TENSORBOARD_AVAILABLE:
            return None
        key = f"{model_size}_{mode}"
        if key not in self._writers:
            log_dir = self.tb_root / model_size / mode
            log_dir.mkdir(parents=True, exist_ok=True)
            self._writers[key] = SummaryWriter(log_dir=str(log_dir))
            print(f"  [TB] Writer → {log_dir}")
        return self._writers[key]

    def log_run(self, results: Dict):
        if not TENSORBOARD_AVAILABLE:
            return

        info   = results["run_info"]
        flops  = results["flops"]
        params = results["model_params"]
        splits = results["splits"]
        ms, mode = info["model_size"], info["mode"]
        tpf, tf  = info["tokens_per_frame"], info["total_frames"]
        gpu_tag  = info.get("gpu_tag", "unknown")

        w = self._get_writer(ms, mode)
        if w is None:
            return

        x = tf  # primary X-axis

        for split_name, sr in splits.items():
            pfx = f"{split_name}/tpf{tpf}/{gpu_tag}"
            for key in [
                "wer_pct", "rtf_overall", "rtf_mean_per_sample",
                "latency_mean_ms", "latency_p50_ms", "latency_p95_ms",
                "latency_p99_ms", "throughput_samples_per_sec",
                "throughput_audio_hrs_per_hr", "vram_after_gb", "avg_output_tokens",
            ]:
                w.add_scalar(f"{pfx}/{key}", sr[key], x)

        for sub in ("total_G", "encoder_G", "decoder_G"):
            w.add_scalar(f"flops/tpf{tpf}/{gpu_tag}/{sub}", flops[f"flops_{sub}"], x)

        for sub in ("total", "encoder", "decoder"):
            w.add_scalar(f"model/{sub}_params_M", params[f"{sub}_params_M"], x)

        # HParams panel
        primary = splits.get("test_clean", splits.get("test", {}))
        w.add_hparams(
            hparam_dict={
                "model_size": ms, "mode": mode,
                "tokens_per_frame": tpf, "total_frames": tf,
                "fp16": int(info.get("fp16", True)),
                "num_beams": info.get("num_beams", 1), "gpu_tag": gpu_tag,
            },
            metric_dict={
                "hparam/wer_pct":                     primary.get("wer_pct", -1),
                "hparam/rtf_overall":                 primary.get("rtf_overall", -1),
                "hparam/latency_p50_ms":              primary.get("latency_p50_ms", -1),
                "hparam/throughput_audio_hrs_per_hr": primary.get("throughput_audio_hrs_per_hr", -1),
                "hparam/flops_total_G":               flops["flops_total_G"],
            },
        )
        w.flush()
        print(f"  [TB] Logged {ms}/{mode} frames={x} tpf={tpf} gpu={gpu_tag}")

    def close_all(self):
        for w in self._writers.values():
            w.close()
        self._writers.clear()


# ─────────────────────────────────────────────────────────────────────────────
# FULL EVALUATION RUN
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    model_size: str, mode: str, checkpoint: Optional[str],
    benchmark: str, total_frames: int, tokens_per_frame: int,
    max_eval_samples: Optional[int], fp16: bool, num_beams: int,
    output_dir: str, gpu_tag: str = "unknown",
    tb_manager: Optional[TBWriterManager] = None,
) -> Dict:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*68}")
    print(f"  whisper-{model_size} | {mode} | tf={total_frames} | tpf={tokens_per_frame} | tag={gpu_tag}")
    print(f"  Runtime device: {device}"
          + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    print(f"{'='*68}")

    gpu_info = get_gpu_info()
    gpu_dict = asdict(gpu_info) if gpu_info else {"name": "CPU", "available": False}

    t0 = time.time()
    model, processor = load_model_for_inference(
        model_size=model_size, mode=mode, checkpoint=checkpoint,
        device=device, tokens_per_frame=tokens_per_frame, fp16=fp16,
    )
    load_s = round(time.time() - t0, 2)

    inner        = model.model if isinstance(model, WhisperWithTokenSubsampling) else model
    total_params = sum(p.numel() for p in inner.parameters())

    eval_splits   = load_eval_splits(benchmark, max_eval_samples)
    split_results = {}

    for split_name, split_info in eval_splits.items():
        print(f"\n  Evaluating: {split_name}")
        res = evaluate_split(
            model, processor,
            dataset=split_info["dataset"], text_col=split_info["text_col"],
            device=device, total_frames=total_frames, tokens_per_frame=tokens_per_frame,
            fp16=fp16, num_beams=num_beams,
        )
        split_results[split_name] = res
        print(f"  WER={res['wer_pct']:.3f}%  RTF={res['rtf_overall']:.4f}  "
              f"p50={res['latency_p50_ms']:.1f}ms  "
              f"tput={res['throughput_audio_hrs_per_hr']:.1f}×RT")

    actual_tok = int(round(np.mean([r["avg_output_tokens"] for r in split_results.values()])))
    flops      = estimate_flops(model_size, total_frames, tokens_per_frame, actual_tok)

    enc_p = WHISPER_PARAMS[model_size][1]
    dec_p = WHISPER_PARAMS[model_size][2]
    results = {
        "run_info": {
            "timestamp":          datetime.datetime.now().isoformat(),
            "model_size":         model_size, "mode": mode,
            "checkpoint":         checkpoint, "gpu_tag": gpu_tag,
            "benchmark":          benchmark,
            "total_frames":       total_frames, "tokens_per_frame": tokens_per_frame,
            "audio_duration_s":   round((total_frames / WHISPER_ENC_FRAMES) * WHISPER_MAX_DUR, 1),
            "fp16":               fp16, "num_beams": num_beams,
            "max_eval_samples":   max_eval_samples,
            "model_load_time_s":  load_s,
            "python_version":     platform.python_version(),
            "torch_version":      torch.__version__,
            "torchaudio_version": torchaudio.__version__,
        },
        "gpu":          gpu_dict,
        "model_params": {
            "total_params":     total_params,
            "total_params_M":   round(total_params / 1e6, 1),
            "encoder_params":   enc_p, "encoder_params_M": round(enc_p / 1e6, 1),
            "decoder_params":   dec_p, "decoder_params_M": round(dec_p / 1e6, 1),
        },
        "flops":  flops,
        "splits": split_results,
    }

    os.makedirs(output_dir, exist_ok=True)
    fname    = f"eval_{model_size}_{mode}_tf{total_frames}_tpf{tokens_per_frame}_{gpu_tag}_{benchmark}.json"
    out_path = os.path.join(output_dir, fname)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  JSON → {out_path}")

    if tb_manager is not None:
        tb_manager.log_run(results)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(all_results: List[Dict]):
    W = 115
    print(f"\n{'='*W}\n  EVALUATION SUMMARY\n{'='*W}")
    print(f"{'Model':<14} {'Mode':<9} {'Frames':>7} {'TpF':>4} {'GPU-tag':>8} "
          f"{'WER-clean':>10} {'WER-other':>10} {'RTF':>7} "
          f"{'p50-ms':>8} {'FLOPs-G':>9} {'GPU-hw':<22}")
    print("-" * W)
    for r in all_results:
        info   = r["run_info"]
        flops  = r["flops"]
        splits = r["splits"]
        gpu_hw = r["gpu"].get("name", "CPU")[:22]
        primary   = splits.get("test_clean", splits.get("test", {}))
        wc = primary.get("wer_pct",       "-")
        wo = splits.get("test_other", {}).get("wer_pct", "-")
        rtf = primary.get("rtf_overall",  "-")
        p50 = primary.get("latency_p50_ms", "-")
        def _f(v, fmt): return fmt.format(v) if isinstance(v, float) else str(v)
        print(f"  {info['model_size']:<12} {info['mode']:<9} "
              f"{info['total_frames']:>7} {info['tokens_per_frame']:>4} "
              f"{info.get('gpu_tag','?'):>8} "
              f"{_f(wc,'{:.3f}%'):>10} {_f(wo,'{:.3f}%'):>10} "
              f"{_f(rtf,'{:.4f}'):>7} {_f(p50,'{:.1f}'):>8} "
              f"{flops['flops_total_G']:>9.1f} {gpu_hw:<22}")
    print("=" * W)
    print("  TpF=tokens_per_frame | RTF↓=faster | FLOPs-G=GigaFLOPs/sample")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Whisper inference eval — compatible with whisper_finetune.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    cg = p.add_argument_group("Checkpoint strategy")
    cg.add_argument("--checkpoint_root", type=str, default=None,
                    help="Root dir to auto-discover all checkpoints.")
    cg.add_argument("--checkpoint",  type=str, default=None,
                    help="Single checkpoint dir.")
    cg.add_argument("--model_size",  type=str, default=None,
                    choices=["tiny", "base", "small", "medium", "large-v3"])
    cg.add_argument("--mode",        type=str, default=None,
                    choices=["baseline", "lora", "full"])

    fg = p.add_argument_group("Discovery filters")
    fg.add_argument("--filter_model_size", type=str, default=None)
    fg.add_argument("--filter_mode",       type=str, default=None)

    ag = p.add_argument_group("Compute axes / sweeps")
    ag.add_argument("--tokens_per_frame",      type=int, default=None,
                    help="Override tpf from experiment_cfg.json.")
    ag.add_argument("--total_frames",          type=int, default=None,
                    help="Override total_frames from experiment_cfg.json.")
    ag.add_argument("--sweep_tokens_per_frame", type=str, default=None)
    ag.add_argument("--sweep_total_frames",     type=str, default=None)

    p.add_argument("--benchmark",        type=str, default="librispeech",
                   choices=["librispeech", "common_voice", "fleurs"])
    p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--fp16",             action="store_true", default=True)
    p.add_argument("--num_beams",        type=int, default=1)
    p.add_argument("--output_dir",       type=str,
                   default="/home/vyomwal5/SAME/eval_results")
    p.add_argument("--no_tensorboard",   action="store_true", default=False)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Hardware banner
    print("\n" + "="*68 + "\n  HARDWARE INFO\n" + "="*68)
    gpu = get_gpu_info()
    if gpu:
        print(f"  GPU              : {gpu.name}")
        print(f"  VRAM total/free  : {gpu.total_vram_gb:.1f} / {gpu.free_vram_gb:.1f} GB")
        print(f"  Compute cap.     : {gpu.compute_capability}  SMs={gpu.sm_count}")
        print(f"  Driver / CUDA    : {gpu.driver_version} / {gpu.cuda_version}")
        print(f"  PyTorch          : {torch.__version__}")
    else:
        print("  Running on CPU (no CUDA GPU detected)")
    print("="*68)

    # TensorBoard
    tb_root    = os.path.join(args.output_dir, "tensorboard")
    tb_manager = None
    if not args.no_tensorboard:
        if TENSORBOARD_AVAILABLE:
            tb_manager = TBWriterManager(tb_root)
            print(f"\n  TensorBoard → {tb_root}")
            print(f"  Launch: tensorboard --logdir {tb_root}\n")
        else:
            print("  WARNING: TensorBoard not available.")

    # ── Build job list ────────────────────────────────────────────────────────
    @dataclass
    class EvalJob:
        model_size: str; mode: str; checkpoint: Optional[str]
        tokens_per_frame: int; total_frames: int; gpu_tag: str = "runtime"

    eval_jobs: List[EvalJob] = []

    if args.checkpoint_root:
        filter_sizes = ([s.strip() for s in args.filter_model_size.split(",")]
                        if args.filter_model_size else None)
        filter_modes = ([m.strip() for m in args.filter_mode.split(",")]
                        if args.filter_mode else None)
        specs = discover_checkpoints(
            args.checkpoint_root,
            filter_model_sizes=filter_sizes,
            filter_modes=filter_modes,
        )
        for spec in specs:
            tpf = args.tokens_per_frame if args.tokens_per_frame is not None else spec.tokens_per_frame
            tf  = args.total_frames     if args.total_frames     is not None else spec.total_frames
            eval_jobs.append(EvalJob(
                model_size=spec.model_size, mode=spec.mode,
                checkpoint=spec.path, tokens_per_frame=tpf,
                total_frames=tf, gpu_tag=spec.gpu_tag,
            ))

    else:
        assert args.model_size, "--model_size required without --checkpoint_root"
        assert args.mode,       "--mode required without --checkpoint_root"

        saved = _read_experiment_cfg(args.checkpoint) if args.checkpoint else {}
        base_tpf = args.tokens_per_frame if args.tokens_per_frame is not None \
                   else saved.get("tokens_per_frame", 1)
        base_tf  = args.total_frames if args.total_frames is not None \
                   else saved.get("total_frames", 1500)

        configs: List[Tuple[int, int]] = []
        if args.sweep_tokens_per_frame:
            for tpf in [int(x) for x in args.sweep_tokens_per_frame.split(",")]:
                configs.append((base_tf, tpf))
        elif args.sweep_total_frames:
            for tf in [int(x) for x in args.sweep_total_frames.split(",")]:
                configs.append((tf, base_tpf))
        else:
            configs.append((base_tf, base_tpf))

        for tf, tpf in configs:
            eval_jobs.append(EvalJob(
                model_size=args.model_size, mode=args.mode,
                checkpoint=args.checkpoint, tokens_per_frame=tpf, total_frames=tf,
            ))

    print(f"\n  Jobs queued: {len(eval_jobs)}")
    for j in eval_jobs:
        dur = round((j.total_frames / WHISPER_ENC_FRAMES) * WHISPER_MAX_DUR, 1)
        print(f"    {j.model_size:<10} {j.mode:<9} tf={j.total_frames}({dur}s) "
              f"tpf={j.tokens_per_frame}  tag={j.gpu_tag}")

    all_results = []
    for job in eval_jobs:
        result = run_evaluation(
            model_size=job.model_size, mode=job.mode, checkpoint=job.checkpoint,
            benchmark=args.benchmark, total_frames=job.total_frames,
            tokens_per_frame=job.tokens_per_frame,
            max_eval_samples=args.max_eval_samples,
            fp16=args.fp16, num_beams=args.num_beams,
            output_dir=args.output_dir, gpu_tag=job.gpu_tag,
            tb_manager=tb_manager,
        )
        all_results.append(result)

    print_summary(all_results)

    if len(all_results) > 1:
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(args.output_dir, f"sweep_{args.benchmark}_{ts}.json")
        with open(path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Combined sweep JSON → {path}")

    if tb_manager is not None:
        tb_manager.close_all()
        print(f"\n  tensorboard --logdir {tb_root}")


if __name__ == "__main__":
    main()
