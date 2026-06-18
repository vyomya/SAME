"""
inference_eval.py — Comprehensive Inference Evaluation for Whisper ASR
=======================================================================
Compatible with whisper_finetune.py — handles:
  - Plain WhisperForConditionalGeneration  (mode=baseline / mode=full)
  - PeftModel (LoRA) wrapped in WhisperWithTokenSubsampling  (mode=lora, tpf>1)
  - experiment_cfg.json auto-read from checkpoint directory
  - GPU-agnostic loading (A100 / V100 / H100 / CPU)

Quantization axis (xQ) — post-training, applied after loading:
  fp16  → model trained in fp16 OR fp32 model cast to fp16
  int8  → torch.quantization.quantize_dynamic (CPU-friendly, no retraining)
  int4  → bitsandbytes NF4 (GPU only, requires bitsandbytes>=0.41)

  Pipeline:
    FP16-trained model  → evaluate as-is (fp16)
                        → cast .float() → quantize to int8 / int4

    FP32-trained model  → evaluate as-is (fp32) [baseline reference]
                        → cast .half() → evaluate as fp16
                        → cast .float() → quantize to int8 / int4

Pruning axis (xP) — structured attention-head pruning, applied after loading:
  0.0 → dense (no pruning)
  0.1 → remove 10% of attention heads per encoder layer
  0.2 / 0.3 / 0.4 → increasing pruning ratio
  NOTE: inference_eval applies pruning WITHOUT recovery finetuning.
        For results with recovery, use prune_and_recover() in finetune.py
        then point --checkpoint at the pruned+recovered directory.

Computes and logs:
  - WER  (test_other)
  - Real-Time Factor (RTF)
  - Theoretical inference FLOPs (precision-scaled)
  - Measured latency per sample (mean, p50, p95, p99) via CUDA events
  - Model size on disk (MB), peak GPU RAM (MB)
  - GPU name, VRAM usage, compute capability
  - Throughput (samples/sec, audio-hours/hour)
  - Model parameter counts (total, encoder, decoder)
  - TensorBoard scalars + HParam panel

Output:
  - Per-run JSON  → {output_dir}/eval_{size}_{mode}_{quant}_{prune}_{bench}.json
  - TensorBoard   → {output_dir}/tensorboard/{model_size}/{mode}/
  - Combined JSON → {output_dir}/sweep_{bench}_{timestamp}.json  (multi-run)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Standard fp16 eval (existing behaviour — unchanged)
python inference_eval.py \\
    --model_size small --mode lora \\
    --checkpoint /path/to/ckpt --benchmark librispeech

# 2. Sweep all quantization levels on one checkpoint
python inference_eval.py \\
    --model_size small --mode lora \\
    --checkpoint /path/to/ckpt \\
    --sweep_quantization fp16,int8,int4

# 3. Evaluate a pruned checkpoint (pruned+recovered externally)
python inference_eval.py \\
    --model_size large-v3 --mode lora \\
    --checkpoint /path/to/ckpt_pruned30 \\
    --quantization int8 --pruning_ratio 0.3

# 4. Sweep quantization × pruning at once
python inference_eval.py \\
    --model_size large-v3 --mode lora \\
    --checkpoint /path/to/ckpt \\
    --sweep_quantization fp16,int8,int4 \\
    --sweep_pruning_ratio 0.0,0.1,0.2,0.3

# 5. Auto-discover all checkpoints, evaluate each at int8
python inference_eval.py \\
    --checkpoint_root /home/vyomwal5/SAME/checkpoints \\
    --quantization int8 --benchmark librispeech

# 6. FP32 model: auto-cascade to fp16 + int8 + int4
python inference_eval.py \\
    --model_size medium --mode full \\
    --checkpoint /path/to/fp32_ckpt \\
    --training_dtype fp32 --sweep_quantization fp16,int8,int4

TensorBoard:
  tensorboard --logdir /home/vyomwal5/SAME/eval_results/tensorboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ─────────────────────────────────────────────────────────────────────────────
# CACHE — before ALL imports
# ─────────────────────────────────────────────────────────────────────────────
from transformers.utils import logging

logging.set_verbosity_error()
import os

CACHE_DIR = "/fs/nexus-scratch/vyomwal5/anaconda3/envs/whisper/hf_cache"

# os.environ["LD_LIBRARY_PATH"] = (
#     "/scratch/zt1/project/msml604/user/vyomwal5/anaconda3/envs/same/lib:"
#     "/scratch/zt1/project/msml604/user/vyomwal5/anaconda3/envs/same/lib/python3.11/site-packages/torch/lib:"
#     "/scratch/zt1/project/msml604/user/vyomwal5/anaconda3/envs/same/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:"
#     "/scratch/zt1/project/msml604/user/vyomwal5/anaconda3/envs/same/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:"
#     "/scratch/zt1/project/msml604/user/vyomwal5/anaconda3/envs/same/lib/python3.11/site-packages/nvidia/npp/lib:"
#     + os.environ.get("LD_LIBRARY_PATH", "")
# )

LOCAL_PATH = {
    "small":    "/fs/nexus-scratch/vyomwal5/models/models/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d",
    "medium":   "/fs/nexus-scratch/vyomwal5/models/models/models--openai--whisper-medium/snapshots/abdf7c39ab9d0397620ccaea8974cc764cd0953e",
    "tiny":     "/fs/nexus-scratch/vyomwal5/models/models/models--openai--whisper-tiny/snapshots/169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
    "large-v3": "/fs/nexus-scratch/vyomwal5/models/models/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1",
    # Distilled — will download on first run; add local paths after caching
    "distil-small":  "/fs/nexus-scratch/vyomwal5/models/distil-small",
    "distil-medium": "/fs/nexus-scratch/vyomwal5/models/distil-medium",
    "distil-large":  "/fs/nexus-scratch/vyomwal5/models/distil-large",
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
import copy
import datetime
import argparse
import platform
import subprocess
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field

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
    "tiny":          (39_000_000,    14_000_000,  25_000_000),
    "base":          (74_000_000,    24_000_000,  50_000_000),
    "small":         (244_000_000,   88_000_000, 156_000_000),
    "medium":        (769_000_000,  307_000_000, 462_000_000),
    "large-v3":      (1_540_000_000, 633_000_000, 907_000_000),
    "distil-small":  (166_000_000,   88_000_000,  78_000_000),
    "distil-medium": (394_000_000,  307_000_000,  87_000_000),
    "distil-large":  (756_000_000,  633_000_000, 123_000_000),
}

WHISPER_D_MODEL = {
    "tiny": 384, "base": 512, "small": 768, "medium": 1024, "large-v3": 1280,
    "distil-small": 768, "distil-medium": 1024, "distil-large": 1280,
}

WHISPER_LAYERS = {
    "tiny":          (4,  4),  "base":         (6,  6),
    "small":         (12, 12), "medium":        (24, 24),
    "large-v3":      (32, 32), "distil-small":  (12, 2),
    "distil-medium": (24, 2),  "distil-large":  (32, 2),
}

# Precision → effective FLOP multiplier relative to FP32
# Based on hardware throughput ratios for Tensor Core accelerated ops.
# INT8: 4× FP32 throughput on A100/V100 with INT8 Tensor Cores
# INT4: 8× FP32 throughput (theoretical, hardware dependent)
PRECISION_FLOP_SCALE = {
    "fp32": 1.0,
    "fp16": 0.5,    # 2× throughput vs FP32
    "int8": 0.25,   # 4× throughput vs FP32
    "int4": 0.125,  # 8× throughput vs FP32
}

# Bytes per parameter for model size calculation
PRECISION_BYTES = {
    "fp32": 4,
    "fp16": 2,
    "int8": 1,
    "int4": 0.5,
}

VALID_QUANTIZATIONS  = ["fp16", "int8", "int4"]
VALID_PRUNING_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4]

# Folder name pattern produced by whisper_finetune.py
CHECKPOINT_NAME_RE = re.compile(
    r"^whisper-(?P<size>tiny|base|small|medium|large-v3)"
    r"-(?P<mode>lora|full)"
    r"-(?P<lorar>8|16||64)"
    r"-librispeech-asr"
    r"-tpf(?P<tpf>\d+)"
    r"-tf(?P<frames>\d+)"
    r"(?:-(?P<gpu>[a-zA-Z0-9]+))?$"
)


# ─────────────────────────────────────────────────────────────────────────────
# WhisperWithTokenSubsampling  (must stay in sync with finetune.py)
# ─────────────────────────────────────────────────────────────────────────────

class WhisperWithTokenSubsampling(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, tokens_per_frame: int = 1):
        super().__init__()
        self.model            = base_model
        self.tokens_per_frame = tokens_per_frame

    def _get_encoder(self):
        m = self.model
        if isinstance(m, PeftModel):
            m = m.base_model.model
        whisper_model = getattr(m, "model", m)
        encoder = getattr(whisper_model, "encoder", None)
        if encoder is None:
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
# QUANTIZATION  (xQ axis)
# ─────────────────────────────────────────────────────────────────────────────

def apply_quantization(
    model:            torch.nn.Module,
    quantization:     str,
    training_dtype:   str,
    device:           torch.device,
    model_size:       str,
    checkpoint:       Optional[str] = None,
) -> torch.nn.Module:
    """
    Apply post-training quantization to a loaded, merged model.

    Handles both FP16-trained and FP32-trained models:
      FP16 trained:
        fp16 → no-op (already fp16)
        int8 → .float() then quantize_dynamic
        int4 → reload with bitsandbytes 4-bit config

      FP32 trained:
        fp16 → .half()
        int8 → quantize_dynamic (already fp32, correct dtype)
        int4 → reload with bitsandbytes 4-bit config

    The WhisperWithTokenSubsampling wrapper (if present) is stripped before
    quantization and re-applied after, so quantization sees a plain
    WhisperForConditionalGeneration and its .generate() path is unaffected.

    Parameters
    ----------
    model           : loaded model (may be wrapped in WhisperWithTokenSubsampling)
    quantization    : one of "fp16" | "int8" | "int4"
    training_dtype  : "fp16" or "fp32" — the dtype the model was trained in
    device          : target device after quantization
    model_size      : Whisper size key for reloading (int4 requires reload)
    checkpoint      : checkpoint dir for reloading finetuned weights (int4)
    """
    assert quantization in VALID_QUANTIZATIONS, \
        f"quantization must be one of {VALID_QUANTIZATIONS}, got '{quantization}'"

    print(f"\n  [xQ] Applying quantization: {training_dtype} → {quantization}")

    # ── Unwrap subsampling wrapper if present ─────────────────────────────────
    tokens_per_frame = 1
    if isinstance(model, WhisperWithTokenSubsampling):
        tokens_per_frame = model.tokens_per_frame
        inner_model      = model.model
        print(f"  [xQ] Unwrapped subsampling (tpf={tokens_per_frame}) for quantization.")
    else:
        inner_model = model

    # ── FP16: cast or no-op ───────────────────────────────────────────────────
    if quantization == "fp16":
        if training_dtype == "fp16":
            print("  [xQ] Model already in fp16 — no conversion needed.")
            quantized = inner_model
        else:
            print("  [xQ] Casting fp32 → fp16 ...")
            quantized = inner_model.half()
        quantized = quantized.to(device)

    # ── INT8: torch.quantization.quantize_dynamic ─────────────────────────────
    # quantize_dynamic requires fp32 input weights.
    # Runs on CPU; move back to device after.
    # Compatible with V100/A100/H100 — no special hardware required.
    elif quantization == "int8":
        print("  [xQ] Converting to fp32 for INT8 quantization...")
        fp32_model = inner_model.float().cpu()   # quantize_dynamic needs CPU fp32

        print("  [xQ] Applying INT8 dynamic quantization (torch.quantization)...")
        quantized = torch.quantization.quantize_dynamic(
            fp32_model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        # INT8 dynamic quantization runs on CPU only — do NOT move to CUDA.
        # CUDA execution will silently fall back to fp32 on most hardware.
        print("  [xQ] INT8 quantization complete. Model will run on CPU.")
        device = torch.device("cpu")   # override for INT8

    # ── INT4: bitsandbytes NF4 ────────────────────────────────────────────────
    # bitsandbytes requires GPU. Loads a fresh copy of the model with
    # 4-bit quantization config, then overlays finetuned weights.
    elif quantization == "int4":
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "bitsandbytes not installed. "
                "Run: pip install bitsandbytes>=0.41.0"
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "INT4 quantization (bitsandbytes) requires a CUDA GPU. "
                "Use int8 for CPU-only evaluation."
            )

        print("  [xQ] Loading INT4 model via bitsandbytes NF4 ...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,    # double quant saves ~0.4 bits/param
            bnb_4bit_quant_type="nf4",         # NF4 optimal for normally-distributed weights
        )
        base_path = LOCAL_PATH[model_size]
        quantized = WhisperForConditionalGeneration.from_pretrained(
            base_path,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        quantized.config.forced_decoder_ids         = None
        quantized.generation_config.suppress_tokens = []
        quantized.config.use_cache                  = True

        # Overlay finetuned weights (non-quantized layers e.g. layer norms)
        print("  [xQ] Overlaying finetuned weights onto INT4 base ...")
        ft_state   = inner_model.state_dict()
        quant_state = quantized.state_dict()
        # Only copy weights that exist in both and have matching shapes
        copied = 0
        for key in ft_state:
            if key in quant_state and ft_state[key].shape == quant_state[key].shape:
                try:
                    quant_state[key].copy_(ft_state[key].to(quant_state[key].dtype))
                    copied += 1
                except Exception:
                    pass
        quantized.load_state_dict(quant_state, strict=False)
        print(f"  [xQ] INT4 overlay complete ({copied} tensors matched).")
        # device_map="auto" handles placement — don't call .to(device)
        device = torch.device("cuda")

    # ── Re-wrap with subsampling if it was present ────────────────────────────
    if tokens_per_frame > 1:
        quantized = WhisperWithTokenSubsampling(quantized, tokens_per_frame)
        print(f"  [xQ] Re-applied subsampling wrapper (tpf={tokens_per_frame}).")

    if quantization != "int4":
        quantized = quantized.to(device)

    quantized.eval()
    print(f"  [xQ] Done. Running on: {device}")
    return quantized, device


# ─────────────────────────────────────────────────────────────────────────────
# PRUNING  (xP axis — inference-time, no recovery)
# ─────────────────────────────────────────────────────────────────────────────

def apply_pruning(
    model:         torch.nn.Module,
    pruning_ratio: float,
) -> torch.nn.Module:
    """
    Apply structured attention-head pruning to the encoder of a loaded model.

    Pruning is based on accumulated weight MAGNITUDE (L1 norm of q_proj rows
    grouped by head), which is a fast proxy for importance at inference time.
    No gradient computation or recovery finetuning is performed here.

    For results WITH recovery finetuning, use prune_and_recover() in
    finetune.py and point --checkpoint at the recovered directory.

    Parameters
    ----------
    model         : loaded model (may be wrapped in WhisperWithTokenSubsampling)
    pruning_ratio : fraction of attention heads to remove per encoder layer
                    e.g. 0.3 removes the 30% least-important heads per layer
    """
    if pruning_ratio == 0.0:
        print("  [xP] pruning_ratio=0.0 — no pruning applied.")
        return model

    assert 0.0 < pruning_ratio <= 0.4, \
        f"pruning_ratio must be in (0.0, 0.4], got {pruning_ratio}. " \
        f"Beyond 0.4 risks WER collapse without recovery finetuning."

    print(f"\n  [xP] Applying structured pruning: ratio={pruning_ratio:.0%}")

    # ── Unwrap subsampling wrapper if present ─────────────────────────────────
    tokens_per_frame = 1
    if isinstance(model, WhisperWithTokenSubsampling):
        tokens_per_frame = model.tokens_per_frame
        inner_model      = model.model
    else:
        inner_model = model

    # ── Navigate to WhisperEncoder ────────────────────────────────────────────
    whisper_model = getattr(inner_model, "model", inner_model)
    encoder       = getattr(whisper_model, "encoder", None)
    if encoder is None:
        encoder = getattr(
            getattr(whisper_model, "model", whisper_model), "encoder"
        )

    # ── Compute per-head importance (L1 magnitude of q_proj rows) ────────────
    heads_to_prune = {}
    total_pruned   = 0
    total_heads    = 0

    for layer_idx, layer in enumerate(encoder.layers):
        n_heads  = layer.self_attn.num_heads
        head_dim = layer.self_attn.head_dim if hasattr(layer.self_attn, "head_dim") \
                   else layer.self_attn.q_proj.weight.shape[0] // n_heads

        # L1 norm of q_proj weight rows, grouped by head
        q_weight = layer.self_attn.q_proj.weight.data   # (d_model, d_model)
        # Reshape to (n_heads, head_dim, d_model) and compute mean L1 per head
        importance = q_weight.view(n_heads, head_dim, -1).abs().mean(dim=(1, 2))

        n_prune = max(1, int(n_heads * pruning_ratio))
        # Prune heads with lowest importance
        _, sorted_idx     = torch.sort(importance)
        heads_to_prune[layer_idx] = sorted_idx[:n_prune].tolist()
        total_pruned  += n_prune
        total_heads   += n_heads

    print(f"  [xP] Pruning {total_pruned}/{total_heads} encoder heads "
          f"({100 * total_pruned / total_heads:.1f}%) across "
          f"{len(encoder.layers)} layers.")

    # ── Apply pruning via HuggingFace prune_heads API ─────────────────────────
    # prune_heads expects {layer_index: set_of_head_indices}
    # For WhisperForConditionalGeneration the correct call path is:
    #   model.model.encoder.layers[i].self_attn  (self attention only)
    #   model.prune_heads({"encoder": heads_to_prune})
    try:
        inner_model.prune_heads({"encoder": heads_to_prune})
        print("  [xP] prune_heads() succeeded.")
    except Exception as e:
        print(f"  [xP] prune_heads() raised {e}. "
              f"Attempting manual head zeroing fallback...")
        # Fallback: zero out the pruned head dimensions rather than removing them.
        # WER impact is similar to structural removal at small ratios.
        for layer_idx, head_list in heads_to_prune.items():
            layer    = encoder.layers[layer_idx]
            n_heads  = layer.self_attn.num_heads
            head_dim = layer.self_attn.q_proj.weight.shape[0] // n_heads
            for head_idx in head_list:
                start = head_idx * head_dim
                end   = start + head_dim
                with torch.no_grad():
                    layer.self_attn.q_proj.weight.data[start:end] = 0.0
                    layer.self_attn.k_proj.weight.data[start:end] = 0.0
                    layer.self_attn.v_proj.weight.data[start:end] = 0.0
        print(f"  [xP] Manual head zeroing complete.")

    # ── Re-wrap ───────────────────────────────────────────────────────────────
    if tokens_per_frame > 1:
        model = WhisperWithTokenSubsampling(inner_model, tokens_per_frame)
    else:
        model = inner_model

    print(f"  [xP] Pruning applied. "
          f"{'Note: no recovery finetuning — WER may be higher than reported in paper.' if pruning_ratio >= 0.2 else ''}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# MODEL SIZE MEASUREMENT
# ─────────────────────────────────────────────────────────────────────────────

def measure_model_size_mb(model: torch.nn.Module, quantization: str) -> float:
    """
    Measure actual model size in MB based on stored tensor dtypes.
    More accurate than param_count × bytes_per_param for mixed-precision
    models where some layers stay in fp32 (e.g. layer norms in INT8).
    """
    inner = model.model if isinstance(model, WhisperWithTokenSubsampling) else model
    total_bytes = 0
    for param in inner.parameters():
        if param.dtype == torch.float32:
            total_bytes += param.nelement() * 4
        elif param.dtype in (torch.float16, torch.bfloat16):
            total_bytes += param.nelement() * 2
        elif param.dtype == torch.int8:
            total_bytes += param.nelement() * 1
        else:
            # INT4 packed tensors — bitsandbytes stores them as uint8
            # with 2 values per byte; nelement() reports logical count
            total_bytes += param.nelement() * 0.5
    return round(total_bytes / (1024 ** 2), 2)


def measure_peak_ram_mb() -> float:
    """Peak GPU RAM in MB since last reset. Falls back to CPU RSS."""
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
    else:
        try:
            import psutil
            return round(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2), 2)
        except ImportError:
            return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CheckpointSpec:
    path:             str
    model_size:       str
    mode:             str
    lorar:            int
    tokens_per_frame: int
    total_frames:     int
    gpu_tag:          str

    def __str__(self):
        return (f"whisper-{self.model_size} mode={self.mode} lora_r={self.lorar}"
                f"tpf={self.tokens_per_frame} frames={self.total_frames} "
                f"gpu={self.gpu_tag}  [{self.path}]")


def _read_experiment_cfg(ckpt_path: str) -> Dict:
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
        lorar = int(m.group("lorar"))
        tpf  = int(m.group("tpf"))
        tf   = int(m.group("frames"))
        gpu  = m.group("gpu") or "unknown"
        if filter_model_sizes and size not in filter_model_sizes:
            continue
        if filter_modes and mode not in filter_modes:
            continue
        if not any((entry / f).exists()
                   for f in ("adapter_config.json", "config.json")):
            print(f"  [discover] Skip (no config file): {entry.name}")
            continue
        saved = _read_experiment_cfg(str(entry))
        tpf   = saved.get("tokens_per_frame", tpf)
        tf    = saved.get("total_frames", tf)
        specs.append(CheckpointSpec(
            path=str(entry), model_size=size, mode=mode, lorar =lorar,
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
# MODEL LOADING
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
    Load base model + LoRA merge (if applicable) to CPU, then move to device.
    Quantization and pruning are applied AFTER this function returns,
    via apply_quantization() and apply_pruning().

    Always loads in the training dtype (fp16 or fp32) — conversion happens
    in apply_quantization().
    """
    model_name = LOCAL_PATH[model_size]
    dtype      = torch.float16 if fp16 else torch.float32

    # ── Mel bin detection ─────────────────────────────────────────────────────
    _base_cfg_path = os.path.join(model_name, "config.json")
    _num_mel_bins  = 80
    if os.path.exists(_base_cfg_path):
        with open(_base_cfg_path) as _f:
            _num_mel_bins = json.load(_f).get("num_mel_bins", _num_mel_bins)
    print(f"\n  [mel] Conv1d expects {_num_mel_bins} mel bins")

    # ── Processor loading ─────────────────────────────────────────────────────
    print(f"  Loading feature extractor from base model: {model_name}")
    processor = WhisperProcessor.from_pretrained(
        model_name, language="English", task="transcribe"
    )
    if checkpoint and os.path.exists(os.path.join(checkpoint, "tokenizer_config.json")):
        from transformers import WhisperTokenizer
        ckpt_tokenizer = WhisperTokenizer.from_pretrained(
            checkpoint, language="English", task="transcribe"
        )
        processor.tokenizer = ckpt_tokenizer
        print(f"  Tokenizer loaded from checkpoint.")

    # ── Enforce mel bins ──────────────────────────────────────────────────────
    _loaded_mel = getattr(processor.feature_extractor, "feature_size", None)
    if _loaded_mel != _num_mel_bins:
        print(f"  [mel] MISMATCH: overriding feature_size {_loaded_mel} → {_num_mel_bins}")
        processor.feature_extractor.feature_size = _num_mel_bins
        processor.feature_extractor.num_mel_bins = _num_mel_bins
        try:
            import librosa
            processor.feature_extractor.mel_filters = librosa.filters.mel(
                sr=WHISPER_SR, n_fft=processor.feature_extractor.n_fft,
                n_mels=_num_mel_bins, fmin=0.0, fmax=WHISPER_SR // 2,
            )
        except Exception as _e:
            print(f"  [mel] librosa unavailable ({_e}); proceeding.")
    else:
        print(f"  [mel] Feature extractor confirmed: {_num_mel_bins} bins. ✓")

    # ── Load base weights to CPU ──────────────────────────────────────────────
    print(f"  Loading Whisper-{model_size} to CPU (dtype={dtype}) ...")
    base = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    base.config.forced_decoder_ids         = None
    base.generation_config.suppress_tokens = []
    base.config.use_cache                  = True

    # ── Mode-specific loading ─────────────────────────────────────────────────
    if mode == "baseline":
        model = base

    elif mode == "lora":
        assert checkpoint, "--checkpoint required for mode=lora"
        print(f"  Applying LoRA from: {checkpoint}")
        peft_model = PeftModel.from_pretrained(base, checkpoint, is_trainable=False)
        model      = peft_model.merge_and_unload()
        print("  LoRA merged into base weights.")

    elif mode == "full":
        assert checkpoint, "--checkpoint required for mode=full"
        print(f"  Loading full finetune from: {checkpoint}")
        model = WhisperForConditionalGeneration.from_pretrained(
            checkpoint, torch_dtype=dtype, low_cpu_mem_usage=True,
        )
        model.config.forced_decoder_ids         = None
        model.generation_config.suppress_tokens = []
        model.config.use_cache                  = True

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose: baseline | lora | full")

    # ── Subsampling wrapper ───────────────────────────────────────────────────
    if tokens_per_frame > 1:
        model = WhisperWithTokenSubsampling(model, tokens_per_frame)
        print(f"  [xV] Wrapper applied: stride={tokens_per_frame} "
              f"→ {WHISPER_ENC_FRAMES // tokens_per_frame} encoder tokens.")
    else:
        print("  [xV] No subsampling (tokens_per_frame=1).")

    # ── Move to device ────────────────────────────────────────────────────────
    print(f"  Moving to {device} ...")
    model = model.to(device).eval()

    inner        = model.model if isinstance(model, WhisperWithTokenSubsampling) else model
    total_params = sum(p.numel() for p in inner.parameters())
    trainable    = sum(p.numel() for p in inner.parameters() if p.requires_grad)
    print(f"  Params  total={total_params:,}  trainable={trainable:,}")

    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
# FLOP ESTIMATION  (precision-aware)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_flops(
    model_size:       str,
    total_frames:     int,
    tokens_per_frame: int,
    quantization:     str = "fp16",
    pruning_ratio:    float = 0.0,
    avg_output_tokens: int = 50,
) -> Dict:
    """
    Theoretical inference FLOPs, scaled for quantization and pruning.

    Quantization scaling: INT8 ops are 4× cheaper than FP32 on hardware
    with INT8 tensor cores (A100, V100 with INT8 IMMA).
    Pruning scaling: removing pruning_ratio fraction of heads reduces
    attention FLOPs proportionally.
    """
    d            = WHISPER_D_MODEL.get(model_size, 1024)
    n_enc, n_dec = WHISPER_LAYERS.get(model_size, (24, 24))
    T            = total_frames
    T_dec        = T // tokens_per_frame
    L            = avg_output_tokens

    # Effective head count after pruning
    head_scale   = 1.0 - pruning_ratio   # fraction of heads remaining

    # Encoder FLOPs (self-attention + FFN)
    # Self-attention FLOPs scale with head_scale; FFN is unaffected by pruning
    enc_attn = n_enc * (4 * T * d * d + 2 * T * T * d) * head_scale
    enc_ffn  = n_enc * 8 * T * d * d    # FFN: 2 linear layers, each 2*T*d*(4d)
    enc      = enc_attn + enc_ffn

    # Decoder FLOPs (cross-attention + FFN)
    dec = n_dec * (2 * T_dec * d * d + 2 * d * d + 8 * d * d) * L

    tot = enc + dec

    # Precision scaling — effective FLOPs accounting for hardware throughput
    prec_scale  = PRECISION_FLOP_SCALE.get(quantization, 1.0)
    enc_eff     = enc * prec_scale
    dec_eff     = dec * prec_scale
    tot_eff     = tot * prec_scale

    return {
        # Raw FLOPs (precision-agnostic, for architecture comparison)
        "flops_encoder_G":          round(enc     / 1e9, 3),
        "flops_decoder_G":          round(dec     / 1e9, 3),
        "flops_total_G":            round(tot     / 1e9, 3),
        # Effective FLOPs (precision-weighted, for deployment comparison)
        "flops_encoder_eff_G":      round(enc_eff / 1e9, 3),
        "flops_decoder_eff_G":      round(dec_eff / 1e9, 3),
        "flops_total_eff_G":        round(tot_eff / 1e9, 3),
        "precision_flop_scale":     prec_scale,
        "pruning_head_scale":       head_scale,
        "encoder_frames":           T,
        "decoder_cross_attn_len":   T_dec,
        "avg_output_tokens":        L,
        "quantization":             quantization,
        "pruning_ratio":            pruning_ratio,
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
    tokens_per_frame: int,
    fp16:             bool = True,
    quantization:     str  = "fp16",
) -> torch.Tensor:
    """
    xT axis: clip audio.
    Feature extractor always produces (num_mel_bins, 3000).
    Dtype matches quantization mode:
      fp16 / fp32  → half / float
      int8 / int4  → float32 (quantization happens inside the model)
    """
    max_samples = int((total_frames / WHISPER_ENC_FRAMES) * WHISPER_MAX_DUR * WHISPER_SR)
    audio_array = audio_array[:max_samples]

    features = processor.feature_extractor(
        audio_array.astype(np.float32), sampling_rate=WHISPER_SR,
    ).input_features[0]

    feat = torch.tensor(features).unsqueeze(0).to(device)

    # INT8 runs on CPU in fp32; INT4 uses fp16 compute internally via bnb
    if quantization == "int8":
        feat = feat.float()
    elif quantization == "int4":
        feat = feat.half()
    elif fp16:
        feat = feat.half()
    else:
        feat = feat.float()

    return feat


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_eval_splits(benchmark: str, max_samples: Optional[int]) -> Dict:
    splits = {}
    if benchmark == "librispeech":
        for cfg, label in [("other", "test_other")]:
            print(f"  Loading librispeech [{cfg}] test (streaming=False)...")
            ds = load_dataset("librispeech_asr", cfg, split=f"test", streaming=False)
            if max_samples:
                ds = ds.take(max_samples)
            splits[label] = {"dataset": ds, "text_col": "text"}
    elif benchmark == "common_voice":
        print("  Loading common_voice [en] test...")
        ds = load_dataset(
            "mozilla-foundation/common_voice_13_0", "en",
            split="test", streaming=True,
        )
        if max_samples:
            ds = ds.take(max_samples)
        splits["test"] = {"dataset": ds, "text_col": "sentence"}
    elif benchmark == "fleurs":
        print("  Loading fleurs [en_us] test...")
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
    quantization: str = "fp16",
) -> Dict:
    """
    Inference loop for one dataset split.
    CUDA event timing for GPU; wall-clock fallback for CPU (INT8).
    Resets peak memory stats before the loop for accurate RAM measurement.
    """
    all_preds, all_refs      = [], []
    latencies_ms, audio_durs = [], []
    output_tok_lens          = []
    use_cuda                 = (device.type == "cuda")
    vram_before              = get_vram_used_gb()

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    # ── Warmup ────────────────────────────────────────────────────────────────
    _mel_bins    = getattr(processor.feature_extractor, "feature_size", 80)
    dummy_dtype  = torch.float32 if quantization == "int8" \
                   else torch.float16 if fp16 else torch.float32
    dummy_device = torch.device("cpu") if quantization == "int8" else device
    print(f"  Warming up ({warmup_steps} steps, mel_bins={_mel_bins}, "
          f"quant={quantization}, device={dummy_device})...")
    dummy = torch.zeros(1, _mel_bins, 3000, dtype=dummy_dtype, device=dummy_device)
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
            quantization=quantization,
        )

        # ── Timing ────────────────────────────────────────────────────────────
        if use_cuda:
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            with torch.no_grad():
                pred_ids = model.generate(
                    features, language="en", task="transcribe",
                    num_beams=num_beams, max_new_tokens=max_new_tokens,
                )
            ev1.record()
            torch.cuda.synchronize()
            lat_ms = ev0.elapsed_time(ev1)
        else:
            # CPU path (INT8 dynamic quantization)
            t0 = time.perf_counter()
            with torch.no_grad():
                pred_ids = model.generate(
                    features, language="en", task="transcribe",
                    num_beams=num_beams, max_new_tokens=max_new_tokens,
                )
            lat_ms = (time.perf_counter() - t0) * 1000.0

        pred_text = processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True)
        all_preds.append(normalize_text(pred_text))
        all_refs.append(normalize_text(ref_text))
        latencies_ms.append(lat_ms)
        audio_durs.append(dur_s)
        output_tok_lens.append(pred_ids.shape[1])

    vram_after    = get_vram_used_gb()
    peak_ram_mb   = measure_peak_ram_mb()

    wer_val  = jiwer_wer(all_refs, all_preds)
    lat_arr  = np.array(latencies_ms)
    dur_arr  = np.array(audio_durs)
    rtf_each = lat_arr / 1000.0 / np.maximum(dur_arr, 1e-6)
    tot_audio = float(dur_arr.sum())
    tot_infer = float(lat_arr.sum()) / 1000.0
    n         = len(all_preds)

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
        "peak_ram_mb":                 peak_ram_mb,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TENSORBOARD MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class TBWriterManager:
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
        comp   = results.get("compression", {})
        ms, mode     = info["model_size"], info["mode"]
        tpf, tf      = info["tokens_per_frame"], info["total_frames"]
        gpu_tag      = info.get("gpu_tag", "unknown")
        quant        = comp.get("quantization", "fp16")
        prune        = comp.get("pruning_ratio", 0.0)

        w = self._get_writer(ms, mode)
        if w is None:
            return

        x = tf  # primary X-axis

        for split_name, sr in splits.items():
            pfx = f"{split_name}/tpf{tpf}/{gpu_tag}/{quant}/p{int(prune*100)}"
            for key in [
                "wer_pct", "rtf_overall", "rtf_mean_per_sample",
                "latency_mean_ms", "latency_p50_ms", "latency_p95_ms",
                "latency_p99_ms", "throughput_samples_per_sec",
                "throughput_audio_hrs_per_hr", "vram_after_gb",
                "avg_output_tokens", "peak_ram_mb",
            ]:
                if key in sr:
                    w.add_scalar(f"{pfx}/{key}", sr[key], x)

        for sub in ("total_G", "encoder_G", "decoder_G",
                    "total_eff_G", "encoder_eff_G", "decoder_eff_G"):
            key = f"flops_{sub}"
            if key in flops:
                w.add_scalar(f"flops/tpf{tpf}/{gpu_tag}/{quant}/{sub}", flops[key], x)

        for sub in ("total", "encoder", "decoder"):
            w.add_scalar(f"model/{sub}_params_M", params[f"{sub}_params_M"], x)

        if "model_size_mb" in comp:
            w.add_scalar(f"model/size_mb/{quant}", comp["model_size_mb"], x)

        primary = splits.get("test_other", splits.get("test", {}))
        w.add_hparams(
            hparam_dict={
                "model_size": ms, "mode": mode,
                "tokens_per_frame": tpf, "total_frames": tf,
                "fp16": int(info.get("fp16", True)),
                "quantization": quant,
                "pruning_ratio": prune,
                "num_beams": info.get("num_beams", 1),
                "gpu_tag": gpu_tag,
            },
            metric_dict={
                "hparam/wer_pct":                     primary.get("wer_pct", -1),
                "hparam/rtf_overall":                 primary.get("rtf_overall", -1),
                "hparam/latency_p50_ms":              primary.get("latency_p50_ms", -1),
                "hparam/throughput_audio_hrs_per_hr": primary.get("throughput_audio_hrs_per_hr", -1),
                "hparam/flops_total_G":               flops["flops_total_G"],
                "hparam/flops_total_eff_G":           flops.get("flops_total_eff_G", -1),
                "hparam/model_size_mb":               comp.get("model_size_mb", -1),
                "hparam/peak_ram_mb":                 primary.get("peak_ram_mb", -1),
            },
        )
        w.flush()
        print(f"  [TB] Logged {ms}/{mode} quant={quant} prune={prune} "
              f"frames={x} tpf={tpf} gpu={gpu_tag}")

    def close_all(self):
        for w in self._writers.values():
            w.close()
        self._writers.clear()


# ─────────────────────────────────────────────────────────────────────────────
# FULL EVALUATION RUN
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    model_size:       str,
    mode:             str,
    lorar:            int,
    checkpoint:       Optional[str],
    benchmark:        str,
    total_frames:     int,
    tokens_per_frame: int,
    max_eval_samples: Optional[int],
    fp16:             bool,
    num_beams:        int,
    output_dir:       str,
    quantization:     str   = "fp16",
    pruning_ratio:    float = 0.0,
    training_dtype:   str   = "fp16",
    gpu_tag:          str   = "unknown",
    tb_manager:       Optional[TBWriterManager] = None,
) -> Dict:
    """
    Full evaluation pipeline:
      1. Load model (base + LoRA merge if applicable)
      2. Apply pruning (xP)
      3. Apply quantization (xQ)
      4. Measure model size
      5. Run inference on eval splits
      6. Compute FLOPs (precision-scaled)
      7. Save JSON + TensorBoard
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*72}")
    print(f"  whisper-{model_size} | {mode} | {lorar} | tf={total_frames} | "
          f"tpf={tokens_per_frame} | quant={quantization} | "
          f"prune={pruning_ratio:.0%} | tag={gpu_tag}")
    print(f"  Device: {device}"
          + (f" ({torch.cuda.get_device_name(device)})"
             if device.type == "cuda" else ""))
    print(f"{'='*72}")

    gpu_info = get_gpu_info()
    gpu_dict = asdict(gpu_info) if gpu_info else {"name": "CPU", "available": False}

    # ── 1. Load ───────────────────────────────────────────────────────────────
    t0 = time.time()
    model, processor = load_model_for_inference(
        model_size=model_size, mode=mode, checkpoint=checkpoint,
        device=device, tokens_per_frame=tokens_per_frame, fp16=fp16,
    )
    load_s = round(time.time() - t0, 2)

    # ── 2. Pruning (xP) ───────────────────────────────────────────────────────
    if pruning_ratio > 0.0:
        model = apply_pruning(model, pruning_ratio)

    # ── 3. Quantization (xQ) ─────────────────────────────────────────────────
    model, device = apply_quantization(
        model=model,
        quantization=quantization,
        training_dtype=training_dtype,
        device=device,
        model_size=model_size,
        checkpoint=checkpoint,
    )

    # ── 4. Model size measurement ─────────────────────────────────────────────
    model_size_mb = measure_model_size_mb(model, quantization)
    print(f"  Model size in memory: {model_size_mb} MB")

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    eval_splits   = load_eval_splits(benchmark, max_eval_samples)
    split_results = {}

    for split_name, split_info in eval_splits.items():
        print(f"\n  Evaluating: {split_name}")
        res = evaluate_split(
            model, processor,
            dataset=split_info["dataset"], text_col=split_info["text_col"],
            device=device, total_frames=total_frames,
            tokens_per_frame=tokens_per_frame,
            fp16=fp16, num_beams=num_beams,
            quantization=quantization,
        )
        split_results[split_name] = res
        print(f"  WER={res['wer_pct']:.3f}%  RTF={res['rtf_overall']:.4f}  "
              f"p50={res['latency_p50_ms']:.1f}ms  "
              f"Peak RAM={res['peak_ram_mb']:.0f}MB  "
              f"tput={res['throughput_audio_hrs_per_hr']:.1f}×RT")

    # ── 6. FLOPs (precision-scaled) ───────────────────────────────────────────
    actual_tok = int(round(
        np.mean([r["avg_output_tokens"] for r in split_results.values()])
    ))
    flops = estimate_flops(
        model_size, total_frames, tokens_per_frame,
        quantization=quantization,
        pruning_ratio=pruning_ratio,
        avg_output_tokens=actual_tok,
    )

    # ── 7. Assemble result dict ───────────────────────────────────────────────
    inner        = model.model if isinstance(model, WhisperWithTokenSubsampling) else model
    total_params = sum(p.numel() for p in inner.parameters())
    enc_p        = WHISPER_PARAMS.get(model_size, (total_params, 0, 0))[1]
    dec_p        = WHISPER_PARAMS.get(model_size, (total_params, 0, 0))[2]

    results = {
        "run_info": {
            "timestamp":          datetime.datetime.now().isoformat(),
            "model_size":         model_size,
            "mode":               mode,
            "lorar":              lorar,
            "checkpoint":         checkpoint,
            "gpu_tag":            gpu_tag,
            "benchmark":          benchmark,
            "total_frames":       total_frames,
            "tokens_per_frame":   tokens_per_frame,
            "audio_duration_s":   round((total_frames / WHISPER_ENC_FRAMES) * WHISPER_MAX_DUR, 1),
            "fp16":               fp16,
            "num_beams":          num_beams,
            "max_eval_samples":   max_eval_samples,
            "model_load_time_s":  load_s,
            "python_version":     platform.python_version(),
            "torch_version":      torch.__version__,
            "torchaudio_version": torchaudio.__version__,
        },
        # NEW: compression axis summary for this run
        "compression": {
            "quantization":   quantization,
            "pruning_ratio":  pruning_ratio,
            "training_dtype": training_dtype,
            "model_size_mb":  model_size_mb,
            "bytes_per_param_theoretical":
                PRECISION_BYTES.get(quantization, 2),
        },
        "gpu":    gpu_dict,
        "model_params": {
            "total_params":     total_params,
            "total_params_M":   round(total_params / 1e6, 1),
            "encoder_params":   enc_p,
            "encoder_params_M": round(enc_p / 1e6, 1),
            "decoder_params":   dec_p,
            "decoder_params_M": round(dec_p / 1e6, 1),
        },
        "flops":  flops,
        "splits": split_results,
    }

    # ── Save JSON ─────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    fname = (
        f"eval_{model_size}_{mode}_{lorar}"
        f"_tf{total_frames}_tpf{tokens_per_frame}"
        f"_{quantization}_p{int(pruning_ratio*100)}"
        f"_{gpu_tag}_{benchmark}.json"
    )
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
    W = 135
    print(f"\n{'='*W}\n  EVALUATION SUMMARY\n{'='*W}")
    print(
        f"{'Model':<14} {'Mode':<8} {'Frames':>7} {'TpF':>4} "
        f"{'Quant':<6} {'Prune':>6} "
        f"{'WER%':>8} {'RTF':>7} "
        f"{'p50ms':>7} {'FLOPs-G':>8} {'EffFLOPs':>9} "
        f"{'SizeMB':>7} {'PeakRAM':>8} {'GPU':<20}"
    )
    print("-" * W)
    for r in all_results:
        info  = r["run_info"]
        flops = r["flops"]
        comp  = r.get("compression", {})
        splits = r["splits"]
        gpu_hw = r["gpu"].get("name", "CPU")[:20]
        primary = splits.get("test_other", splits.get("test", {}))

        wer    = primary.get("wer_pct",       "-")
        rtf    = primary.get("rtf_overall",   "-")
        p50    = primary.get("latency_p50_ms","-")
        ram_mb = primary.get("peak_ram_mb",   "-")

        def _f(v, fmt):
            return fmt.format(v) if isinstance(v, (int, float)) else str(v)

        print(
            f"  {info['model_size']:<12} {info['mode']:<8} "
            f"{info['total_frames']:>7} {info['tokens_per_frame']:>4} "
            f"{comp.get('quantization','?'):<6} {comp.get('pruning_ratio',0):>5.0%} "
            f"{_f(wer, '{:.3f}'):>8} {_f(rtf, '{:.4f}'):>7} "
            f"{_f(p50, '{:.1f}'):>7} "
            f"{flops['flops_total_G']:>8.1f} "
            f"{flops.get('flops_total_eff_G', flops['flops_total_G']):>9.1f} "
            f"{comp.get('model_size_mb', '-'):>7} "
            f"{_f(ram_mb, '{:.0f}'):>8} "
            f"{gpu_hw:<20}"
        )
    print("=" * W)
    print(
        "  TpF=tokens_per_frame | RTF↓=faster | FLOPs-G=raw GigaFLOPs | "
        "EffFLOPs=precision-scaled | SizeMB=in-memory | PeakRAM=GPU peak MB"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Whisper inference eval — quantization + pruning axes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    cg = p.add_argument_group("Checkpoint strategy")
    cg.add_argument("--checkpoint_root", type=str, default=None)
    cg.add_argument("--checkpoint",  type=str, default=None)
    cg.add_argument("--model_size",  type=str, default=None,
                    choices=list(LOCAL_PATH.keys()))
    cg.add_argument("--mode", type=str, default=None,
                    choices=["baseline", "lora", "full"])

    fg = p.add_argument_group("Discovery filters")
    fg.add_argument("--filter_model_size", type=str, default=None)
    fg.add_argument("--filter_mode",       type=str, default=None)

    ag = p.add_argument_group("Compute axes / sweeps")
    ag.add_argument("--tokens_per_frame",       type=int,   default=None)
    ag.add_argument("--total_frames",            type=int,   default=None)
    ag.add_argument("--sweep_tokens_per_frame",  type=str,   default=None)
    ag.add_argument("--sweep_total_frames",      type=str,   default=None)

    # ── NEW: quantization axis ────────────────────────────────────────────────
    qg = p.add_argument_group("Quantization axis (xQ)")
    qg.add_argument("--quantization", type=str, default="fp16",
                    choices=VALID_QUANTIZATIONS,
                    help=(
                        "Post-training quantization precision.\n"
                        "  fp16 → model in fp16 (default, no conversion if already fp16)\n"
                        "  int8 → torch.quantization.quantize_dynamic (CPU)\n"
                        "  int4 → bitsandbytes NF4 (GPU only, requires bitsandbytes>=0.41)"
                    ))
    qg.add_argument("--sweep_quantization", type=str, default=None,
                    help="Comma-separated quantization levels to sweep, e.g. fp16,int8,int4")
    qg.add_argument("--training_dtype", type=str, default="fp16",
                    choices=["fp16", "fp32"],
                    help=(
                        "Dtype the model was trained in.\n"
                        "  fp16 → standard (default); most finetuned checkpoints\n"
                        "  fp32 → triggers fp16 + int8 + int4 cascade if "
                        "        --sweep_quantization not specified"
                    ))

    # ── NEW: pruning axis ─────────────────────────────────────────────────────
    pg = p.add_argument_group("Pruning axis (xP)")
    pg.add_argument("--pruning_ratio", type=float, default=0.0,
                    help=(
                        "Structured attention-head pruning ratio (0.0–0.4).\n"
                        "  0.0  → dense, no pruning (default)\n"
                        "  0.1  → remove 10%% of heads per encoder layer\n"
                        "  0.3  → aggressive; use pruned+recovered checkpoint for best WER\n"
                        "NOTE: inference_eval applies pruning WITHOUT recovery finetuning.\n"
                        "      For recovered results, prune via finetune.py first."
                    ))
    pg.add_argument("--sweep_pruning_ratio", type=str, default=None,
                    help="Comma-separated pruning ratios, e.g. 0.0,0.1,0.2,0.3")

    p.add_argument("--benchmark",        type=str, default="librispeech",
                   choices=["librispeech", "common_voice", "fleurs"])
    p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--fp16",             action="store_true", default=True)
    p.add_argument("--num_beams",        type=int, default=1)
    p.add_argument("--output_dir",       type=str,
                   default="/nfshomes/vyomwal5/SAME/eval_results")
    p.add_argument("--no_tensorboard",   action="store_true", default=False)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Hardware banner ───────────────────────────────────────────────────────
    print("\n" + "="*72 + "\n  HARDWARE INFO\n" + "="*72)
    gpu = get_gpu_info()
    if gpu:
        print(f"  GPU              : {gpu.name}")
        print(f"  VRAM total/free  : {gpu.total_vram_gb:.1f} / {gpu.free_vram_gb:.1f} GB")
        print(f"  Compute cap.     : {gpu.compute_capability}  SMs={gpu.sm_count}")
        print(f"  Driver / CUDA    : {gpu.driver_version} / {gpu.cuda_version}")
    else:
        print("  Running on CPU (no CUDA GPU detected)")
        if args.quantization == "int4":
            print("  WARNING: INT4 requires a CUDA GPU. Falling back to int8.")
            args.quantization = "int8"
    print("="*72)

    # ── TensorBoard ───────────────────────────────────────────────────────────
    tb_root    = os.path.join(args.output_dir, "tensorboard")
    tb_manager = None
    if not args.no_tensorboard and TENSORBOARD_AVAILABLE:
        tb_manager = TBWriterManager(tb_root)
        print(f"\n  TensorBoard → {tb_root}")
        print(f"  Launch: tensorboard --logdir {tb_root}\n")

    # ── Build quantization list ───────────────────────────────────────────────
    if args.sweep_quantization:
        quant_list = [q.strip() for q in args.sweep_quantization.split(",")]
    elif args.training_dtype == "fp32" and not args.sweep_quantization:
        # FP32-trained model: automatically cascade to all precisions
        print("  [xQ] FP32-trained model detected — auto-cascading: fp16 → int8 → int4")
        quant_list = ["fp16", "int8", "int4"]
    else:
        quant_list = [args.quantization]

    # ── Build pruning ratio list ──────────────────────────────────────────────
    if args.sweep_pruning_ratio:
        prune_list = [float(p.strip()) for p in args.sweep_pruning_ratio.split(",")]
    else:
        prune_list = [args.pruning_ratio]

    # ── Build (tf, tpf) axis list ─────────────────────────────────────────────
    @dataclass
    class EvalJob:
        model_size:       str
        mode:             str
        lorar:            int
        checkpoint:       Optional[str]
        tokens_per_frame: int
        total_frames:     int
        quantization:     str
        pruning_ratio:    float
        training_dtype:   str
        gpu_tag:          str = "runtime"

    base_jobs: List[EvalJob] = []

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
            base_jobs.append(EvalJob(
                model_size=spec.model_size, mode=spec.mode, lorar=spec.lorar,
                checkpoint=spec.path, tokens_per_frame=tpf,
                total_frames=tf, gpu_tag=spec.gpu_tag,
                quantization="fp16",   # filled in outer loop
                pruning_ratio=0.0,
                training_dtype=args.training_dtype,
            ))
    else:
        assert args.model_size, "--model_size required without --checkpoint_root"
        assert args.mode,       "--mode required without --checkpoint_root"

        saved    = _read_experiment_cfg(args.checkpoint) if args.checkpoint else {}
        base_tpf = args.tokens_per_frame if args.tokens_per_frame is not None \
                   else saved.get("tokens_per_frame", 1)
        base_tf  = args.total_frames if args.total_frames is not None \
                   else saved.get("total_frames", 1500)

        axis_configs: List[Tuple[int, int]] = []
        if args.sweep_tokens_per_frame:
            for tpf in [int(x) for x in args.sweep_tokens_per_frame.split(",")]:
                axis_configs.append((base_tf, tpf))
        elif args.sweep_total_frames:
            for tf in [int(x) for x in args.sweep_total_frames.split(",")]:
                axis_configs.append((tf, base_tpf))
        else:
            axis_configs.append((base_tf, base_tpf))

        for tf, tpf in axis_configs:
            base_jobs.append(EvalJob(
                model_size=args.model_size, mode=args.mode,
                checkpoint=args.checkpoint, tokens_per_frame=tpf,
                total_frames=tf, quantization="fp16", pruning_ratio=0.0,
                training_dtype=args.training_dtype,
            ))

    # ── Expand jobs across quantization × pruning ─────────────────────────────
    eval_jobs: List[EvalJob] = []
    for job in base_jobs:
        for quant in quant_list:
            for prune in prune_list:
                eval_jobs.append(EvalJob(
                    model_size=job.model_size, mode=job.mode,
                    checkpoint=job.checkpoint,
                    lorar = job.lorar,
                    tokens_per_frame=job.tokens_per_frame,
                    total_frames=job.total_frames,
                    quantization=quant,
                    pruning_ratio=prune,
                    training_dtype=job.training_dtype,
                    gpu_tag=job.gpu_tag,
                ))

    print(f"\n  Jobs queued: {len(eval_jobs)}")
    for j in eval_jobs:
        dur = round((j.total_frames / WHISPER_ENC_FRAMES) * WHISPER_MAX_DUR, 1)
        print(f"    {j.model_size:<12} {j.mode:<8} lora_r={j.lorar} tf={j.total_frames}({dur}s) "
              f"tpf={j.tokens_per_frame}  quant={j.quantization:<5}  "
              f"prune={j.pruning_ratio:.0%}  tag={j.gpu_tag}")

    # ── Run all jobs ──────────────────────────────────────────────────────────
    all_results = []
    for job in eval_jobs:
        result = run_evaluation(
            model_size=job.model_size,
            mode=job.mode,
            lorar = job.lorar,
            checkpoint=job.checkpoint,
            benchmark=args.benchmark,
            total_frames=job.total_frames,
            tokens_per_frame=job.tokens_per_frame,
            max_eval_samples=args.max_eval_samples,
            fp16=args.fp16,
            num_beams=args.num_beams,
            output_dir=args.output_dir,
            quantization=job.quantization,
            pruning_ratio=job.pruning_ratio,
            training_dtype=job.training_dtype,
            gpu_tag=job.gpu_tag,
            tb_manager=tb_manager,
        )
        all_results.append(result)

    print_summary(all_results)

    if len(all_results) > 1:
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(
            args.output_dir,
            f"sweep_{args.benchmark}_{ts}.json"
        )
        with open(path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Combined sweep JSON → {path}")

    if tb_manager is not None:
        tb_manager.close_all()
        print(f"\n  tensorboard --logdir {tb_root}")


if __name__ == "__main__":
    main()