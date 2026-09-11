"""
inference_quantized.py — REAL post-training quantization + evaluation
=======================================================================
Companion to inference2.py, for a gap that script never actually closes:
--qat_mode int4 in finetune.py trains weights to be ROBUST to 4-bit
rounding noise (real, genuine effect on the saved float weights -- see
apply_qat()'s docstring), but never converts anything to actual 4-bit
storage -- there's no native torch int4 tensor type, so
finalize_and_convert()'s int4 branch just strips fake-quant back to
plain float and returns converted=False. Every "INT4" checkpoint this
project has produced so far is, on disk, an ordinary FP32 float
checkpoint. inference2.py evaluates it as exactly that.

This script is the deliberately separate, explicit step that actually
applies real 4-bit quantization (bitsandbytes NF4) on top of a trained
checkpoint, so you can measure:
  (a) genuine INT4 deployment WER/size/RTF/FLOPs for the first time, and
  (b) whether QAT-INT4 training measurably helps that deployment
      outcome compared to quantizing a checkpoint that never saw
      QAT-INT4 fake-quant noise during training at all (the actual
      open question this project has been circling).

IMPORTANT CAVEAT, stated once and left un-repeated below: bitsandbytes
NF4 is a NON-LINEAR quantization scheme (levels concentrated near zero,
tuned for roughly-normal weight distributions) -- it is NOT the same
encoding as the LINEAR, symmetric 4-bit levels apply_qat()'s
_Int4FakeQuantize simulates during training. Training a model to be
robust to linear 4-bit rounding does not guarantee it is specifically
well-suited to NF4's non-uniform grid. This script measures whether it
helps in practice; it does not assume the answer.

Runs on GPU (bitsandbytes NF4 requires CUDA -- unlike inference2.py's
quantize_dynamic() INT8 path, which is CPU-only).

Usage:
    python inference_quantized.py \\
        --checkpoint /path/to/checkpoint --model_size large-v3 --mode full \\
        --benchmark librispeech --num_beams 16

    # Compare against a checkpoint that never saw QAT-INT4 training,
    # to directly test whether QAT training helped:
    python inference_quantized.py \\
        --checkpoint /path/to/dense_no_qat_checkpoint --model_size large-v3 --mode full \\
        --benchmark librispeech --num_beams 16

    # Sweep every int4-relevant checkpoint under a directory:
    python inference_quantized.py --checkpoint_root /path/to/checkpoints --num_beams 16
"""

from transformers.utils import logging
logging.set_verbosity_error()
import os

CACHE_DIR = "/fs/nexus-scratch/vyomwal5/anaconda3/envs/whisper/hf_cache"
os.environ["HF_HOME"]               = CACHE_DIR
os.environ["HF_DATASETS_CACHE"]     = f"{CACHE_DIR}/datasets"
os.environ["TRANSFORMERS_CACHE"]    = f"{CACHE_DIR}/models"
os.environ["HUGGINGFACE_HUB_CACHE"] = f"{CACHE_DIR}/hub"
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"

import re
import json
import time
import tempfile
import datetime
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    WhisperForConditionalGeneration, WhisperProcessor, BitsAndBytesConfig,
)
from peft import PeftModel
from jiwer import wer as jiwer_wer
import tqdm

LOCAL_PATH = {
    "small":    "/fs/nexus-scratch/vyomwal5/models/models/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d",
    "medium":   "/fs/nexus-scratch/vyomwal5/models/models/models--openai--whisper-medium/snapshots/abdf7c39ab9d0397620ccaea8974cc764cd0953e",
    "tiny":     "/fs/nexus-scratch/vyomwal5/models/models/models--openai--whisper-tiny/snapshots/169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
    "large-v3": "/fs/nexus-scratch/vyomwal5/models/models/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1",
    "distil-small":  "/fs/nexus-scratch/vyomwal5/models/distil-small",
    "distil-medium": "/fs/nexus-scratch/vyomwal5/models/distil-medium",
    "distil-large":  "/fs/nexus-scratch/vyomwal5/models/distil-large",
}
WHISPER_ENC_FRAMES = 1500
WHISPER_SR          = 16_000
WHISPER_MAX_DUR      = 30.0

WHISPER_PARAMS = {
    "tiny": 39_000_000, "small": 244_000_000, "medium": 769_000_000,
    "large-v3": 1_540_000_000, "distil-small": 166_000_000,
    "distil-medium": 394_000_000, "distil-large": 756_000_000,
}

# ── kept in sync with inference2.py's copies ─────────────────────────────────
WHISPER_D_MODEL = {
    "tiny": 384, "base": 512, "small": 768, "medium": 1024, "large-v3": 1280,
    "distil-small": 768, "distil-medium": 1024, "distil-large": 1280,
}
WHISPER_LAYERS = {
    "tiny":          (4,  4),  "base":          (6,  6),
    "small":         (12, 12), "medium":         (24, 24),
    "large-v3":      (32, 32), "distil-small":   (12,  4),
    "distil-medium": (24,  2), "distil-large":   (32,  2),
}
PRECISION_FLOP_SCALE = {
    "fp32": 1.0, "fp16": 0.5, "int8": 0.25, "int4": 0.125,
}
SPARSITY_FLOP_SCALE = {
    "dense": 1.0, "2:4": 0.5, "1:4": 0.5,
}


# ─────────────────────────────────────────────────────────────────────────────
# WhisperWithTokenSubsampling  (kept in sync with finetune.py / inference2.py)
# ─────────────────────────────────────────────────────────────────────────────

class WhisperWithTokenSubsampling(nn.Module):
    def __init__(self, base_model, tokens_per_frame: int = 1):
        super().__init__()
        self.model = base_model
        self.tokens_per_frame = tokens_per_frame

    def _get_encoder(self):
        m = self.model
        if isinstance(m, PeftModel):
            m = m.base_model.model
        whisper_model = getattr(m, "model", m)
        return getattr(whisper_model, "encoder")

    def _subsample(self, hidden):
        if self.tokens_per_frame > 1:
            hidden = hidden[:, :: self.tokens_per_frame, :]
        return hidden

    def _encode_and_subsample(self, input_features):
        encoder_out = self._get_encoder()(input_features)
        encoder_out.last_hidden_state = self._subsample(encoder_out.last_hidden_state)
        return encoder_out

    def forward(self, input_features, labels=None, **kwargs):
        encoder_out = self._encode_and_subsample(input_features)
        return self.model(encoder_outputs=encoder_out, labels=labels, **kwargs)

    def generate(self, input_features, **kwargs):
        encoder_out = self._encode_and_subsample(input_features)
        return self.model.generate(encoder_outputs=encoder_out, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def _prune_base_nm(model: nn.Module, sparsity_pattern: str) -> nn.Module:
    """Same deterministic magnitude-based reconstruction used throughout
    this project (inference2.py's copy) -- needed for a LoRA checkpoint
    whose base was pruned, since only the adapter is ever saved."""
    if sparsity_pattern == "dense":
        return model
    n, m = (int(x) for x in sparsity_pattern.split(":"))
    output_head = None
    if hasattr(model, "get_output_embeddings"):
        try:
            output_head = model.get_output_embeddings()
        except Exception:
            output_head = None
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear) or "lora_" in name:
                continue
            if module is output_head or name.endswith("proj_out"):
                continue
            w = module.weight.data
            rows, cols = w.shape
            if rows < m or cols < m or rows % m != 0 or cols % m != 0:
                continue
            w_blocks = w.reshape(-1, m)
            _, sorted_idx = torch.sort(w_blocks.abs(), dim=1)
            zero_idx = sorted_idx[:, : m - n]
            mask = torch.ones_like(w_blocks, dtype=torch.bool)
            mask.scatter_(1, zero_idx, False)
            module.weight.data = (w_blocks * mask.to(w.dtype)).reshape(w.shape)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT LOADING (float, then real NF4 quantization on top)
# ─────────────────────────────────────────────────────────────────────────────

def load_and_merge_float_model(
    checkpoint_dir: str, model_size: str, mode: str, saved_cfg: Dict,
) -> nn.Module:
    """
    Load the checkpoint at whatever precision it was actually saved at
    (always fp32 -- see this project's dtype fix) and, for mode=lora,
    reconstruct the sparse base + merge the adapter into a single dense
    model. NF4 quantization needs one flat model to serialize and
    reload -- it cannot operate on an unmerged PeftModel the way
    inference2.py's INT8 dynamic-quantization path can (that path
    deliberately keeps base+adapter separate; bitsandbytes' NF4 doesn't
    offer an equivalent unmerged story here, so we merge first).
    """
    model_name = LOCAL_PATH[model_size]
    sparsity_pattern = saved_cfg.get("sparsity_pattern", "dense")

    if mode == "lora":
        base = WhisperForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float32,
        )
        base.config.forced_decoder_ids         = None
        base.generation_config.suppress_tokens = []
        base.config.use_cache                  = True
        if sparsity_pattern != "dense":
            base = _prune_base_nm(base, sparsity_pattern)
        print(f"  Merging LoRA adapter from {checkpoint_dir} ...")
        peft_model = PeftModel.from_pretrained(base, checkpoint_dir, is_trainable=False)
        model = peft_model.merge_and_unload()
        if sparsity_pattern != "dense":
            print("  (sparse base + dense LoRA delta merged — result is "
                  "intentionally NOT strictly N:M anymore)")
    elif mode == "full":
        model = WhisperForConditionalGeneration.from_pretrained(
            checkpoint_dir, torch_dtype=torch.float32,
        )
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    model.config.forced_decoder_ids         = None
    model.generation_config.suppress_tokens = []
    model.config.use_cache                  = True
    return model


def apply_real_nf4_quantization(model: nn.Module, device: torch.device) -> nn.Module:
    """
    Genuine bitsandbytes NF4 quantization (GPU-only). Serializes the
    merged fine-tuned model to a temp dir and reloads it with
    BitsAndBytesConfig -- reusing the exact fix from inference2.py's
    apply_quantization(): the naive "quantize a fresh pretrained base,
    then copy fine-tuned weights over" approach silently drops every
    quantized layer (Params4bit's packed shape never matches the fp
    source), producing a Frankenstein model wearing only fine-tuned
    embeddings. Quantizing the ACTUAL fine-tuned weights directly avoids
    that entirely.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    with tempfile.TemporaryDirectory(prefix="nf4_quant_") as tmpdir:
        print(f"  Serializing merged model to {tmpdir} for NF4 reload ...")
        model.cpu().save_pretrained(tmpdir, safe_serialization=True)
        print("  Reloading with real NF4 quantization ...")
        quantized = WhisperForConditionalGeneration.from_pretrained(
            tmpdir, quantization_config=bnb_config,
            # torch_dtype=float16 matters here: bitsandbytes only converts
            # nn.Linear layers to 4-bit -- everything else (conv1/conv2,
            # layernorms, embeddings) loads at whatever dtype is requested
            # here, defaulting to fp32 if omitted. run_eval() feeds fp16
            # input (matching bnb_4bit_compute_dtype above); leaving this
            # unset meant conv1's fp32 bias met an fp16 input directly,
            # crashing with "Input type (c10::Half) and bias type (float)
            # should be the same" -- the exact same class of mismatch
            # already fixed once in finetune.py's WhisperDataCollator, now
            # recurring here since this script builds its own load path.
            torch_dtype=torch.float16,
            device_map="auto", low_cpu_mem_usage=True,
        )
    quantized.config.forced_decoder_ids         = None
    quantized.generation_config.suppress_tokens = []
    quantized.config.use_cache                  = True
    quantized.eval()
    return quantized


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY MEASUREMENT — corrected for bitsandbytes' packed uint8 storage
# ─────────────────────────────────────────────────────────────────────────────

def measure_model_size_mb(model: nn.Module) -> float:
    """
    bitsandbytes Params4bit tensors report dtype=torch.uint8 and pack TWO
    real 4-bit values into each uint8 byte -- the packed tensor's
    nelement() already reflects the compressed size, so each element is
    genuinely 1 physical byte, not 0.5. A naive dtype-unaware byte count
    (assuming "small dtype = fewer bytes than reported" and applying an
    extra 0.5x on top of an already-packed uint8) double-applies the
    compression and undercounts real size by 2x -- the same class of bug
    fixed in inference2.py's measure_model_size_mb, now specifically for
    Params4bit rather than quantize_dynamic's packed tuples.
    """
    total_bytes = 0.0
    for p in model.parameters():
        if p.dtype == torch.float32:
            total_bytes += p.nelement() * 4
        elif p.dtype in (torch.float16, torch.bfloat16):
            total_bytes += p.nelement() * 2
        elif p.dtype in (torch.int8, torch.uint8, torch.qint8, torch.quint8):
            total_bytes += p.nelement() * 1     # packed 4-bit -- 1 real byte/element
        else:
            total_bytes += p.nelement() * p.element_size()
    for b in model.buffers():
        if b.dtype == torch.float32:
            total_bytes += b.nelement() * 4
        elif b.dtype in (torch.float16, torch.bfloat16):
            total_bytes += b.nelement() * 2
        elif b.dtype in (torch.int8, torch.uint8, torch.qint8, torch.quint8):
            total_bytes += b.nelement() * 1
        else:
            total_bytes += b.nelement() * b.element_size()
    return round(total_bytes / (1024 ** 2), 2)


# ─────────────────────────────────────────────────────────────────────────────
# FLOP ESTIMATION — precision-aware + sparsity-aware
# ─────────────────────────────────────────────────────────────────────────────
# Same formula as inference2.py's estimate_flops(), copied rather than
# imported (these two files deliberately don't import from each other --
# see _quantize_dynamic_matching_skeleton's docstring in inference2.py for
# why that's a real, acknowledged risk: keeping formulas in sync across
# files is a manual responsibility). ONE deliberate difference from
# inference2.py's version: inference2.py hardcodes quantization="fp16"
# unconditionally for FLOPs accounting regardless of what actually ran --
# a real, separately-flagged bug there. This script has no such bug to
# begin with, because it only ever runs ONE precision (genuine NF4) --
# callers below always pass quantization="int4", which is simply true.

def estimate_flops(
    model_size: str, total_frames: int, tokens_per_frame: int,
    quantization: str = "int4", sparsity_pattern: str = "dense",
    avg_output_tokens: int = 50,
) -> Dict:
    d            = WHISPER_D_MODEL.get(model_size, 1024)
    n_enc, n_dec = WHISPER_LAYERS.get(model_size, (24, 24))
    T            = total_frames
    T_dec        = T // tokens_per_frame
    L            = avg_output_tokens

    enc_attn = n_enc * (4 * T * d * d + 2 * T * T * d)
    enc_ffn  = n_enc * 8 * T * d * d
    enc      = enc_attn + enc_ffn
    dec      = n_dec * (2 * T_dec * d * d + 2 * d * d + 8 * d * d) * L
    tot      = enc + dec

    prec_scale     = PRECISION_FLOP_SCALE.get(quantization, 1.0)
    sparse_scale   = SPARSITY_FLOP_SCALE.get(sparsity_pattern, 1.0)
    combined_scale = prec_scale * sparse_scale

    return {
        "flops_encoder_G":      round(enc / 1e9, 3),
        "flops_decoder_G":      round(dec / 1e9, 3),
        "flops_total_G":        round(tot / 1e9, 3),
        "flops_encoder_eff_G":  round(enc * combined_scale / 1e9, 3),
        "flops_decoder_eff_G":  round(dec * combined_scale / 1e9, 3),
        "flops_total_eff_G":    round(tot * combined_scale / 1e9, 3),
        "precision_flop_scale": prec_scale,
        "sparsity_flop_scale":  sparse_scale,
        "combined_flop_scale":  combined_scale,
        "sparsity_pattern":     sparsity_pattern,
        "encoder_frames":       T,
        "decoder_cross_attn_len": T_dec,
        "avg_output_tokens":    L,
        "quantization":         quantization,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEXT NORMALIZATION + EVAL LOOP
# ─────────────────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s\']", "", text)
    return re.sub(r"\s+", " ", text)


def run_eval(model, processor, dataset, text_col, device, total_frames,
             tokens_per_frame, num_beams, max_new_tokens=256):
    all_preds, all_refs, latencies_ms, audio_durs = [], [], [], []
    output_tok_lens = []
    inner = model.model if isinstance(model, WhisperWithTokenSubsampling) else model
    is_multilingual = getattr(inner.config, "is_multilingual", False)
    gen_kwargs = {"num_beams": num_beams, "max_new_tokens": max_new_tokens}
    if is_multilingual:
        gen_kwargs["language"] = "en"
        gen_kwargs["task"]     = "transcribe"

    max_samples = int((total_frames / WHISPER_ENC_FRAMES) * WHISPER_MAX_DUR * WHISPER_SR)

    print("  Running inference (NF4-quantized model) ...")
    for sample in tqdm.tqdm(dataset):
        audio_array = np.array(sample["audio"]["array"], dtype=np.float32)[:max_samples]
        ref_text    = sample[text_col]
        dur_s       = len(audio_array) / WHISPER_SR

        features = processor.feature_extractor(
            audio_array, sampling_rate=WHISPER_SR,
        ).input_features[0]
        feat = torch.tensor(features).unsqueeze(0).to(device).half()

        t0 = time.perf_counter()
        with torch.no_grad():
            pred_ids = model.generate(feat, **gen_kwargs)
        lat_ms = (time.perf_counter() - t0) * 1000.0

        raw_ids = pred_ids[0].clamp(min=0, max=processor.tokenizer.vocab_size - 1)
        pred_text = processor.tokenizer.decode(raw_ids, skip_special_tokens=True)
        all_preds.append(normalize_text(pred_text))
        all_refs.append(normalize_text(ref_text))
        latencies_ms.append(lat_ms)
        audio_durs.append(dur_s)
        # NEW: needed both to compute real FLOPs (decoder FLOPs scale
        # directly with output length in estimate_flops' formula) and to
        # diagnose whether a high-WER run is genuine transcription failure
        # or degenerate/repetitive generation hitting max_new_tokens --
        # previously discarded, meaning neither question was answerable
        # from this script's output alone.
        output_tok_lens.append(pred_ids.shape[1])

    wer_val   = jiwer_wer(all_refs, all_preds)
    lat_arr   = np.array(latencies_ms)
    dur_arr   = np.array(audio_durs)
    tot_audio = float(dur_arr.sum())
    tot_infer = float(lat_arr.sum()) / 1000.0

    return {
        "wer_pct":           round(100 * wer_val, 4),
        "n_samples":         len(all_preds),
        "total_audio_h":     round(tot_audio / 3600, 4),
        "rtf_overall":       round(tot_infer / max(tot_audio, 1e-6), 5),
        "latency_mean_ms":   round(float(lat_arr.mean()), 2),
        "latency_p50_ms":    round(float(np.percentile(lat_arr, 50)), 2),
        "avg_output_tokens": round(float(np.mean(output_tok_lens)), 1),
        "max_output_tokens": int(np.max(output_tok_lens)),
        "hit_max_new_tokens_pct": round(
            100 * float(np.mean(np.array(output_tok_lens) >= max_new_tokens)), 2
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI / MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _read_experiment_cfg(ckpt_path: str) -> Dict:
    cfg_path = os.path.join(ckpt_path, "experiment_cfg.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT DISCOVERY (mirrors inference2.py's discover_checkpoints)
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_NAME_RE = re.compile(
    r"^whisper-(?P<size>tiny|base|small|medium|large-v3|distil-small|distil-medium|distil-large)"
    r"-(?P<mode>lora|full)"
    r"-(?P<lorar>\d+)"
    r"-librispeech-asr"
    r"-tpf(?P<tpf>\d+)"
    r"-tf(?P<frames>\d+)"
    r"(?:-xP(?P<xp>[a-zA-Z0-9]+))?"
    r"(?:-xQ(?P<xq>[a-zA-Z0-9]+))?"
    r"(?:-(?P<gpu>[a-zA-Z0-9\-]+))?$"
)


def discover_int4_relevant_checkpoints(
    root: str, include_qat_none: bool = True,
) -> List[Dict]:
    """
    Scan `root` for checkpoints worth running through REAL NF4
    quantization: qat_mode="int4" (the actual gap this script exists to
    fill) and, if include_qat_none, qat_mode="none" counterparts too
    (needed for the "does QAT-INT4 training actually help NF4 deployment"
    comparison this script's docstring describes -- without a none
    baseline at the SAME model/mode/tpf/tf/sparsity, there's nothing to
    compare an int4-trained result against).

    Deliberately does NOT include qat_mode="int8"/"fp16" checkpoints --
    INT8 already has its own correct, separate real-conversion path via
    finetune.py's finalize_and_convert()/inference2.py's
    load_int8_converted_full/lora (quantize_dynamic, CPU-only); running
    THIS script's NF4 path on an INT8-trained checkpoint would apply a
    third, different quantization scheme on top of one already tuned for
    a different one, answering a question nobody asked.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Checkpoint root not found: {root}")

    keep_modes = {"int4"} | ({"none"} if include_qat_none else set())
    specs = []
    for entry in sorted(root_path.iterdir()):
        if not entry.is_dir():
            continue
        m = CHECKPOINT_NAME_RE.match(entry.name)
        if m is None:
            continue
        if not any((entry / f).exists()
                   for f in ("adapter_config.json", "config.json")):
            continue

        cfg = _read_experiment_cfg(str(entry))
        qat_mode = cfg.get("qat_mode", m.group("xq") or "none")
        if qat_mode not in keep_modes:
            continue

        size  = m.group("size")
        mode  = m.group("mode")
        tpf   = cfg.get("tokens_per_frame", int(m.group("tpf")))
        tf    = cfg.get("total_frames", int(m.group("frames")))
        # NEW: lora rank, preferring experiment_cfg.json's own "lora_r"
        # (authoritative) over the folder-name digit (also correct, but
        # experiment_cfg.json is the source of truth used everywhere else
        # in this project). 0 for mode=full, where rank isn't meaningful.
        lorar = cfg.get("lora_r", int(m.group("lorar")))
        specs.append({
            "path": str(entry), "model_size": size, "mode": mode,
            "lorar": lorar,
            "tokens_per_frame": tpf, "total_frames": tf,
            "sparsity_pattern": cfg.get("sparsity_pattern", "dense"),
            "qat_mode": qat_mode,
        })

    print(f"\n  Discovered {len(specs)} INT4-relevant checkpoint(s) under "
          f"'{root}' (qat_mode in {sorted(keep_modes)}):")
    for s in specs:
        print(f"    {s['model_size']:<16} {s['mode']:<6} lora_r={s['lorar']} "
              f"tpf={s['tokens_per_frame']} tf={s['total_frames']:<5} "
              f"xP={s['sparsity_pattern']:<6} qat_mode={s['qat_mode']}  "
              f"[{s['path']}]")
    print()
    return specs


# ─────────────────────────────────────────────────────────────────────────────
# PER-CHECKPOINT RUN (single-checkpoint entry point, reused by the sweep)
# ─────────────────────────────────────────────────────────────────────────────

def run_one_checkpoint(
    checkpoint: str, model_size: str, mode: str,
    tokens_per_frame: Optional[int], total_frames: Optional[int],
    num_beams: int, max_eval_samples: Optional[int],
    output_dir: str, lorar: Optional[int] = None,
    ds_cache: Optional[Dict] = None,
) -> Dict:
    device = torch.device("cuda")
    saved_cfg = _read_experiment_cfg(checkpoint)
    tpf = tokens_per_frame or saved_cfg.get("tokens_per_frame", 1)
    tf  = total_frames     or saved_cfg.get("total_frames", 1500)
    qat_mode_train   = saved_cfg.get("qat_mode", "none")
    sparsity_pattern = saved_cfg.get("sparsity_pattern", "dense")

    # NEW: lora rank -- prefer explicit arg (sweep already resolved it),
    # else experiment_cfg.json, else parse the checkpoint dirname itself
    # (single-checkpoint CLI invocations never had this before at all).
    if lorar is None:
        lorar = saved_cfg.get("lora_r")
        if lorar is None:
            m = CHECKPOINT_NAME_RE.match(os.path.basename(checkpoint.rstrip("/")))
            lorar = int(m.group("lorar")) if m else 0

    print(f"\n{'='*72}")
    print(f"  REAL NF4 QUANTIZATION — {model_size} | {mode} | lora_r={lorar} | "
          f"tpf={tpf} tf={tf} | xP={sparsity_pattern} | "
          f"trained with qat_mode={qat_mode_train}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"{'='*72}\n")

    processor = WhisperProcessor.from_pretrained(
        LOCAL_PATH[model_size], language="English", task="transcribe"
    )
    if os.path.exists(os.path.join(checkpoint, "tokenizer_config.json")):
        from transformers import WhisperTokenizer
        processor.tokenizer = WhisperTokenizer.from_pretrained(
            checkpoint, language="English", task="transcribe"
        )

    model = load_and_merge_float_model(checkpoint, model_size, mode, saved_cfg)
    model = apply_real_nf4_quantization(model, device)

    if tpf > 1:
        model = WhisperWithTokenSubsampling(model, tpf)

    model_size_mb = measure_model_size_mb(model)
    print(f"  Real quantized model size: {model_size_mb} MB")

    # ds_cache lets a sweep reuse the same loaded dataset across
    # checkpoints instead of re-downloading/re-loading it every time.
    if ds_cache is not None and "ds" in ds_cache:
        ds = ds_cache["ds"]
    else:
        print("  Loading librispeech test-other ...")
        ds = load_dataset("librispeech_asr", "other", split="test", streaming=False)
        if max_eval_samples:
            ds = ds.take(max_eval_samples)
        if ds_cache is not None:
            ds_cache["ds"] = ds

    result = run_eval(
        model, processor, ds, text_col="text", device=device,
        total_frames=tf, tokens_per_frame=tpf, num_beams=num_beams,
    )

    # NEW: real FLOPs, using the ACTUAL average output length from this
    # run (not borrowed from a different checkpoint's eval, which is what
    # every FLOPs number in earlier batches of this project's NF4 results
    # had to fall back to since this field didn't exist yet) and the
    # correct quantization="int4" scale -- this script only ever runs one
    # real precision, so there's no mislabeling risk the way inference2.py
    # has with its hardcoded "fp16" label.
    flops = estimate_flops(
        model_size, tf, tpf, quantization="int4",
        sparsity_pattern=sparsity_pattern,
        avg_output_tokens=int(round(result["avg_output_tokens"])),
    )

    print(f"\n  WER={result['wer_pct']:.3f}%  RTF={result['rtf_overall']:.4f}  "
          f"size={model_size_mb}MB  EffFLOPs={flops['flops_total_eff_G']}G  "
          f"avg_tok={result['avg_output_tokens']}  "
          f"hit_max={result['hit_max_new_tokens_pct']}%")
    if result["hit_max_new_tokens_pct"] > 20:
        print(f"  [WARNING] {result['hit_max_new_tokens_pct']}% of samples hit "
              f"max_new_tokens without an EOS -- this WER may reflect "
              f"degenerate/repetitive generation rather than genuine "
              f"transcription failure. Worth inspecting predictions "
              f"directly before trusting this number as-is.")

    os.makedirs(output_dir, exist_ok=True)
    out = {
        "run_info": {
            "timestamp":        datetime.datetime.now().isoformat(),
            "checkpoint":        checkpoint,
            "model_size":        model_size,
            "mode":              mode,
            "lorar":             lorar,
            "tokens_per_frame":  tpf,
            "total_frames":      tf,
            "num_beams":         num_beams,
        },
        "compression": {
            "sparsity_pattern":        sparsity_pattern,
            "qat_mode_train":          qat_mode_train,
            "real_quantization":       "nf4",
            "real_quantization_note": (
                "Genuine bitsandbytes NF4 4-bit weights, applied here as an "
                "explicit post-training step. Distinct from qat_mode_train, "
                "which only reflects what noise (if any) was simulated "
                "during training -- compare this result against the same "
                "config's qat_mode_train='none' counterpart to test "
                "whether QAT training actually helped this NF4 outcome."
            ),
            "model_size_mb": model_size_mb,
        },
        "flops":  flops,
        "result": result,
    }
    fname = (f"nf4_{model_size}_{mode}_r{lorar}_tf{tf}_tpf{tpf}"
             f"_xP{sparsity_pattern.replace(':','')}_qat{qat_mode_train}.json")
    out_path = os.path.join(output_dir, fname)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  JSON → {out_path}")

    # Free GPU memory before the next checkpoint in a sweep -- otherwise
    # peak-memory readings for smaller models later in the sweep can be
    # inflated by an earlier, larger model's still-cached allocations
    # (the same class of cross-checkpoint contamination flagged for
    # inference2.py's sweep loop a few messages back).
    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()

    return out


def main():
    p = argparse.ArgumentParser(description="Real NF4 quantization + evaluation")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="Single checkpoint path")
    src.add_argument("--checkpoint_root",
                      help="Directory to scan for INT4-relevant checkpoints "
                           "(qat_mode=int4, plus qat_mode=none counterparts "
                           "unless --no_qat_none_baseline is passed)")
    p.add_argument("--no_qat_none_baseline", action="store_true", default=False,
                   help="With --checkpoint_root, skip qat_mode=none checkpoints "
                        "-- only run the actual int4-trained ones.")
    p.add_argument("--model_size",  choices=list(LOCAL_PATH.keys()),
                   help="Required with --checkpoint; ignored with --checkpoint_root")
    p.add_argument("--mode",        choices=["lora", "full"],
                   help="Required with --checkpoint; ignored with --checkpoint_root")
    p.add_argument("--benchmark",   default="librispeech", choices=["librispeech"])
    p.add_argument("--tokens_per_frame", type=int, default=None)
    p.add_argument("--total_frames",     type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--num_beams",   type=int, default=16)
    p.add_argument("--output_dir",  default="/fs/nexus-scratch/vyomwal5/eval_results_nf4")
    args = p.parse_args()

    assert torch.cuda.is_available(), (
        "bitsandbytes NF4 requires a CUDA GPU — unlike inference2.py's "
        "INT8 path (quantize_dynamic, CPU-only), this one can't fall "
        "back to CPU."
    )

    if args.checkpoint:
        assert args.model_size and args.mode, \
            "--model_size and --mode are required with --checkpoint"
        run_one_checkpoint(
            args.checkpoint, args.model_size, args.mode,
            args.tokens_per_frame, args.total_frames,
            args.num_beams, args.max_eval_samples, args.output_dir,
        )
        return

    # ── Sweep ─────────────────────────────────────────────────────────────
    specs = discover_int4_relevant_checkpoints(
        args.checkpoint_root, include_qat_none=not args.no_qat_none_baseline,
    )
    if not specs:
        print("Nothing to run — no int4-relevant checkpoints found under "
              f"{args.checkpoint_root}.")
        return

    ds_cache: Dict = {}
    all_results = []
    for i, spec in enumerate(specs, 1):
        print(f"\n>>> [{i}/{len(specs)}] {spec['path']}")
        try:
            out = run_one_checkpoint(
                spec["path"], spec["model_size"], spec["mode"],
                args.tokens_per_frame or spec["tokens_per_frame"],
                args.total_frames or spec["total_frames"],
                args.num_beams, args.max_eval_samples, args.output_dir,
                lorar=spec["lorar"], ds_cache=ds_cache,
            )
            all_results.append(out)
        except Exception as e:
            print(f"  [SWEEP] FAILED on {spec['path']}: {e}")
            print("  [SWEEP] Continuing with remaining checkpoints.")
            continue

    print(f"\n{'='*72}\n  SWEEP COMPLETE — {len(all_results)}/{len(specs)} succeeded\n{'='*72}")
    for r in all_results:
        ri, cp, fl = r["run_info"], r["compression"], r["flops"]
        print(f"  {ri['model_size']:<16} {ri['mode']:<6} r={ri['lorar']:<3} "
              f"qat={cp['qat_mode_train']:<5} xP={cp['sparsity_pattern']:<6} "
              f"WER={r['result']['wer_pct']:.3f}%  size={cp['model_size_mb']}MB  "
              f"EffFLOPs={fl['flops_total_eff_G']}G")

    if len(all_results) > 1:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_path = os.path.join(args.output_dir, f"nf4_sweep_{ts}.json")
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Combined sweep JSON → {combined_path}")


if __name__ == "__main__":
    main()