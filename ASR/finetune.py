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

Axes of compute reduction (aligned with Wang et al. and USM-Lite):
  xN → --model_size        : model capacity (tiny/small/medium/large-v3/distil-*)
  xT → --total_frames      : audio duration truncation (fewer encoder frames)
  xV → --tokens_per_frame  : encoder output subsampling before decoder cross-attn
  xR → --lora_r            : LoRA rank {0=no-LoRA, 8, 16, 32, 64}
  xP → --sparsity_pattern  : N:M weight sparsity {dense, 2:4, 1:4}

                             PRUNING STRATEGY (unified, both patterns):
                             Sparsity is now applied AFTER a dense LoRA run
                             has converged, followed by a fixed-length
                             recovery fine-tune (default 500 steps). This
                             replaces the old "prune before LoRA training"
                             approach, which forced the adapter to learn
                             the task and compensate for a heavily-pruned
                             base simultaneously from random init — this
                             was the direct cause of the instability seen
                             on xP=1:4 sweep configs. Pre- vs. post-training
                             pruning produces IDENTICAL final FLOPs/RTF
                             (same sparsity ratio either way); the only
                             thing post-training pruning buys you is
                             better WER / stabler convergence, since the
                             adapter starts recovery from an
                             already-converged solution instead of from
                             scratch under a handicap.

                             Within the shared post-hoc-prune + recovery
                             strategy, patterns still differ in how the
                             mask is *updated* during recovery (this is a
                             property of the sparsity ratio itself, not a
                             competing training strategy):
                               2:4 → mask computed once at the start of
                                     recovery, then frozen. Hardware
                                     sparse GEMM via
                                     to_sparse_semi_structured (Ampere+).
                               1:4 → mask recomputed every 50 steps for
                                     the full recovery window (default
                                     500 steps), then frozen. A 75% sparse
                                     mask is a bigger shock than 50%, so
                                     it benefits from being re-derived as
                                     the adapter adjusts.

                             See prune_and_recover() for the entry point.
                             The old prune-before-LoRA path in
                             build_model_lora() / build_model_full() is
                             kept only for backward compatibility with
                             mode=full (where the base is trainable and
                             genuinely drifts under the sparsity mask —
                             pre- vs post- distinction doesn't apply the
                             same way there). For mode=lora, use
                             --prune_after_lora instead of
                             --sparsity_pattern at train() time.
  xQ (QAT) → --qat_mode    : Quantization-aware training {none, int8, int4}
                             Applied only to Pareto-optimal configs (Stage 2).
                             Uses torch.quantization fake-quant observers.
                             PTQ (post-training) lives in inference_eval.py.

Pipeline
--------
  Stage 1 — Dense sweep         : vary xN × xT × xV; log FLOPs / WER / size
  Stage 1.5 — Prune + recover   : for LoRA configs on the Pareto frontier,
                                   take the converged dense checkpoint and
                                   apply xP via prune_and_recover() (this
                                   file), which prunes the base then runs
                                   a short recovery fine-tune.
  Stage 2 — QAT                 : apply xQ to configs on the Pareto
                                   frontier from Stage 1 / Stage 1.5
  Stage 3 — PTQ                 : apply PTQ to Stage-1/2 checkpoints
                                   (handled in inference_eval.py)

N:M sparsity implementation note
---------------------------------
We use PyTorch's built-in semi-structured sparsity support
(torch.sparse.to_sparse_semi_structured / torchao WeightNormSparsifier).
This gives hardware-accelerated sparse GEMM on Ampere Tensor Cores
(A100, A6000, RTX 3090+) with cuSPARSELt or CUTLASS backends.

  2:4  → one-shot magnitude pruning at start of recovery; mask fixed
  1:4  → iterative: mask computed every 50 steps for the recovery
          window, then frozen (ramp-in schedule avoids accuracy collapse)

Sparse base weights + dense LoRA (Option A):
  Sparsity is applied to the *base* model weights; LoRA delta matrices
  (B, A) remain dense. This mimics the USM-Lite approach of compressing
  the backbone while keeping the task-specific adaptation pathway at
  full precision.

QAT implementation note
------------------------
QAT is implemented as an additional training mode triggered by --qat_mode.
Fake-quant observers (per-channel, symmetric) are inserted into all
nn.Linear weight tensors via torch.quantization.prepare_qat().
INT4 QAT uses a custom 4-bit observer with min/max range calibration.
QAT forward pass emulates quantisation noise during training so that
weights converge to values that survive actual post-training quantisation.

References
----------
  Radford et al. (2023), Whisper
  Hu et al. (2022), LoRA
  Wang et al. (2025), Inference compute-optimal VLMs
  Gandhi et al. (2024), Distil-Whisper
  Panayotov et al. (2015), LibriSpeech
  Ding et al. (2024), USM-Lite, ICASSP 2024   ← compression framework extended here
  NVIDIA (2020), Automatic Sparsity (ASP)     ← prune-then-recover pattern
  Pool & Yu (2021), NVIDIA 2:4 sparsity whitepaper

Can be used standalone OR imported/called by run_experiment.py.

Usage (standalone):
  # Standard LoRA fine-tune (dense baseline) — always dense at this stage now
  python whisper_finetune.py --model_size small --mode lora --streaming

  # Post-training pruning + recovery on a converged dense LoRA checkpoint
  # (Stage 1.5, xP axis, unified strategy for both 2:4 and 1:4)
  python whisper_finetune.py --prune_after_lora \\
      --dense_checkpoint /path/to/dense/lora/checkpoint \\
      --sparsity_pattern 2:4 --recovery_steps 500

  python whisper_finetune.py --prune_after_lora \\
      --dense_checkpoint /path/to/dense/lora/checkpoint \\
      --sparsity_pattern 1:4 --recovery_steps 500

  # QAT INT8 on a Pareto-optimal config (Stage 2)
  python whisper_finetune.py --model_size small --mode lora \\
      --qat_mode int8 --sparsity_pattern dense

  # Full sweep over xN × xT × xV (Stage 1)
  python whisper_finetune.py --sweep \\
      --sweep_sizes small,medium --sweep_modes lora \\
      --total_frames 1500 --tokens_per_frame 1
"""

# ─────────────────────────────────────────────────────────────────────────────
# CACHE SETUP — must be before ALL other imports
# ─────────────────────────────────────────────────────────────────────────────
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
VYOM_CACHE = "/scratch/zt1/project/msml604/user/vyomwal5/anaconda3/envs/asr/hf_cache"

local_path = {
    "small": f"{VYOM_CACHE}/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d",
    "medium": f"{VYOM_CACHE}/models/models--openai--whisper-medium/snapshots/abdf7c39ab9d0397620ccaea8974cc764cd0953e",
    "tiny": f"{VYOM_CACHE}/models/models--openai--whisper-tiny/snapshots/169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
    "large-v3": f"{VYOM_CACHE}/models/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1"
}

os.environ["HF_HOME"]                = CACHE_DIR
os.environ["HF_DATASETS_CACHE"]      = f"{VYOM_CACHE}/datasets"
os.environ["TRANSFORMERS_CACHE"]     = f"{VYOM_CACHE}/models"
os.environ["HUGGINGFACE_HUB_CACHE"]  = f"{VYOM_CACHE}/hub"
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"
# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import re
import torchaudio
import time
import json
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import evaluate
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Union
from jiwer import wer as jiwer_wer
from datasets import (load_dataset, Audio, IterableDataset, Dataset)
from transformers import (
    WhisperForConditionalGeneration, WhisperProcessor,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel
import tqdm

# ── Optional sparsity imports (requires torch >= 2.1 + Ampere GPU) ───────────
try:
    from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
    from torch.ao.pruning import WeightNormSparsifier
    SPARSE_AVAILABLE = True
except ImportError:
    SPARSE_AVAILABLE = False
    print("WARNING: torch.sparse semi-structured API not available "
          "(requires PyTorch >= 2.1 and Ampere+ GPU). "
          "Sparsity patterns will be emulated as dense with zero-masking.")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

WHISPER_SIZES = {
    "tiny":          "openai/whisper-tiny",
    "base":          "openai/whisper-base",
    "small":         "openai/whisper-small",
    "distil-small":  "distil-whisper/distil-small.en",
    "medium":        "openai/whisper-medium",
    "distil-medium": "distil-whisper/distil-medium.en",
    "large-v3":      "openai/whisper-large-v3",
    "distil-large":  "distil-whisper/distil-large-v3",
}

SUPPORTED_TASKS = ["asr", "translation"]

BENCHMARK_REGISTRY = {
    "librispeech": {
        "hf_path":       "librispeech_asr",
        "text_column":   "text",
        "language":      "English",
        "default_train": "train.100",
        "default_eval":  "validation",
        "default_test":  [("clean", "test")],
        "task":          "transcribe",
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

# Valid xP sparsity patterns
VALID_SPARSITY_PATTERNS = ["dense", "2:4", "1:4"]

# Valid QAT modes (training-time quantization)
VALID_QAT_MODES = ["none", "int8", "int4"]

# Mask ramp-in schedule for 1:4: apply mask update every N steps for first M steps
# NOTE: this now describes the RECOVERY window (post-training pruning),
# not a schedule applied during the original dense LoRA training.
ITERATIVE_MASK_STEPS   = 500   # total steps during which mask is updated
ITERATIVE_MASK_FREQ    = 50    # update mask every this many steps


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# xP AXIS — N:M SEMI-STRUCTURED SPARSITY
# ─────────────────────────────────────────────────────────────────────────────

def _nm_pattern_to_zeros(pattern: str) -> int:
    """Parse 'N:M' → number of zeros per block (N)."""
    n, m = pattern.split(":")
    return int(n)


def _get_linear_modules_for_sparsity(model: nn.Module) -> List[str]:
    """
    Return fully-qualified names of all nn.Linear modules in the base
    Whisper encoder and decoder that are candidates for sparsification.
    We skip LoRA adapter matrices (A/B projections) so that only base
    weights are made sparse (Option A: sparse base + dense LoRA).
    """
    target_fqns = []
    # Navigate to the underlying WhisperForConditionalGeneration
    base = model
    if isinstance(base, WhisperWithTokenSubsampling):
        base = base.model
    if isinstance(base, PeftModel):
        base = base.get_base_model()

    for name, module in base.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # Skip LoRA injected adapters (named lora_A, lora_B by PEFT)
        if "lora_" in name:
            continue
        # Skip tiny projection layers (vocab embedding projections)
        # to avoid shape mismatches with 4-element block requirement
        rows, cols = module.weight.shape
        if rows < 4 or cols < 4:
            continue
        if rows % 4 != 0 or cols % 4 != 0:
            continue
        target_fqns.append(f"{name}.weight")
    return target_fqns


def apply_nm_sparsity_oneshot(
    model:   nn.Module,
    pattern: str,    # "2:4" or "1:4"
    use_hardware_sparse: bool = True,
) -> nn.Module:
    """
    Apply N:M magnitude-based one-shot sparsity to all eligible nn.Linear
    weight tensors in the base model.

    For 2:4: one-shot, mask frozen immediately — uses hardware-accelerated
             SparseSemiStructuredTensor if available (Ampere+ GPU).
    For 1:4: same one-shot application; iterative updating during the
             recovery window is handled by NMSparseCallback.

    This function is called on the base model — for the unified
    post-training pruning strategy, that's a base whose values were
    produced by a fully-converged dense LoRA run.

    IMPORTANT: this function is deterministic given the same input weights
    (magnitude-based mask). Since the base is frozen throughout LoRA
    training/recovery (never gradient-updated) and is not touched by
    training in mode=lora, it is safe (and required) to re-run this exact
    function on a freshly-loaded pretrained base at eval time to
    reconstruct the identical sparse structure that was actually trained
    against — see evaluate_checkpoint().

    Parameters
    ----------
    model              : WhisperForConditionalGeneration (plain, pre-LoRA)
                         or a base already wrapped by PEFT
                         (get_base_model() is resolved internally where
                         relevant callers need it — this function itself
                         expects a plain nn.Module and walks named_modules).
    pattern            : "2:4" or "1:4"
    use_hardware_sparse: if True and SPARSE_AVAILABLE, convert to
                         SparseSemiStructuredTensor for accelerated GEMM.
                         Set False for emulation on non-Ampere hardware.
    """
    if pattern == "dense":
        return model

    n, m = int(pattern.split(":")[0]), int(pattern.split(":")[1])
    print(f"\n  [xP] Applying {pattern} N:M sparsity (one-shot magnitude) ...")

    output_head = None
    if hasattr(model, "get_output_embeddings"):
        try:
            output_head = model.get_output_embeddings()
        except Exception:
            output_head = None

    pruned_count  = 0
    skipped_count = 0

    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if "lora_" in name:
                continue
            if module is output_head or name.endswith("proj_out"):
                # Never prune the vocab output head — it's weight-tied to
                # the token embedding and, in LoRA mode, is never trained
                # (LORA_TARGET_MODULES only covers q_proj/v_proj), so any
                # damage here is permanent and unrecoverable. Zeroing 75%
                # of it (at 1:4) is a direct cause of persistently high
                # WER even after decoding/repetition fixes are applied.
                skipped_count += 1
                continue
            w = module.weight.data
            rows, cols = w.shape
            if rows < m or cols < m or rows % m != 0 or cols % m != 0:
                skipped_count += 1
                continue

            # ── Magnitude-based N:M mask per row block of size m ─────────────
            # Reshape to (-1, m), zero the n smallest-magnitude per block
            orig_shape   = w.shape
            w_blocks     = w.reshape(-1, m)
            _, sorted_idx = torch.sort(w_blocks.abs(), dim=1)
            # Zero out the (m - n) smallest elements in each block
            zeros_per_block = m - n
            zero_idx  = sorted_idx[:, :zeros_per_block]
            mask      = torch.ones_like(w_blocks, dtype=torch.bool)
            mask.scatter_(1, zero_idx, False)
            w_sparse  = w_blocks * mask.float()
            module.weight.data = w_sparse.reshape(orig_shape)

            # ── Convert to hardware sparse format if available ────────────────
            if use_hardware_sparse and SPARSE_AVAILABLE and n == 2 and m == 4:
                try:
                    module.weight = nn.Parameter(
                        to_sparse_semi_structured(module.weight.data.contiguous())
                    )
                    pruned_count += 1
                    continue
                except Exception as e:
                    print(f"  [xP] SparseSemiStructuredTensor failed for {name}: {e}. "
                          f"Keeping zero-masked dense tensor.")

            pruned_count += 1

    print(f"  [xP] {pattern} sparsity applied to {pruned_count} Linear layers "
          f"({skipped_count} skipped due to shape constraints).")
    return model


class NMSparseCallback:
    """
    Enforces N:M sparsity DURING training, faithful to Algorithm 1 of the
    reference paper (USM-Lite):

      • MASK SELECTION happens on a schedule: one-shot (mask computed once
        at the start, `mask_update_steps=0`) or few-shot (mask recomputed
        from current weight magnitudes every `mask_update_freq` steps while
        `step < mask_update_steps`, then frozen). The paper found few-shot
        matters for 1:4 (10.6 vs 11.7 WER) and not for 2:4 (4.3 vs 4.4).

      • MASK ENFORCEMENT is continuous: the (current) stored mask re-zeros
        the pruned positions after EVERY optimizer step, for the entire
        run — matching Algorithm 1's unconditional "Prune each weight
        matrix through the mask" at every iteration t. This is what stops
        the optimizer from regrowing pruned weights when they are
        trainable (mode=full / full-recovery), and is a no-op safety net
        when the base is PEFT-frozen (LoRA recovery).

    IMPORTANT — what this deliberately does NOT do: recompute the mask
    from scratch every step for the whole run. An earlier revision did
    exactly that, and it makes the surviving-weight set a moving target
    for all of training (weights near the magnitude threshold flip in and
    out of the mask step after step), which empirically stalls
    convergence entirely (~96-103% WER, loss stuck >10). The paper never
    does this: after the selection window, WHICH weights survive is
    frozen; only their VALUES keep training.

    Usage:
        cb = NMSparseCallback(pattern, mask_update_steps=..., mask_update_freq=...)
        cb.capture_masks(model)          # once, right after the initial prune
        cb.on_step_end(model, step)      # after every optimizer step
    """

    def __init__(self, pattern: str = "1:4",
                 mask_update_steps: int = 0,
                 mask_update_freq: int = ITERATIVE_MASK_FREQ):
        assert pattern in ("2:4", "1:4"), \
            "NMSparseCallback pattern must be '2:4' or '1:4'."
        self.pattern   = pattern
        self.n, self.m = int(pattern.split(":")[0]), int(pattern.split(":")[1])
        # mask_update_steps=0 → pure one-shot (capture once, never update).
        # mask_update_steps>0 → few-shot: recompute mask every
        # mask_update_freq steps while step < mask_update_steps.
        self.mask_update_steps = mask_update_steps
        self.mask_update_freq  = max(1, mask_update_freq)
        self.masks: dict = {}          # module_name -> bool mask (weight shape)
        self._frozen_announced = False

    # ── internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _unwrap(model: nn.Module) -> nn.Module:
        base = model
        # peel DDP / accelerate wrappers first (they expose .module)
        while hasattr(base, "module") and isinstance(
                getattr(base, "module"), nn.Module):
            base = base.module
        if isinstance(base, WhisperWithTokenSubsampling):
            base = base.model
        if isinstance(base, PeftModel):
            base = base.get_base_model()
        return base

    @staticmethod
    def _eligible_linears(base: nn.Module, m: int):
        output_head = None
        if hasattr(base, "get_output_embeddings"):
            try:
                output_head = base.get_output_embeddings()
            except Exception:
                output_head = None
        for name, module in base.named_modules():
            if not isinstance(module, nn.Linear) or "lora_" in name:
                continue
            if module is output_head or name.endswith("proj_out"):
                continue  # never prune the tied vocab output head
            w = module.weight.data
            rows, cols = w.shape
            if rows < m or cols < m or rows % m != 0 or cols % m != 0:
                continue
            yield name, module

    def _compute_mask(self, w: torch.Tensor) -> torch.Tensor:
        """Magnitude-based N:M mask for one weight tensor (bool, w.shape)."""
        n, m = self.n, self.m
        w_blocks = w.reshape(-1, m)
        _, sorted_idx = torch.sort(w_blocks.abs(), dim=1)
        zero_idx = sorted_idx[:, : m - n]
        mask = torch.ones_like(w_blocks, dtype=torch.bool)
        mask.scatter_(1, zero_idx, False)
        return mask.reshape(w.shape)

    # ── public API ───────────────────────────────────────────────────────────

    def capture_masks(self, model: nn.Module):
        """(Re)compute and store masks from current weight magnitudes, and
        immediately zero through them. Call once after the initial prune;
        called again automatically during a few-shot update window."""
        base = self._unwrap(model)
        with torch.no_grad():
            self.masks = {}
            for name, module in self._eligible_linears(base, self.m):
                w = module.weight.data
                mask = self._compute_mask(w)
                self.masks[name] = mask
                module.weight.data = w * mask.to(w.dtype)
        print(f"  [xP] Mask captured for {len(self.masks)} Linear layer(s) "
              f"({self.pattern}).")

    def on_step_end(self, model: nn.Module, step: int):
        """Call after every optimizer step. Recomputes the mask only inside
        the few-shot window; otherwise re-zeros through the frozen mask."""
        if not self.masks:            # lazy init if capture_masks wasn't called
            self.capture_masks(model)
            return

        in_update_window = (self.mask_update_steps > 0
                            and step < self.mask_update_steps)
        if in_update_window and step % self.mask_update_freq == 0:
            self.capture_masks(model)   # few-shot: re-select AND re-zero
            return
        if (self.mask_update_steps > 0 and not in_update_window
                and not self._frozen_announced):
            print(f"  [xP] Few-shot window complete at step {step} — mask "
                  f"frozen for the remainder of training.")
            self._frozen_announced = True

        base = self._unwrap(model)
        with torch.no_grad():
            for name, module in self._eligible_linears(base, self.m):
                mask = self.masks.get(name)
                if mask is None:
                    continue
                w = module.weight.data
                if mask.device != w.device:
                    mask = mask.to(w.device)
                    self.masks[name] = mask
                module.weight.data = w * mask.to(w.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# xQ (QAT) AXIS — QUANTIZATION-AWARE TRAINING
# ─────────────────────────────────────────────────────────────────────────────

class _Int4FakeQuantize(torch.quantization.FakeQuantize):
    """
    4-bit fake-quantize observer.
    Symmetric, PER-CHANNEL, 16 quantization levels ([-8, 7] for signed
    INT4).  Used for INT4 QAT to simulate 4-bit weight noise during
    training.

    Accepts and swallows **kwargs (in particular `factory_kwargs`): newer
    torch.ao.quantization.prepare_qat() passes device/dtype hints to every
    observer/fake-quant constructor it instantiates. The previous
    zero-argument __init__ rejected these, causing prepare_qat() to raise
    "_Int4FakeQuantize.__init__() got an unexpected keyword argument
    'factory_kwargs'" on every INT4 run and silently fall back to the
    manual fake-quant insertion path instead of the tested, official one.

    PER-CHANNEL, not per-tensor: this was a real bug, found and fixed
    after the fact — the INT8 weight-only fix (see apply_qat's docstring)
    switched INT8 to per-channel quantization but INT4 was never checked
    and was left on MovingAverageMinMaxObserver, the per-TENSOR variant.
    This matters far more at 4 bits than at 8: per-tensor forces every
    output channel to share one scale calibrated to the whole tensor's
    max, so channels with smaller natural magnitude get crushed toward
    very few of the 16 available levels while only the single
    largest-magnitude channel gets full resolution. At INT8's 256 levels
    there's enough headroom that this mostly doesn't matter; at INT4's 16
    levels there's almost no slack to lose. Per-channel scales each
    output channel independently, avoiding this. Verified
    MovingAveragePerChannelMinMaxObserver exists and is usable with the
    same construction pattern before making this change.
    """
    def __init__(self, **kwargs):
        kwargs.pop("factory_kwargs", None)
        super().__init__(
            observer=torch.quantization.MovingAveragePerChannelMinMaxObserver,
            quant_min=-8,
            quant_max=7,
            dtype=torch.qint8,       # closest supported dtype; levels limited below
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
            reduce_range=False,
        )
        # Override bit-width to 4 by clamping the effective range
        self.quant_min = -8
        self.quant_max =  7


def apply_qat(model: nn.Module, qat_mode: str) -> nn.Module:
    """
    Insert fake-quantize observers into all nn.Linear weight tensors for
    quantization-aware training.

    Both INT8 and INT4 QAT are now WEIGHT-ONLY: per-channel (INT8) / custom
    4-bit (INT4) symmetric fake-quant on weights; activations are left at
    full precision (torch.nn.Identity — no observer, no fake-quant at all).

    This was NOT always true and the change is a direct bug fix. Verified
    empirically (see finetune.py history / project notes): the previous
    INT8 qconfig came from torch.quantization.get_default_qat_qconfig(),
    which attaches BOTH weight fake-quant AND an activation fake-quant
    using MovingAverageMinMaxObserver — a plain, unbounded, per-TENSOR
    running min/max with no outlier protection. Speech activations vary
    batch-to-batch with the actual audio content (loud vs. quiet, silence
    vs. dense energy), which is exactly the kind of input that produces
    occasional extreme values; a single one landing early, before the
    observer has enough history to smooth it out, permanently blows out
    the observed range for the whole tensor (since it's per-tensor, one
    outlier anywhere corrupts the scale for everything). This is the
    leading explanation for a real observed failure: an INT8 QAT training
    run whose loss was converging cleanly (3.0 -> 0.42 over 375 steps)
    suddenly spiked to 10+ at step 400 and never fully recovered over the
    remaining 3600 steps.

    The INT4 branch's previous code had exactly the same problem despite
    its own docstring claiming "activation quantization at 4-bit is too
    aggressive for speech" (i.e. claiming weight-only) — the actual
    implementation built and attached an INT8 act_observer anyway,
    directly contradicting that stated intent. Both are now genuinely
    weight-only, matching what INT4's docstring always claimed.

    Per-channel weight quantization (used for both modes) is far less
    outlier-sensitive than per-tensor activation quantization: each output
    channel's weight distribution is stable throughout training (weights
    change slowly via gradient descent), unlike activations which are
    driven directly by whatever audio happens to be in the current batch.

    Called AFTER LoRA insertion (fake-quant wraps the merged forward path)
    and AFTER sparsity application (xP applied to base, xQ wraps all Linear).

    Returns the prepared model (with fake-quant nodes inserted).
    The model must be converted with torch.quantization.convert() after
    training to obtain a truly-quantized inference model.
    """
    assert qat_mode in ("int8", "int4"), \
        f"qat_mode must be 'int8' or 'int4', got '{qat_mode}'"

    print(f"\n  [xQ-QAT] Inserting {qat_mode.upper()} fake-quantize observers "
          f"(weight-only; activations stay full precision) ...")

    # Navigate to the underlying WhisperForConditionalGeneration
    inner = model
    if isinstance(inner, WhisperWithTokenSubsampling):
        inner = inner.model

    if qat_mode == "int8":
        # Weight-only: per-channel symmetric INT8 fake-quant on weights,
        # activations pass through untouched (nn.Identity — no observer).
        qconfig = torch.quantization.QConfig(
            activation=torch.nn.Identity,
            weight=torch.quantization.default_per_channel_weight_fake_quant,
        )
    else:
        # INT4: custom 4-bit fake-quantize on weights only. No activation
        # observer at all now (previously built one despite the docstring
        # saying otherwise — see function docstring above).
        wt_observer = _Int4FakeQuantize
        qconfig = torch.quantization.QConfig(
            activation=torch.nn.Identity,
            weight=wt_observer,
        )

    # Apply qconfig to Linear layers, EXCLUDING:
    #   (a) the vocab output projection (proj_out), which is weight-tied to
    #       the decoder token embedding. Fake-quantizing it (esp. at INT4,
    #       16 levels) collapses the logits and is the direct cause of
    #       eval WER=100 — the model degenerates to near-constant/garbage
    #       token predictions.
    #   (b) LoRA's own lora_A / lora_B adapter Linears, which must stay at
    #       full precision per the "sparse base + dense LoRA" design
    #       (Option A) described in this file's docstring — the old
    #       blanket `.apply()` was quantizing them too, since they are
    #       also nn.Linear instances nested inside the wrapped target
    #       modules.
    target_root = inner.get_base_model() if isinstance(inner, PeftModel) else inner

    output_head = None
    if hasattr(target_root, "get_output_embeddings"):
        try:
            output_head = target_root.get_output_embeddings()
        except Exception:
            output_head = None

    skipped = []

    def _set_qconfig_by_name(root):
        for name, mod in root.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            if mod is output_head or name.endswith("proj_out"):
                skipped.append(name or "proj_out")
                continue
            if "lora_A" in name or "lora_B" in name:
                skipped.append(name)
                continue
            mod.qconfig = qconfig

    _set_qconfig_by_name(target_root)
    if skipped:
        print(f"  [xQ-QAT] Excluded from quantization (kept full precision): "
              f"{skipped}")

    # Prepare for QAT — inserts FakeQuantize nodes in the forward graph
    try:
        if isinstance(inner, PeftModel):
            torch.quantization.prepare_qat(inner.get_base_model(), inplace=True)
        else:
            torch.quantization.prepare_qat(inner, inplace=True)
        print(f"  [xQ-QAT] {qat_mode.upper()} QAT preparation complete. "
              f"Fake-quant nodes inserted into Linear weights.")
    except Exception as e:
        print(f"  [xQ-QAT] prepare_qat() raised: {e}. "
              f"Falling back to manual fake-quant insertion.")
        # Fallback: manually register fake-quant on weight parameter
        target = inner.get_base_model() if isinstance(inner, PeftModel) else inner
        for name, module in target.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if "lora_A" in name or "lora_B" in name:
                continue  # keep LoRA adapters dense
            if module is output_head or name.endswith("proj_out"):
                continue  # never quantize the tied vocab output head
            if qat_mode == "int8":
                module.weight_fake_quant = torch.quantization.FakeQuantize(
                    observer=torch.quantization.MovingAveragePerChannelMinMaxObserver,
                    quant_min=-128, quant_max=127,
                    dtype=torch.qint8,
                    qscheme=torch.per_channel_symmetric,
                )
            else:
                module.weight_fake_quant = _Int4FakeQuantize()
            original_forward = module.forward

            def _make_qat_forward(m, orig_fwd):
                def _qat_forward(x):
                    # IMPORTANT: do NOT overwrite m.weight.data in place.
                    # That breaks the autograd graph for the fake-quant op
                    # (no straight-through gradient) and permanently
                    # destroys the full-precision "shadow" weights that
                    # QAT is supposed to keep around during training.
                    # Instead, compute the fake-quantized weight on the
                    # fly and use it only for this forward pass.
                    fq_weight = m.weight_fake_quant(m.weight)
                    return nn.functional.linear(x, fq_weight, m.bias)
                return _qat_forward

            module.forward = _make_qat_forward(module, original_forward)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# xQ FINALIZATION — real conversion from fake-quant simulation to a
# genuinely quantized deployable model
# ─────────────────────────────────────────────────────────────────────────────

def _strip_qat_wrappers(model: nn.Module) -> nn.Module:
    """
    Revert QAT-prepared modules back to plain nn.Linear, keeping their
    final trained weights exactly as-is. apply_qat() never modifies
    module.weight in place — fake-quant only simulates rounding noise
    during the forward pass (see the manual-fallback path's comment on
    this) — so "stripping" is a pure structural revert with zero numeric
    change to the underlying weight tensor.

    Handles both insertion paths apply_qat() can take:
      - torch.quantization.prepare_qat() official path: modules carry a
        `.weight_fake_quant` submodule (and, before the weight-only fix,
        an `.activation_post_process` — harmless to also strip if present
        on an older checkpoint).
      - the manual fallback path (when prepare_qat() itself raises):
        weight_fake_quant is bolted directly onto a plain nn.Linear with
        its .forward monkey-patched.

    Best-effort: verified structurally in a minimal PEFT+LoRA test model
    (forward pass, gradient flow) during development, but not against the
    full Whisper+PEFT module tree at scale — worth a sanity check (compare
    logits against the live fake-quant model just before stripping) the
    first time this runs for real.
    """
    replaced = 0
    for parent in list(model.modules()):
        for child_name, child in list(parent.named_children()):
            if not hasattr(child, "weight_fake_quant"):
                continue
            if isinstance(child, nn.Linear):
                plain = nn.Linear(
                    child.in_features, child.out_features,
                    bias=(child.bias is not None),
                )
                plain.weight = child.weight
                if child.bias is not None:
                    plain.bias = child.bias
                setattr(parent, child_name, plain)
            else:
                try:
                    delattr(child, "weight_fake_quant")
                except Exception:
                    pass
            replaced += 1
            if hasattr(child, "activation_post_process"):
                try:
                    delattr(child, "activation_post_process")
                except Exception:
                    pass
            if hasattr(child, "qconfig"):
                child.qconfig = None
    print(f"  [xQ-convert] Stripped QAT wrappers from {replaced} module(s), "
          f"trained float weight values unchanged.")
    return model


def finalize_and_convert(model: nn.Module, qat_mode: str):
    """
    Turn a QAT-prepared (fake-quant-simulated) model into its FINAL
    deployable form. MUST be called on the LIVE, in-memory model
    immediately after its last training/recovery step, before any
    save/reload boundary — the calibration state fake-quant observers
    accumulate only exists in memory and is not persisted by a normal
    checkpoint save.

    Returns (model, converted: bool, note: str).

    qat_mode == "none": no-op. Returns (model, False, "").

    qat_mode == "int8": apply_qat() trains WEIGHT-ONLY fake-quant
        (per-channel symmetric; activations stay nn.Identity). PyTorch's
        native STATIC quantization (torch.quantization.convert() ->
        nn.quantized.Linear) assumes activations are ALSO quantized via
        QuantStub/DeQuantStub at model boundaries; feeding a converted
        static module a plain float tensor breaks at runtime. Since
        training is deliberately weight-only, the matching REAL
        counterpart is DYNAMIC quantization (quantize_dynamic): weights
        become genuinely int8, activations are computed in floating point
        per call — the same weight-only philosophy training used, just
        performed for real, reusing the exact mechanism already
        implemented in inference.py's PTQ path. Runs on CPU only (eager
        quantization has no CUDA kernels).

    qat_mode == "int4": PyTorch has no native int4 quantized tensor type —
        there is nothing built-in to convert to. _Int4FakeQuantize only
        ever simulated 4-bit rounding noise for gradient purposes during
        training. Real INT4 deployment for this project goes through
        bitsandbytes NF4 (inference.py's apply_quantization()) — a
        genuinely different, NON-LINEAR scheme than the linear symmetric
        4-bit simulated here. Fake-quant is stripped back to float
        (QAT-robustified) weights; real INT4 PTQ remains a separate,
        later step, and whether QAT training actually helps NF4
        specifically is an open empirical question (the two schemes
        don't match), not something this function can resolve.
    """
    if qat_mode == "none":
        return model, False, ""

    was_wrapped_subsampling = isinstance(model, WhisperWithTokenSubsampling)
    tpf = model.tokens_per_frame if was_wrapped_subsampling else 1
    inner = model.model if was_wrapped_subsampling else model

    is_peft = isinstance(inner, PeftModel)
    target  = inner.get_base_model() if is_peft else inner
    target.eval()
    target.to("cpu")

    target = _strip_qat_wrappers(target)

    if qat_mode == "int4":
        note = (
            "INT4: no native torch quantized dtype — fake-quant stripped "
            "back to float (QAT-robustified) weights. Real INT4 "
            "deployment is a separate step via bitsandbytes NF4 in "
            "inference.py (a different, nonlinear scheme than what was "
            "simulated during training — whether QAT training here "
            "actually helps NF4 specifically is untested)."
        )
        print(f"  [xQ-convert] {note}")
        converted_target = target
        converted = False
    else:  # int8
        output_head = target.get_output_embeddings() \
            if hasattr(target, "get_output_embeddings") else None
        qspec = {}
        for name, mod in target.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            if mod is output_head or name.endswith("proj_out"):
                continue
            if "lora_A" in name or "lora_B" in name:
                continue
            qspec[name] = torch.quantization.default_dynamic_qconfig
        converted_target = torch.quantization.quantize_dynamic(
            target, qconfig_spec=qspec, dtype=torch.qint8,
        )
        converted = True
        note = (
            f"INT8: genuinely quantized via dynamic quantization "
            f"({len(qspec)} Linear layer(s)), matching the weight-only "
            f"scheme QAT trained for. CPU only from this point on."
        )
        print(f"  [xQ-convert] {note}")

    # Re-wrap PEFT / subsampling structure. quantize_dynamic() returns a
    # NEW top-level module object (not in-place) — verified empirically
    # that PeftModel.base_model.model is the correct attribute to
    # reassign so the swapped-in module is reachable through the
    # existing lora.Linear wrappers' forward (which call
    # self.base_layer(x) + adapter delta) with no other changes needed.
    if is_peft:
        inner.base_model.model = converted_target
        final_model = inner
    else:
        final_model = converted_target

    if was_wrapped_subsampling:
        final_model = WhisperWithTokenSubsampling(final_model, tpf)

    return final_model, converted, note


def _save_possibly_converted(model: nn.Module, output_dir: str,
                              dense_mode: str, converted: bool) -> None:
    """
    Save a model that may or may not have gone through real quantization
    conversion. HF's save_pretrained()/safetensors cannot serialize
    genuinely quantized modules correctly — a dynamically-quantized
    Linear's packed weight lives inside an opaque
    `_packed_params._packed_params` tuple, invisible to the normal
    parameter iteration safetensors relies on (verified empirically
    during development: sum of .parameters().numel() is exactly ZERO for
    a dynamically-quantized model). So a converted model is saved via
    plain torch.save() on its full state_dict instead, which DOES capture
    the packed tuple correctly.

    mode=lora: unaffected either way — PeftModel.save_pretrained() only
      ever saves the ADAPTER (lora_A/lora_B), never the base, so a
      quantized base underneath doesn't change this call at all. The
      base itself doesn't need saving: for LoRA, the base is frozen
      throughout (never gradient-updated), so it's a deterministic
      function of the original pretrained weights + (if pruned) the
      magnitude mask — both fully reconstructable at eval time without
      persisting anything base-related, exactly like the existing
      dense/pruned LoRA eval path already does.
    mode=full: the base itself is genuinely different from any
      reconstructable reference (it was actually trained), so if
      converted, its real quantized weights must be persisted here as
      the state_dict, alongside config/generation_config (plain JSON,
      unaffected by quantization) saved separately since save_pretrained
      itself isn't used in this branch.

    NOTE: this only handles the SAVE side. Loading a converted checkpoint
    back (in evaluate_checkpoint() or inference.py) needs corresponding
    reload logic that isn't implemented yet — this function alone doesn't
    make converted checkpoints round-trip end-to-end.
    """
    if dense_mode == "lora":
        # Unaffected by conversion — see docstring. Existing behavior.
        model.save_pretrained(output_dir)
        return

    # mode == "full"
    if not converted:
        model.save_pretrained(output_dir, safe_serialization=True)
        return

    print(f"  [xQ-convert] Saving genuinely quantized full-finetune model "
          f"via torch.save() (safetensors cannot represent packed "
          f"quantized weights) ...")
    torch.save(model.state_dict(),
               os.path.join(output_dir, "pytorch_model_quantized.bin"))
    model.config.save_pretrained(output_dir)
    if hasattr(model, "generation_config"):
        model.generation_config.save_pretrained(output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# xV AXIS — ENCODER OUTPUT SUBSAMPLING WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class WhisperWithTokenSubsampling(nn.Module):
    """
    Wraps WhisperForConditionalGeneration (or a PeftModel around it) to
    subsample encoder hidden states BEFORE they are passed to the decoder's
    cross-attention layers.

    xV axis: tokens_per_frame=1 → 1500 encoder tokens (baseline)
             tokens_per_frame=2 → 750 tokens
             tokens_per_frame=4 → 375 tokens
    """

    def __init__(self, base_model: nn.Module, tokens_per_frame: int = 1):
        super().__init__()
        self.model            = base_model
        self.tokens_per_frame = tokens_per_frame

    def _subsample(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.tokens_per_frame > 1:
            hidden = hidden[:, :: self.tokens_per_frame, :]
        return hidden

    def _encode_and_subsample(self, input_features: torch.Tensor):
        model_dtype = next(self.model.parameters()).dtype
        input_features = input_features.to(dtype=model_dtype)

        base = self.model
        whisper_model = getattr(base, "model", base)
        encoder = getattr(whisper_model, "model", whisper_model).encoder \
                if hasattr(whisper_model, "model") \
                else whisper_model.encoder

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
        base_model  = (
            self.model.get_base_model()
            if hasattr(self.model, "get_base_model")
            else self.model
        )
        original = getattr(base_model, "_maybe_reduce_batch", None)
        if original is not None:
            def noop_reduce_batch(input_features=None, cur_bsz=None,
                                  batch_idx_map=None, **kw):
                return (input_features, cur_bsz, batch_idx_map)
            base_model._maybe_reduce_batch = noop_reduce_batch
        try:
            return self.model.generate(encoder_outputs=encoder_out, **kwargs)
        finally:
            if original is not None:
                base_model._maybe_reduce_batch = original

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
        info["hf_path"], config, split=split,
        streaming=streaming, trust_remote_code=True,
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
    Converts raw audio + transcript to model inputs.

    xT: total_frames clips audio (fewer active encoder frames).
    xV: tokens_per_frame applied in model.forward() via the wrapper.
    """
    audio       = batch["audio"]
    audio_array = np.array(audio["array"], dtype=np.float32)
    audio_sr    = audio["sampling_rate"]

    if audio_sr != sampling_rate:
        waveform    = torch.tensor(audio_array).unsqueeze(0)
        resampler   = torchaudio.transforms.Resample(audio_sr, sampling_rate)
        audio_array = resampler(waveform).squeeze(0).numpy().astype(np.float32)

    max_audio_samples   = int((total_frames / 1500) * 30 * sampling_rate)
    audio_array         = audio_array[:max_audio_samples]

    batch["input_features"] = processor.feature_extractor(
        audio_array, sampling_rate=sampling_rate,
    ).input_features[0]

    transcript = batch[text_column]
    if isinstance(transcript, str):
        transcript = transcript.lower()

    batch["labels"] = processor.tokenizer(
        transcript, max_length=max_label_len, truncation=True,
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
        )
    else:
        dataset = dataset.map(
            map_fn,
            remove_columns=dataset.column_names,
            num_proc=1,
            writer_batch_size=50,
            desc="Preprocessing",
        )
    return dataset


@dataclass
class WhisperDataCollator:
    processor: Any
    fp16: bool = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        # NOTE: input_features are deliberately kept fp32 here regardless of
        # self.fp16. Training always loads the model at fp32 (Trainer's own
        # fp16=True flag drives autocast for mixed-precision compute — see
        # build_model_full/build_model_lora), and Trainer's training_step
        # correctly wraps the forward/backward pass in autocast, so an
        # fp16 input reconciles fine there. But Seq2SeqTrainer's eval-time
        # model.generate() call (in prediction_step) is NOT reliably
        # wrapped in that same autocast context in this transformers
        # version — feeding it fp16 input against the model's genuinely
        # fp32 conv1/conv2 weights crashes with "Input type (c10::Half)
        # and bias type (float) should be the same" right at the first
        # eval boundary. Keeping input_features fp32 always avoids this
        # entirely, in both train and eval, at negligible memory cost
        # (this tensor is tiny next to activations/gradients).

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(
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
        return re.sub(r"\s+", " ", text)

    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids

        # ── guard against -100 or any out-of-range token IDs ──────────────────
        pred_ids  = np.where(pred_ids  < 0, processor.tokenizer.pad_token_id, pred_ids)
        label_ids = np.where(label_ids < 0, processor.tokenizer.pad_token_id, label_ids)

        pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_str  = [normalize(p) for p in pred_str]
        label_str = [normalize(r) for r in label_str]
        wer = jiwer_wer(label_str, pred_str)
        return {"wer": round(100 * wer, 4)}

    return compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM TRAINER — sparse mask updates + PEFT safe save
# ─────────────────────────────────────────────────────────────────────────────

class SparseAwarePeftTrainer(Seq2SeqTrainer):
    """
    Seq2SeqTrainer with two extensions:

    1. N:M sparse mask updates (xP axis):
       For 1:4 sparsity, NMSparseCallback.on_step_end() is called after
       each training step during the iterative ramp-in phase, recomputing
       magnitude-based masks on the current weights.

    2. PEFT-safe checkpoint saving:
       Unwraps WhisperWithTokenSubsampling and uses PeftModel.save_pretrained()
       to avoid safetensors shared-memory tensor crash from tied proj_out /
       embed_tokens weights.
    """

    def __init__(self, *args, sparse_callback: Optional[NMSparseCallback] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.sparse_callback = sparse_callback

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        # Update 1:4 sparsity mask during ramp-in phase
        if self.sparse_callback is not None:
            step = self.state.global_step
            self.sparse_callback.on_step_end(model, step)
        return loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        model = self.model
        if isinstance(model, WhisperWithTokenSubsampling):
            model = model.model

        if isinstance(model, PeftModel):
            model.save_pretrained(output_dir)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)
        else:
            print(f"  [save] Full finetune detected — using save_pretrained() "
                f"to handle tied proj_out/embed_tokens weights safely.")
            model.save_pretrained(output_dir, safe_serialization=True)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None):
        """
        Mirrors the unwrap in _save(): our checkpoints on disk always hold
        FLAT, single-prefix state dicts (WhisperWithTokenSubsampling's
        inner model, or the PeftModel's base+adapter), because _save()
        deliberately unwraps before writing — so that checkpoints stay
        directly loadable by inference2.py's plain
        WhisperForConditionalGeneration.from_pretrained().

        The base Trainer._load_from_checkpoint() doesn't know about that:
        it loads the on-disk (flat) state dict straight into `self.model`,
        which — whenever tokens_per_frame>1 wraps the model in
        WhisperWithTokenSubsampling — expects DOUBLE-prefixed keys
        ("model.model.encoder..."). Every key mismatches (you're seeing
        this as "missing keys" for the double-prefixed names and, if QAT
        is active, the fake_quant observer buffers underneath them too —
        same mismatch, just also touching the FakeQuantize submodules'
        own buffers since they live inside the same shifted namespace).
        This previously hit load_best_model_at_end (fixed by disabling
        that flag and copying files explicitly) and, now that
        resume_from_checkpoint is actually being passed through, hits
        resume too — same root cause, different call site, needs the
        same unwrap treatment.
        """
        target = model if model is not None else self.model
        if isinstance(target, WhisperWithTokenSubsampling):
            target = target.model
        super()._load_from_checkpoint(resume_from_checkpoint, model=target)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL SETUP
# ─────────────────────────────────────────────────────────────────────────────

def _wrap_if_subsampling(model: nn.Module, tokens_per_frame: int) -> nn.Module:
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
    sparsity_pattern: str  = "dense",
    qat_mode:         str  = "none",
) -> nn.Module:
    """
    All weights trainable. Applies xP sparsity then xQ QAT, then xV wrapper.

    NOTE: mode=full is the one place the OLD prune-before-training path
    still applies as-is. Because the base is fully trainable here (not
    frozen like in LoRA mode), the mask genuinely needs to interact with
    gradient updates throughout training — there isn't a clean
    "post-hoc prune a converged model" story the way there is for LoRA,
    since a full finetune already touches every weight, sparse or not.
    """
    # TRAINING loads at fp32 regardless of --fp16. The Trainer's own
    # fp16=True (set below via Seq2SeqTrainingArguments) drives real
    # torch.cuda.amp mixed precision: autocast casts activations to fp16
    # under the hood for the forward/backward pass while GradScaler
    # expects fp32 MASTER weights/gradients to unscale against. Loading
    # literal torch.float16 weights here as well double-applies fp16 and
    # crashes with "Attempting to unscale FP16 gradients" the moment
    # gradient clipping runs, because gradients w.r.t. fp16 parameters
    # are themselves fp16, which GradScaler explicitly refuses to
    # unscale. --fp16 still fully controls training precision — it just
    # now does so the way Trainer's AMP is designed to be driven.
    dtype = torch.float32
    print(f"Loading {model_name} (full finetune)...")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype
    )
    model.config.forced_decoder_ids         = None
    model.generation_config.suppress_tokens = []
    model.config.use_cache                  = False

    # xP: sparsify base weights BEFORE any adaptation
    if sparsity_pattern != "dense":
        model = apply_nm_sparsity_oneshot(model, sparsity_pattern)

    # xQ: insert QAT observers (prepare_qat requires train mode)
    if qat_mode != "none":
        model.train()
        model = apply_qat(model, qat_mode)

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
    qat_mode:         str   = "none",
) -> nn.Module:
    """
    Build a DENSE LoRA model (no sparsity applied here anymore).

    Sparsity (xP) for LoRA runs is now applied post-hoc, after this dense
    model has finished training, via prune_and_recover(). This is the
    unified pruning strategy for both 2:4 and 1:4 (see module docstring):
    prune a converged checkpoint, then run a short recovery fine-tune,
    rather than pruning a randomly-initialized adapter's base before it
    has learned anything.

    Build order:
      1. Load base WhisperForConditionalGeneration
      2. Insert LoRA adapters via PEFT
      3. Apply QAT fake-quant observers (xQ) — QAT is independent of xP
         and can still be applied to a dense LoRA run if desired
      4. Wrap with token subsampling (xV)
    """
    # See build_model_full for why training always loads at fp32 and lets
    # the Trainer's fp16=True AMP flag (Seq2SeqTrainingArguments) drive
    # mixed precision instead of literal fp16 weights — otherwise LoRA's
    # adapter weights (which PEFT creates matching the base layer's
    # dtype) end up fp16 too, and GradScaler crashes trying to unscale
    # fp16 gradients during grad-norm clipping.
    dtype = torch.float32
    print(f"Loading {model_name} (LoRA r={lora_r}, alpha={lora_alpha}, "
          f"xQ={qat_mode}) [dense base — xP applied post-hoc if requested]...")

    model = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype
    )
    model.config.forced_decoder_ids         = None
    model.generation_config.suppress_tokens = []
    model.config.use_cache                  = False

    # ── LoRA adapters (dense base, dense adapters) ────────────────────────────
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.train()
    # ── xQ — QAT observers AFTER LoRA so they wrap the merged path ───────────
    if qat_mode != "none":
        model = apply_qat(model, qat_mode)

    # ── xV wrapper ────────────────────────────────────────────────────────────
    model = _wrap_if_subsampling(model, tokens_per_frame)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def _find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    Look for HF-style step checkpoints (checkpoint-500, checkpoint-1000, ...)
    inside output_dir, and return the path to the most advanced one, or
    None if no valid checkpoint exists.

    A checkpoint is considered valid only if it contains trainer_state.json
    AND either pytorch_model.bin/model.safetensors (full mode) or
    adapter_model.safetensors (LoRA mode) — this guards against the
    tied-weight safetensors crash leaving a half-written checkpoint folder.
    """
    if not os.path.isdir(output_dir):
        return None
    print(output_dir)
    candidates = []
    for entry in os.listdir(output_dir):
        if not entry.startswith("checkpoint-"):
            continue
        ckpt_path = os.path.join(output_dir, entry)
        if not os.path.isdir(ckpt_path):
            continue

        state_path = os.path.join(ckpt_path, "trainer_state.json")
        if not os.path.exists(state_path):
            continue   # incomplete save — trainer_state.json is written LAST

        has_weights = any(
            os.path.exists(os.path.join(ckpt_path, fname))
            for fname in (
                "adapter_model.safetensors", "adapter_model.bin",
                "model.safetensors", "pytorch_model.bin",
            )
        )
        if not has_weights:
            continue   # crashed before weights were written

        try:
            step = int(entry.split("-")[-1])
        except ValueError:
            continue
        candidates.append((step, ckpt_path))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _print_cache_files(dataset, label: str) -> None:
    """
    Report which on-disk arrow cache file(s) actually back this dataset
    object. Every non-streaming HF `Dataset` (never IterableDataset,
    which is streamed and doesn't write this kind of cache at all)
    exposes exactly this via `.cache_files` -- whether that entry points
    at a cache load_dataset()/map() just WROTE this run, or one it found
    and REUSED from a previous run, is the same field either way, so
    this also doubles as a quick way to tell "did this actually
    re-tokenize everything, or did it just reuse what's already there."
    """
    cache_files = getattr(dataset, "cache_files", None)
    if not cache_files:
        print(f"  [cache] {label}: no on-disk cache file (streaming, or "
              f"in-memory dataset).")
        return
    print(f"  [cache] {label}: {len(cache_files)} cache file(s):")
    for entry in cache_files:
        path = entry.get("filename", entry) if isinstance(entry, dict) else entry
        exists = os.path.exists(path) if isinstance(path, str) else "?"
        size_mb = (round(os.path.getsize(path) / (1024 ** 2), 1)
                   if isinstance(path, str) and exists is True else "?")
        print(f"    {path}  (exists={exists}, {size_mb} MB)")


def _load_and_preprocess_datasets(args):
    """Shared dataset-loading/preprocessing logic used by both train() and
    prune_and_recover(), so the recovery run sees exactly the same data
    pipeline the original dense run did."""
    benchmark  = getattr(args, "benchmark_dataset", "librispeech")
    bench_info = BENCHMARK_REGISTRY.get(benchmark, BENCHMARK_REGISTRY["librispeech"])

    train_split      = getattr(args, "train_split", None) or bench_info["default_train"]
    eval_split       = getattr(args, "eval_split",  None) or bench_info["default_eval"]
    text_column      = bench_info["text_column"]
    language         = bench_info["language"]
    whisper_task     = bench_info["task"]
    tokens_per_frame = getattr(args, "tokens_per_frame", None) or 1
    total_frames     = getattr(args, "total_frames",     None) or 1500

    streaming         = getattr(args, "streaming",         False)
    max_train_samples = getattr(args, "max_train_samples", None)
    max_eval_samples  = getattr(args, "max_eval_samples",  None)

    model_name = local_path[args.model_size]
    processor = WhisperProcessor.from_pretrained(
        model_name, language=language, task=whisper_task
    )

    train_dataset = load_benchmark_dataset(
        benchmark, train_split,
        streaming=streaming, max_samples=max_train_samples,
    )
    eval_dataset = load_benchmark_dataset(
        benchmark, eval_split,
        streaming=streaming, max_samples=max_eval_samples,
    )
    _print_cache_files(train_dataset, f"raw train ({train_split})")
    _print_cache_files(eval_dataset,  f"raw eval ({eval_split})")

    print("Preprocessing datasets...")
    train_dataset = apply_preprocessing(
        train_dataset, processor,
        text_column=text_column, max_label_len=args.max_label_len,
        tokens_per_frame=tokens_per_frame, total_frames=total_frames,
        num_proc=args.num_proc if not streaming else 1,
    )
    eval_dataset = apply_preprocessing(
        eval_dataset, processor,
        text_column=text_column, max_label_len=args.max_label_len,
        tokens_per_frame=tokens_per_frame, total_frames=total_frames,
        num_proc=args.num_proc if not streaming else 1,
    )
    _print_cache_files(train_dataset, "preprocessed train (.map() output)")
    _print_cache_files(eval_dataset,  "preprocessed eval (.map() output)")

    return processor, train_dataset, eval_dataset


def train(args):
    set_seed(getattr(args, "seed", 42))
    benchmark  = getattr(args, "benchmark_dataset", "librispeech")
    task       = getattr(args, "task",              "asr")
    bench_info = BENCHMARK_REGISTRY.get(benchmark, BENCHMARK_REGISTRY["librispeech"])

    train_split      = getattr(args, "train_split", None) or bench_info["default_train"]
    tokens_per_frame = getattr(args, "tokens_per_frame", None) or 1
    total_frames     = getattr(args, "total_frames",     None) or 1500
    # keep resolved values on args so downstream (datasets, run_name) agree
    args.tokens_per_frame = tokens_per_frame
    args.total_frames     = total_frames
    qat_mode         = getattr(args, "qat_mode",         "none")

    # xP STRATEGY:
    #   qat_mode == "none": UNCHANGED from before — normal train() calls
    #     always build DENSE; sparsity (if any) is applied entirely
    #     separately via --prune_after_lora afterward. Two independent
    #     CLI invocations, no calibration state to preserve across them.
    #   qat_mode != "none": this run may CONTINUE DIRECTLY (same process)
    #     into pruning + recovery after dense training finishes, keeping
    #     QAT's fake-quant calibration alive throughout, converting to a
    #     real deployable model only at the very end. This is required
    #     because fake-quant observers' calibration state only exists in
    #     the live model object — it is not persisted by a normal
    #     checkpoint save, so a QAT-active run can't be split across
    #     separate CLI invocations the way a no-QAT run can. See
    #     finalize_and_convert()/_run_recovery_training() docstrings.
    requested_sparsity = getattr(args, "sparsity_pattern", "dense") or "dense"
    if qat_mode == "none":
        if requested_sparsity != "dense":
            print(f"\n  [xP] NOTE: --sparsity_pattern '{requested_sparsity}' is "
                  f"IGNORED during train() (mode={args.mode}, qat_mode=none). "
                  f"Training proceeds DENSE; apply sparsity afterwards with:\n"
                  f"    python ASR/finetune.py --prune_after_lora "
                  f"--dense_checkpoint <this run's output dir> "
                  f"--sparsity_pattern {requested_sparsity} --recovery_steps 500\n")
    else:
        if requested_sparsity != "dense":
            print(f"\n  [xQ+xP] QAT + sparsity requested together: this run "
                  f"trains dense first (QAT active throughout), then "
                  f"CONTINUES DIRECTLY into pruning + recovery in this same "
                  f"process (QAT still active), converting to a real "
                  f"deployable model only at the very end. No separate "
                  f"--prune_after_lora invocation is used for this "
                  f"combination — see finalize_and_convert()'s docstring.")
    sparsity_pattern = "dense"  # the INITIAL build is always dense either
                                # way; pruning (if requested) always
                                # happens on a converged model, never at
                                # random init.

    streaming         = getattr(args, "streaming",         False)
    max_train_samples = getattr(args, "max_train_samples", None)

    model_name = local_path[args.model_size]
    run_name   = (
        f"whisper-{args.model_size}-{args.mode}-{args.lora_r}-{benchmark}-{task}"
        f"-tpf{tokens_per_frame}-tf{total_frames}"
        f"-xP{sparsity_pattern.replace(':', '')}"
        f"-xQ{qat_mode}"
    )
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    resume_from_checkpoint = None

    latest_ckpt = _find_latest_checkpoint(output_dir)
    if latest_ckpt is not None:
        resume_from_checkpoint = latest_ckpt
        print(f"\n  [resume] Found partial checkpoint: {latest_ckpt}")
        print(f"  [resume] Will resume training from this step.")
    else:
        print(f"\n  [fresh] No valid checkpoint found in {output_dir} — starting fresh.")
    print(f"\nRun              : {run_name}")
    print(f"Cache directory  : {CACHE_DIR}")
    print(f"Streaming mode   : {streaming}")
    print(f"[xN] model_size       = {args.model_size}")
    print(f"[xT] total_frames     = {total_frames}  "
          f"(active audio = {total_frames/1500*30:.1f}s)")
    print(f"[xV] tokens_per_frame = {tokens_per_frame}  "
          f"(decoder sees {1500//tokens_per_frame} encoder tokens)")
    print(f"[xP] initial build    = dense  "
          f"(requested final sparsity: {requested_sparsity}"
          + (f" — will continue into pruning+recovery after dense "
             f"training, same process" if requested_sparsity != "dense"
             and qat_mode != "none" else
             f" — use --prune_after_lora afterward" if requested_sparsity != "dense"
             else "") + ")")
    print(f"[xQ] qat_mode         = {qat_mode}  "
          f"({'active throughout training' + (' + recovery' if requested_sparsity != 'dense' else '') + ', converted at the end' if qat_mode != 'none' else 'dense / PTQ in Stage 3'})")
    if max_train_samples:
        print(f"Max train samples    : {max_train_samples}")

    processor, train_dataset, eval_dataset = _load_and_preprocess_datasets(args)

    # ── Build model ───────────────────────────────────────────────────────────
    # No sparse callback in train(): training is always dense now (see the
    # unified-xP note above); mask enforcement lives in prune_and_recover().
    sparse_callback = None
    if args.mode == "full":
        model = build_model_full(
            model_name, fp16=args.fp16,
            tokens_per_frame=tokens_per_frame,
            sparsity_pattern=sparsity_pattern,
            qat_mode=qat_mode,
        )
    elif args.mode == "lora":
        model = build_model_lora(
            model_name, lora_r=args.lora_r,
            lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            fp16=args.fp16, tokens_per_frame=tokens_per_frame,
            qat_mode=qat_mode,
        )
    else:
        raise ValueError(f"Unknown mode '{args.mode}'.")

    collator    = WhisperDataCollator(processor=processor, fp16=args.fp16)
    is_streaming = isinstance(train_dataset, IterableDataset)

    # ── Guard against greedy-decoding repetition loops during eval ─────────
    # Early in training the decoder hasn't reliably learned to emit EOS.
    # With plain greedy search (num_beams=1, no repetition guard) and a
    # generous max_length, it can get stuck re-predicting the same
    # token/phrase until the length budget is exhausted — producing
    # transcripts far longer than the reference and inflating WER past
    # 100% (or, in bad cases, into the thousands) via insertions, even
    # while teacher-forced eval_loss keeps improving normally. Capping
    # the generation budget for eval and adding a light repetition
    # penalty keeps a few pathological examples from dominating the
    # corpus-level WER while the model is still early in training.
    gen_cfg = model.generation_config if hasattr(model, "generation_config") \
              else model.model.generation_config
    gen_cfg.no_repeat_ngram_size = 3
    gen_cfg.repetition_penalty   = 1.3

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        seed=args.seed, data_seed=args.seed,
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
        generation_max_length=min(args.max_label_len, 225),
        generation_num_beams=getattr(args, "eval_num_beams", 1),
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=1,
        # NOTE: intentionally always False. HF Trainer's load_best_model_at_end
        # tries to reload the on-disk checkpoint straight into self.model —
        # but whenever tokens_per_frame>1 wraps the model in
        # WhisperWithTokenSubsampling, self.model expects DOUBLE-prefixed
        # keys ("model.model.encoder...") while our custom _save() (below)
        # deliberately UNWRAPS before saving, writing flat single-prefix
        # keys ("model.encoder...") so the checkpoint stays a standard,
        # inference2.py-loadable Whisper checkpoint. Every key mismatches,
        # the reload silently no-ops (missing/unexpected key warnings you
        # may have seen), and trainer.model in memory quietly keeps the
        # LAST step's weights instead of the best-eval-WER ones — while
        # still being saved as if it were the "best" checkpoint. Getting
        # the actual best checkpoint is instead handled explicitly below
        # via trainer.state.best_model_checkpoint, which is still tracked
        # correctly regardless of this flag.
        load_best_model_at_end=False,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_steps=args.log_steps,
        report_to=["tensorboard"],
        run_name=run_name,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        eval_accumulation_steps=1,
    )

    trainer = SparseAwarePeftTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=make_compute_metrics(processor),
        processing_class=processor.feature_extractor,
        sparse_callback=sparse_callback,
    )

    print(f"\n{'='*60}")
    print(f"  Run     : {run_name}")
    print(f"  Mode    : {args.mode} | Streaming: {is_streaming}")
    print(f"  xP      : {sparsity_pattern} (dense LoRA base; xP applied post-hoc for lora mode)")
    print(f"  xQ(QAT) : {qat_mode}")
    if not is_streaming:
        print(f"  Train   : {len(train_dataset):,} samples")
        print(f"  Eval    : {len(eval_dataset):,} samples")
    print(f"{'='*60}\n")

    t0 = time.time()
    # BUG FIX: resume_from_checkpoint was computed above and printed as if
    # honored ("[resume] Will resume training from this step"), but
    # trainer.train() was previously called with no arguments at all —
    # every prior "resumed" run actually silently restarted from the
    # pretrained base at step 0. This passes it through so optimizer
    # state, LR scheduler, and global_step correctly resume.
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except FileNotFoundError as e:
        # BUG IN THE ORIGINAL VERSION OF THIS GUARD, FIXED HERE: this
        # except clause was meant to catch ONE narrow thing — transformers'
        # own _finalize_training() cleanup crashing via os.path.samefile()
        # AFTER all real training steps already succeeded (see the long
        # explanation below). But FileNotFoundError is a generic exception
        # type, and this cluster has independently shown disk/NFS write
        # corruption manifesting as unpredictable low-level errors at
        # checkpoint-save boundaries (the "iostream error" / "unexpected
        # pos" crashes). Catching FileNotFoundError unconditionally meant
        # a GENUINE early interruption that happened to also raise
        # FileNotFoundError would be silently swallowed here, falsely
        # reported as "training already completed successfully", and then
        # train() would proceed to write experiment_cfg.json/run_meta.json
        # and copy a "best" checkpoint for a run that only trained a
        # fraction of max_steps — exactly what happened on a large-v3 run
        # that stopped at step 500 of a configured 4000, silently treated
        # as done. Only swallow this exception if training actually
        # reached its target step count; otherwise it's a real
        # interruption and must surface as one.
        if trainer.state.global_step < training_args.max_steps:
            print(f"\n  [ERROR] Training stopped at step "
                  f"{trainer.state.global_step} of {training_args.max_steps} "
                  f"and then hit a FileNotFoundError ({e}). This is NOT the "
                  f"benign post-completion cleanup crash this guard exists "
                  f"for — training was genuinely interrupted early. "
                  f"Re-raising rather than silently finalizing a partial "
                  f"run as if it were complete.")
            raise
        # transformers' OWN _finalize_training() cleanup (not our code) can
        # crash here: it calls os.path.samefile(checkpoint,
        # self.state.best_model_checkpoint) against every checkpoint dir it
        # finds, and raises if best_model_checkpoint's own directory was
        # already deleted from disk — which can legitimately happen across
        # multiple resume cycles (e.g. an earlier interrupted run's own
        # rotation swept it away before this resume even started, while
        # trainer_state.json still points to it). This is pure
        # post-training bookkeeping that runs AFTER every real training
        # step has already succeeded — by the time this fires, training
        # itself is done and current checkpoints (including the final
        # step's) are intact on disk. Log it and fall through to our own
        # best-checkpoint retrieval below, which already defensively
        # checks os.path.isdir() before trusting best_model_checkpoint and
        # falls back to saving current in-memory weights if it's missing.
        print(f"\n  [WARNING] transformers' internal post-training checkpoint "
              f"cleanup crashed ({e}) — training reached its full "
              f"{training_args.max_steps}-step target first, so this is "
              f"HF's own bookkeeping, not training itself. Continuing with "
              f"this run's own best-checkpoint retrieval.")
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/3600:.2f}h")

    if qat_mode == "none":
        # ── UNCHANGED: existing dense-only save path ──────────────────────────
        # Always retrieve the best-eval-WER checkpoint by copying its files
        # directly from disk (trainer.state.best_model_checkpoint is tracked
        # correctly whenever metric_for_best_model is set, independent of the
        # load_best_model_at_end flag above). This sidesteps needing to trust
        # an in-memory reload at all — safe for mode=lora (adapter-only state
        # dict, no wrapper prefix issue) AND mode=full (avoids the
        # WhisperWithTokenSubsampling prefix mismatch entirely, since it's a
        # pure file copy with no state_dict loading involved).
        best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
        if best_ckpt and os.path.isdir(best_ckpt) and os.path.abspath(best_ckpt) != os.path.abspath(output_dir):
            print(f"  Best checkpoint (wer): {best_ckpt}  (copying to {output_dir})")
            import shutil
            for fname in os.listdir(best_ckpt):
                src = os.path.join(best_ckpt, fname)
                dst = os.path.join(output_dir, fname)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
        else:
            print("  No best_model_checkpoint recorded (e.g. run too short for an "
                  "eval to complete) — saving current in-memory weights instead.")
            trainer.save_model(output_dir)

        processor.save_pretrained(output_dir)

        # Save full experiment config for downstream eval / paper tables
        experiment_cfg = {
            "tokens_per_frame":    tokens_per_frame,
            "total_frames":        total_frames,
            "model_size":          args.model_size,
            "mode":                args.mode,
            "sparsity_pattern":    sparsity_pattern,
            "qat_mode":            qat_mode,
            "lora_r":              args.lora_r if args.mode == "lora" else 0,
            "pruned_after_lora":   False,
        }
        with open(os.path.join(output_dir, "experiment_cfg.json"), "w") as f:
            json.dump(experiment_cfg, f, indent=2)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        meta = {
            "model_size":        args.model_size,
            "mode":              args.mode,
            "task":              task,
            "benchmark_dataset": benchmark,
            "tokens_per_frame":  tokens_per_frame,
            "total_frames":      total_frames,
            "sparsity_pattern":  sparsity_pattern,
            "qat_mode":          qat_mode,
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

    # ── qat_mode != "none" ───────────────────────────────────────────────────
    # NOTE: unlike the qat_mode=="none" path above, we do NOT copy the
    # historically-"best" checkpoint's files here. That trick only works
    # by trusting whatever's saved on disk, but a QAT-active model's
    # fake-quant calibration state only exists in the LIVE in-memory
    # object — reloading "best" weights from disk would mean
    # re-attaching fresh, uncalibrated fake-quant observers, throwing
    # away everything training just calibrated (the exact problem this
    # whole design avoids). Tradeoff: the final converted model reflects
    # the LAST training step, not necessarily the best-eval-WER step. In
    # practice this project's linear LR decay to ~0 by the final step
    # tends to keep the last step close to converged anyway.
    print(f"  [xQ] QAT-active run: continuing with the LIVE in-memory "
          f"model (last training step) rather than reloading a saved "
          f"'best' checkpoint from disk — see code comment for why.")

    if requested_sparsity == "dense":
        # This dense, QAT-active run IS the final deliverable.
        model, converted, convert_note = finalize_and_convert(model, qat_mode)
        _save_possibly_converted(model, output_dir, args.mode, converted)
        processor.save_pretrained(output_dir)

        experiment_cfg = {
            "tokens_per_frame":  tokens_per_frame,
            "total_frames":      total_frames,
            "model_size":        args.model_size,
            "mode":              args.mode,
            "sparsity_pattern":  "dense",
            "qat_mode":          qat_mode,
            "qat_converted":     converted,
            "lora_r":            args.lora_r if args.mode == "lora" else 0,
            "pruned_after_lora": False,
        }
        with open(os.path.join(output_dir, "experiment_cfg.json"), "w") as f:
            json.dump(experiment_cfg, f, indent=2)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        meta = {
            "model_size":        args.model_size,
            "mode":              args.mode,
            "task":              task,
            "benchmark_dataset": benchmark,
            "tokens_per_frame":  tokens_per_frame,
            "total_frames":      total_frames,
            "sparsity_pattern":  "dense",
            "qat_mode":          qat_mode,
            "qat_converted":     converted,
            "qat_convert_note":  convert_note,
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

    # ── requested_sparsity != "dense": continue directly into pruning +
    # recovery in THIS SAME PROCESS, keeping QAT active throughout. This
    # is the combination requested: dense (QAT-active) -> prune -> recover
    # (QAT still active) -> convert -> save, no separate CLI invocation.
    #
    # Resumability note: if this whole train() invocation crashes during
    # the recovery portion and is simply re-run with the same command,
    # the dense phase's OWN resume detection (top of this function) will
    # find its completed checkpoint and Trainer will correctly no-op
    # through it (global_step already >= max_steps) before reaching this
    # point again, then _run_recovery_training()'s OWN internal resume
    # detection picks up the recovery phase from wherever it left off.
    # Composing two independent resume points this way preserves
    # crash-recovery without extra scaffolding.
    recovery_steps = getattr(args, "recovery_steps", 500)
    postprune_run_name = (
        f"{run_name}-postprune-xP{requested_sparsity.replace(':', '')}"
        f"-recov{recovery_steps}-xQ{qat_mode}"
    )
    postprune_output_dir = os.path.join(args.output_dir, postprune_run_name)
    os.makedirs(postprune_output_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  CONTINUING INTO PRUNING + RECOVERY (same process, QAT active)")
    print(f"  xP pattern : {requested_sparsity}")
    print(f"  Recovery   : {recovery_steps} steps")
    print(f"  Output     : {postprune_output_dir}")
    print(f"{'='*60}\n")

    model, recovery_elapsed = _run_recovery_training(
        model, processor, train_dataset, eval_dataset, args,
        sparsity_pattern=requested_sparsity, qat_mode=qat_mode,
        recovery_steps=recovery_steps, output_dir=postprune_output_dir,
        run_name=postprune_run_name, dense_mode=args.mode,
        already_qat_prepared=True,
    )

    model, converted, convert_note = finalize_and_convert(model, qat_mode)
    _save_possibly_converted(model, postprune_output_dir, args.mode, converted)
    processor.save_pretrained(postprune_output_dir)

    experiment_cfg = {
        "tokens_per_frame":  tokens_per_frame,
        "total_frames":      total_frames,
        "model_size":        args.model_size,
        "mode":              args.mode,
        "sparsity_pattern":  requested_sparsity,
        "qat_mode":          qat_mode,
        "qat_converted":     converted,
        "lora_r":            args.lora_r if args.mode == "lora" else 0,
        "pruned_after_lora": True,
        "dense_checkpoint":  output_dir,
        "recovery_steps":    recovery_steps,
    }
    with open(os.path.join(postprune_output_dir, "experiment_cfg.json"), "w") as f:
        json.dump(experiment_cfg, f, indent=2)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    meta = {
        "model_size":           args.model_size,
        "mode":                 args.mode,
        "task":                 task,
        "benchmark_dataset":    benchmark,
        "tokens_per_frame":     tokens_per_frame,
        "total_frames":         total_frames,
        "sparsity_pattern":     requested_sparsity,
        "qat_mode":             qat_mode,
        "qat_converted":        converted,
        "qat_convert_note":     convert_note,
        "streaming":            is_streaming,
        "lora_r":               args.lora_r if args.mode == "lora" else None,
        "dense_checkpoint":     output_dir,
        "trainable_params":     trainable,
        "total_params":         total,
        "trainable_pct":        round(100 * trainable / total, 4),
        "dense_training_hours": round(elapsed / 3600, 3),
        "recovery_minutes":     round(recovery_elapsed / 60, 2),
    }
    with open(os.path.join(postprune_output_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))

    return postprune_output_dir


# ─────────────────────────────────────────────────────────────────────────────
# xP STAGE 1.5 — POST-TRAINING PRUNING + RECOVERY (unified strategy, both modes)
# ─────────────────────────────────────────────────────────────────────────────

def _run_recovery_training(
    model, processor, train_dataset, eval_dataset, args,
    sparsity_pattern, qat_mode, recovery_steps, output_dir, run_name,
    dense_mode, already_qat_prepared=False, device=None,
):
    """
    Shared pruning + recovery-training core, used by BOTH:
      - prune_and_recover() (standalone CLI, --prune_after_lora): model
        freshly loaded from a dense_checkpoint on disk; qat_mode is
        applied fresh here if requested (already_qat_prepared=False).
      - train()'s continuous xQ+xP path: model is the JUST-TRAINED live
        object from Stage 1, ALREADY QAT-prepared if qat_mode != "none"
        (already_qat_prepared=True) — apply_qat is skipped here since
        it's already active. Pruning mutates .weight.data in place, which
        works identically whether or not fake-quant wrappers are already
        attached (verified: QAT-prepared modules remain genuine
        nn.Linear instances and tolerate in-place weight mutation
        cleanly — there is no need to strip/reapply QAT around pruning).

    Accepts `model` either already wrapped in WhisperWithTokenSubsampling
    (the continuous path passes it straight from Stage 1, which already
    wraps it) or not yet wrapped (the standalone path builds it unwrapped
    and expects this function to wrap it) — detects and normalizes either
    way, always leaving the RETURNED model wrapped correctly for
    tokens_per_frame.

    Does NOT save or convert — the caller decides that (continuous-flow
    callers still need finalize_and_convert() afterward; standalone
    callers save independently).

    Returns (model, elapsed_seconds).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokens_per_frame = args.tokens_per_frame

    # Normalize: unwrap subsampling if the caller already wrapped it, so
    # pruning/QAT operate on the raw PeftModel / WhisperForConditionalGeneration
    # underneath — re-wrapped at the end regardless of entry state.
    if isinstance(model, WhisperWithTokenSubsampling):
        tokens_per_frame = model.tokens_per_frame
        model = model.model

    model.to(device)

    # Few-shot mask-update schedule per the paper's Table 2: matters for
    # 1:4 (10.6 vs 11.7 WER), not for 2:4 (4.3 vs 4.4) → one-shot for 2:4.
    # Only meaningful when weights are trainable (mode=full recovery);
    # in LoRA recovery PEFT freezes the base so the mask can't drift and
    # updates would be no-ops on unchanged magnitudes anyway.
    if dense_mode == "full" and sparsity_pattern == "1:4":
        mask_update_steps = min(ITERATIVE_MASK_STEPS, max(1, recovery_steps // 2))
        print(f"  [xP] Few-shot mask schedule: update every "
              f"{ITERATIVE_MASK_FREQ} steps for the first "
              f"{mask_update_steps} steps, then frozen.")
    else:
        mask_update_steps = 0   # one-shot: capture once, frozen throughout

    sparse_callback = NMSparseCallback(
        pattern=sparsity_pattern,
        mask_update_steps=mask_update_steps,
        mask_update_freq=ITERATIVE_MASK_FREQ,
    )

    # ── Prune (mask-only; works whether or not QAT fake-quant is already
    # attached — see docstring) ────────────────────────────────────────────
    if dense_mode == "lora":
        apply_nm_sparsity_oneshot(model.get_base_model(), sparsity_pattern,
                                  use_hardware_sparse=False)
    else:
        apply_nm_sparsity_oneshot(model, sparsity_pattern,
                                  use_hardware_sparse=False)
        trainable = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
        print(f"  [full] Recovery trains all {trainable/1e6:.1f}M parameters "
              f"under frozen-mask enforcement.")
    sparse_callback.capture_masks(model)

    # ── QAT: only apply fresh if not already prepared upstream ──────────────
    if qat_mode != "none" and not already_qat_prepared:
        # prepare_qat() asserts model.training — from_pretrained returns
        # the model in eval mode, which previously forced the manual
        # fallback path every time.
        model.train()
        model = apply_qat(model, qat_mode)

    model = _wrap_if_subsampling(model, tokens_per_frame)

    collator = WhisperDataCollator(processor=processor, fp16=args.fp16)

    gen_cfg = model.generation_config if hasattr(model, "generation_config") \
              else model.model.generation_config
    gen_cfg.no_repeat_ngram_size = 3
    gen_cfg.repetition_penalty   = 1.3

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        seed=args.seed, data_seed=args.seed,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=getattr(args, "recovery_learning_rate", None) or args.learning_rate,
        warmup_steps=min(getattr(args, "recovery_warmup_steps", 50), recovery_steps),
        lr_scheduler_type="linear",
        max_steps=recovery_steps,
        fp16=args.fp16 and torch.cuda.is_available(),
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=max(1, recovery_steps // 5),
        predict_with_generate=True,
        generation_max_length=min(args.max_label_len, 225),
        generation_num_beams=getattr(args, "eval_num_beams", 1),
        save_strategy="steps",
        save_steps=max(1, recovery_steps // 5),
        save_total_limit=1,
        load_best_model_at_end=False,   # streaming-safe; we save at end regardless
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_steps=min(args.log_steps, max(1, recovery_steps // 10)),
        report_to=["tensorboard"],
        run_name=run_name,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        eval_accumulation_steps=1,
    )

    resume_from_checkpoint = None
    _latest_recovery_ckpt = _find_latest_checkpoint(output_dir)
    if _latest_recovery_ckpt is not None:
        resume_from_checkpoint = _latest_recovery_ckpt
        print(f"\n  [resume] Found partial recovery checkpoint: "
              f"{_latest_recovery_ckpt}")
        print(f"  [resume] Will resume recovery from this step.")
    else:
        print(f"\n  [fresh] No partial recovery checkpoint in {output_dir} "
              f"— starting recovery fresh.")

    trainer = SparseAwarePeftTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=make_compute_metrics(processor),
        processing_class=processor.feature_extractor,
        sparse_callback=sparse_callback,
    )

    t0 = time.time()
    # Same guard as train() (and same bug fix — see its comment): only
    # swallow this exception if recovery genuinely reached its target
    # step count. Otherwise it's a real interruption, not benign
    # post-training cleanup noise, and must surface as one rather than
    # silently finalizing a partial recovery run as if it were complete.
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except FileNotFoundError as e:
        if trainer.state.global_step < training_args.max_steps:
            print(f"\n  [ERROR] Recovery stopped at step "
                  f"{trainer.state.global_step} of {training_args.max_steps} "
                  f"and then hit a FileNotFoundError ({e}). Re-raising "
                  f"rather than silently finalizing a partial run.")
            raise
        print(f"\n  [WARNING] transformers' internal post-training checkpoint "
              f"cleanup crashed ({e}) — recovery reached its full "
              f"{training_args.max_steps}-step target first, so this is "
              f"HF's own bookkeeping, not recovery training itself. "
              f"Continuing.")
    elapsed = time.time() - t0
    print(f"\nRecovery fine-tune complete in {elapsed/60:.1f} min "
          f"({recovery_steps} steps).")

    return trainer.model, elapsed


def prune_and_recover(args):
    """
    Take a converged DENSE checkpoint (LoRA or full-finetune — detected from
    the dense run's experiment_cfg.json "mode" field), apply N:M sparsity,
    and run a short recovery fine-tune (default 500 steps).

    mode=lora  : load pretrained base + trained adapter, prune the base
                 (PEFT keeps it frozen, so the mask survives automatically),
                 recovery trains only the adapter against the sparse base.
    mode=full  : load the trained weights straight from the dense run dir,
                 prune them, recovery trains ALL parameters with the frozen
                 mask re-enforced after every optimizer step (via
                 NMSparseCallback), preventing the optimizer from regrowing
                 pruned positions. For 1:4 a few-shot mask-update window
                 runs first (paper Table 2: few-shot helps 1:4, not 2:4).

    Requires: args.dense_checkpoint (the output_dir of a completed dense
    train() run), args.sparsity_pattern ("2:4" or "1:4"),
    args.recovery_steps.

    Eval compatibility — no changes needed downstream:
    mode=lora  : evaluate_checkpoint()/inference2 re-apply
                 apply_nm_sparsity_oneshot() to a freshly-loaded base
                 before merging the adapter (deterministic reconstruction,
                 since the base is frozen throughout recovery).
    mode=full  : the recovery checkpoint saves the entire model with sparse
                 weights baked in — the existing full-mode loaders read it
                 directly.
    """
    set_seed(getattr(args, "seed", 42))

    dense_checkpoint = args.dense_checkpoint
    assert dense_checkpoint and os.path.isdir(dense_checkpoint), \
        "--dense_checkpoint must point to a completed dense run directory (mode=lora or mode=full)."

    sparsity_pattern = args.sparsity_pattern
    assert sparsity_pattern in ("2:4", "1:4"), \
        "--sparsity_pattern must be '2:4' or '1:4' for prune_and_recover()."

    recovery_steps = getattr(args, "recovery_steps", 500)
    qat_mode       = getattr(args, "qat_mode", "none")

    # Pull config written by the original dense run so tpf/tf/model_size
    # line up with what the adapter was actually trained on.
    # experiment_cfg.json is written to the RUN ROOT by train(), not to
    # checkpoint-N subdirs — so if the user pointed at a checkpoint-N dir,
    # also look one level up.
    dense_cfg = {}
    for cand in (dense_checkpoint, os.path.dirname(dense_checkpoint.rstrip("/"))):
        cfg_p = os.path.join(cand, "experiment_cfg.json")
        if os.path.exists(cfg_p):
            with open(cfg_p) as f:
                dense_cfg = json.load(f)
            print(f"  [cfg] Loaded dense run config from: {cfg_p}")
            break
    if not dense_cfg:
        print("  [cfg] WARNING: no experiment_cfg.json found next to (or above) "
              "--dense_checkpoint. Falling back to CLI args — make sure "
              "--model_size / --tokens_per_frame / --total_frames match the "
              "dense run EXACTLY, or the adapter will be merged onto the "
              "wrong base (missing-adapter-keys warning + garbage WER).")

    # CRITICAL: inherit the base model identity from the dense run.
    # Loading a distil-small-trained adapter onto e.g. whisper-small
    # (argparse default) produces PEFT "missing adapter keys" warnings for
    # decoder layers 4..11 and a garbage model.
    cfg_model_size = dense_cfg.get("model_size")
    if cfg_model_size:
        if cfg_model_size != args.model_size:
            print(f"  [cfg] model_size: CLI/default '{args.model_size}' "
                  f"→ overriding with dense run's '{cfg_model_size}'.")
        args.model_size = cfg_model_size

    # NOTE: argparse defaults for tpf/tf are now None (see parse_args), so
    # `or` correctly defers to the dense run's config here.
    tokens_per_frame = getattr(args, "tokens_per_frame", None) \
        or dense_cfg.get("tokens_per_frame", 1)
    total_frames     = getattr(args, "total_frames", None) \
        or dense_cfg.get("total_frames", 1500)
    args.tokens_per_frame = tokens_per_frame
    args.total_frames     = total_frames
    # Mode of the DENSE run decides how we load and what trains during
    # recovery. Inherited from the dense run's config; CLI --mode is
    # ignored here to prevent mismatched assumptions.
    if "mode" in dense_cfg:
        dense_mode = dense_cfg["mode"]
    else:
        # experiment_cfg.json missing (e.g. an earlier run crashed in
        # transformers' own post-training cleanup before reaching the code
        # that writes it — see train()'s FileNotFoundError guard). Blindly
        # defaulting to "lora" here silently misroutes full-finetune
        # checkpoints into PeftModel.from_pretrained(), which then fails
        # confusingly deep inside peft with "Can't find 'adapter_config.json'"
        # rather than any error that points at the actual problem. Instead,
        # detect directly from what's really in the checkpoint directory:
        # LoRA checkpoints have adapter_config.json; full-finetune
        # checkpoints have a plain config.json (and no adapter_config.json).
        _has_adapter_cfg = os.path.exists(
            os.path.join(dense_checkpoint, "adapter_config.json"))
        _has_plain_cfg = os.path.exists(
            os.path.join(dense_checkpoint, "config.json"))
        if _has_adapter_cfg:
            dense_mode = "lora"
        elif _has_plain_cfg:
            dense_mode = "full"
        else:
            raise ValueError(
                f"Cannot determine whether '{dense_checkpoint}' is a LoRA "
                f"or full-finetune checkpoint: experiment_cfg.json is "
                f"missing, and neither adapter_config.json nor config.json "
                f"was found in the directory. Pass --mode explicitly, or "
                f"verify --dense_checkpoint points at a complete run."
            )
        print(f"  [cfg] No experiment_cfg.json — detected mode='{dense_mode}' "
              f"from checkpoint contents "
              f"(adapter_config.json={_has_adapter_cfg}, "
              f"config.json={_has_plain_cfg}).")
    args.mode = dense_mode

    # Inherit lora_r for correct metadata (PEFT itself reads
    # adapter_config.json, but experiment_cfg.json feeds your tables).
    if dense_mode == "lora" and dense_cfg.get("lora_r"):
        args.lora_r = dense_cfg["lora_r"]

    model_name = local_path[args.model_size]
    # If pointed at a checkpoint-N subdir, name the new run after the
    # parent run dir so it still matches inference2's CHECKPOINT_NAME_RE.
    _base_name = os.path.basename(dense_checkpoint.rstrip("/"))
    if _base_name.startswith("checkpoint-"):
        _base_name = os.path.basename(os.path.dirname(dense_checkpoint.rstrip("/")))
    run_name = (
        f"{_base_name}"
        f"-postprune-xP{sparsity_pattern.replace(':', '')}"
        f"-recov{recovery_steps}-xQ{qat_mode}"
    )
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  POST-TRAINING PRUNING + RECOVERY")
    print(f"  Dense checkpoint : {dense_checkpoint}")
    print(f"  xP pattern       : {sparsity_pattern}")
    print(f"  Recovery steps   : {recovery_steps}")
    print(f"  xQ (QAT)         : {qat_mode}")
    print(f"  Output           : {output_dir}")
    print(f"{'='*60}\n")

    processor, train_dataset, eval_dataset = _load_and_preprocess_datasets(args)

    # ── Load the converged model (fresh from disk; not yet QAT-prepared) ─────
    # Both branches load fp32 (see build_model_full: Trainer's fp16=True
    # AMP flag drives mixed precision; literal fp16 weights break
    # GradScaler).
    dtype  = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dense_mode == "lora":
        base = WhisperForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype
        )
        base.config.forced_decoder_ids         = None
        base.generation_config.suppress_tokens = []
        base.config.use_cache                  = False
        base.to(device)
        # is_trainable=True: do NOT merge yet — recovery fine-tuning still
        # needs a live, separate adapter to update. merge_and_unload() only
        # happens at save time (or in evaluate_checkpoint()).
        model = PeftModel.from_pretrained(base, dense_checkpoint,
                                          is_trainable=True)
        model.to(device)

    else:  # dense_mode == "full"
        # Full-finetune dense checkpoints save the ENTIRE model via
        # save_pretrained() into the run root (config.json +
        # model.safetensors) — load the trained weights directly from
        # there, NOT from the pretrained hub path.
        _load_dir = dense_checkpoint
        if not os.path.exists(os.path.join(_load_dir, "config.json")):
            _parent = os.path.dirname(dense_checkpoint.rstrip("/"))
            if os.path.exists(os.path.join(_parent, "config.json")):
                _load_dir = _parent
        print(f"  [full] Loading converged full-finetune weights from: "
              f"{_load_dir}")
        model = WhisperForConditionalGeneration.from_pretrained(
            _load_dir, torch_dtype=dtype
        )
        model.config.forced_decoder_ids         = None
        model.generation_config.suppress_tokens = []
        model.config.use_cache                  = False
        model.to(device)

    # ── Prune + recover (shared core — see its docstring) ────────────────────
    model, elapsed = _run_recovery_training(
        model, processor, train_dataset, eval_dataset, args,
        sparsity_pattern=sparsity_pattern, qat_mode=qat_mode,
        recovery_steps=recovery_steps, output_dir=output_dir,
        run_name=run_name, dense_mode=dense_mode,
        already_qat_prepared=False, device=device,
    )

    # ── Finalize: convert fake-quant simulation into a genuinely quantized
    # (INT8) or QAT-robustified-float (INT4/none) deployable model. Must
    # happen HERE, on the live in-memory model, before any save — the
    # observers' calibration state does not survive a checkpoint save.
    model, converted, convert_note = finalize_and_convert(model, qat_mode)

    # ── Save ──────────────────────────────────────────────────────────────────
    _save_possibly_converted(model, output_dir, dense_mode, converted)
    processor.save_pretrained(output_dir)

    experiment_cfg = {
        "tokens_per_frame":  tokens_per_frame,
        "total_frames":      total_frames,
        "model_size":        args.model_size,
        "mode":              dense_mode,
        "sparsity_pattern":  sparsity_pattern,
        "qat_mode":          qat_mode,
        "qat_converted":     converted,
        "lora_r":            args.lora_r if dense_mode == "lora" else 0,
        # kept under the legacy key name for downstream compatibility;
        # semantically "pruned post-training" for either mode
        "pruned_after_lora": True,
        "dense_checkpoint":  dense_checkpoint,
        "recovery_steps":    recovery_steps,
    }
    with open(os.path.join(output_dir, "experiment_cfg.json"), "w") as f:
        json.dump(experiment_cfg, f, indent=2)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    meta = {
        "model_size":        args.model_size,
        "mode":              dense_mode,
        "tokens_per_frame":  tokens_per_frame,
        "total_frames":      total_frames,
        "sparsity_pattern":  sparsity_pattern,
        "qat_mode":          qat_mode,
        "qat_converted":     converted,
        "qat_convert_note":  convert_note,
        "lora_r":            args.lora_r if dense_mode == "lora" else 0,
        "pruned_after_lora": True,
        "dense_checkpoint":  dense_checkpoint,
        "recovery_steps":    recovery_steps,
        "trainable_params":  trainable,
        "total_params":      total,
        "trainable_pct":     round(100 * trainable / total, 4),
        "recovery_minutes":  round(elapsed / 60, 2),
    }
    with open(os.path.join(output_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))

    return output_dir


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s\']", "", text)
    return re.sub(r"\s+", " ", text)


def evaluate_checkpoint(args):
    """
    Load a saved checkpoint and run WER + RTF on test splits.

    Reads sparsity_pattern and qat_mode from experiment_cfg.json.

    IMPORTANT (LoRA + sparsity fix):
    In LoRA mode, PeftModel.save_pretrained() only saves the trainable
    adapter (lora_A/lora_B) — it does NOT save the frozen base model,
    since PEFT's whole design assumes the base can be reloaded from the
    original pretrained checkpoint unmodified. But when xP sparsity was
    applied, the base WAS modified in-memory (weights zeroed) — whether
    that happened before the (now legacy) dense-LoRA training or, in the
    current unified strategy, after it during prune_and_recover(). If we
    reload a fresh, unpruned base here and merge the trained adapter onto
    it, we get an inconsistent Frankenstein model.

    Fix: since sparsification is a deterministic function of the base
    weights (magnitude-based) and the base is frozen throughout LoRA
    training AND recovery (never gradient-updated), re-applying the exact
    same apply_nm_sparsity_oneshot() call to the freshly-loaded base
    reconstructs the identical sparse structure the adapter was actually
    trained against, before the adapter is merged on top of it. This
    works identically regardless of when in the pipeline pruning
    happened, since it depends only on the base checkpoint's weight
    values, not on training history.
    """
    set_seed(getattr(args, "seed", 42))
    checkpoint_dir = args.checkpoint
    assert checkpoint_dir, "--checkpoint required for eval_only mode"

    cfg_path = os.path.join(checkpoint_dir, "experiment_cfg.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            saved_cfg = json.load(f)
        print(f"Loaded experiment config: {saved_cfg}")
    else:
        saved_cfg = {}

    tokens_per_frame = getattr(args, "tokens_per_frame", None) \
                       or saved_cfg.get("tokens_per_frame", 1)
    total_frames     = getattr(args, "total_frames", None) \
                       or saved_cfg.get("total_frames", 1500)
    sparsity_pattern = saved_cfg.get("sparsity_pattern", "dense")
    qat_mode         = saved_cfg.get("qat_mode", "none")
    pruned_after_lora = saved_cfg.get("pruned_after_lora", False)

    # Inherit the base model identity from the checkpoint's config —
    # merging an adapter onto a different base (e.g. distil-small adapter
    # onto whisper-small because --model_size was left at its default)
    # produces PEFT missing-adapter-keys warnings and garbage WER.
    cfg_model_size = saved_cfg.get("model_size")
    if cfg_model_size and cfg_model_size != args.model_size:
        print(f"[cfg] model_size: CLI/default '{args.model_size}' "
              f"→ overriding with checkpoint's '{cfg_model_size}'.")
        args.model_size = cfg_model_size

    # Inherit mode too when available (lora vs full).
    if saved_cfg.get("mode") and getattr(args, "mode", None) != saved_cfg["mode"]:
        args.mode = saved_cfg["mode"]

    print(f"[xT] total_frames     = {total_frames}")
    print(f"[xV] tokens_per_frame = {tokens_per_frame}")
    print(f"[xP] sparsity_pattern = {sparsity_pattern}  "
          f"({'re-applied to reloaded base (post-training-pruned checkpoint)' if pruned_after_lora else 're-applied to reloaded base — see fix note' if sparsity_pattern != 'dense' and getattr(args, 'mode', None) == 'lora' else 'baked into weights'})")
    print(f"[xQ] qat_mode         = {qat_mode}  (Stage 2 QAT; PTQ via inference_eval.py)")

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

    dtype = torch.float16 if getattr(args, "fp16", True) else torch.float32

    if args.mode == "lora":
        base  = WhisperForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype
        )
        base.config.forced_decoder_ids = None

        # ── Reconstruct the sparse base the adapter was actually trained
        # against, BEFORE attaching/merging the trained adapter. Without
        # this, sparsity is silently discarded at eval time and the
        # trained LoRA delta gets merged onto an inconsistent, unpruned
        # base (see docstring above). Deterministic regardless of whether
        # pruning happened pre- or post-training.
        if sparsity_pattern != "dense":
            # use_hardware_sparse=False: merge_and_unload() must add a dense
            # LoRA delta into these weights; a SparseSemiStructuredTensor
            # weight makes that crash or densify unpredictably. Mask-only
            # here; the merged q/v are (by design) no longer strictly N:M
            # ("sparse base + dense LoRA").
            base = apply_nm_sparsity_oneshot(base, sparsity_pattern,
                                             use_hardware_sparse=False)

        model = PeftModel.from_pretrained(base, checkpoint_dir)
        model = model.merge_and_unload()
    else:
        # Full finetune saves the entire model state (including whatever
        # the sparsified weights drifted to during training), so a plain
        # reload is already self-consistent — no re-application needed.
        model = WhisperForConditionalGeneration.from_pretrained(
            checkpoint_dir, torch_dtype=dtype
        )
    model.config.forced_decoder_ids = None

    if tokens_per_frame > 1:
        model = WhisperWithTokenSubsampling(model, tokens_per_frame)

    model = model.to(device).eval()

    # ── Load test splits ──────────────────────────────────────────────────────
    if benchmark == "librispeech":
        eval_splits = {
            "test_clean": load_benchmark_dataset(
                benchmark, "test", streaming=False, config_override="clean",
                max_samples=getattr(args, "max_eval_samples", None),
            )
        }
    else:
        eval_splits = {}
        for config, split in bench_info["default_test"]:
            eval_splits[f"{config}_{split}"] = load_benchmark_dataset(
                benchmark, split, streaming=True,
                max_samples=getattr(args, "max_eval_samples", None),
            )

    results  = {}
    fp16     = getattr(args, "fp16", True)
    max_audio_samples = int((total_frames / 1500) * 30 * 16_000)

    for split_name, dataset in eval_splits.items():
        print(f"\nEvaluating on {split_name} ...")
        all_preds, all_refs = [], []
        total_audio_s = 0.0
        total_infer_s = 0.0

        for sample in tqdm.tqdm(dataset):
            audio_array = sample["audio"]["array"].astype(np.float32)
            ref_text    = sample[text_column].lower().strip()
            audio_array = audio_array[:max_audio_samples]
            audio_dur_s = len(audio_array) / 16_000

            features = processor.feature_extractor(
                audio_array, sampling_rate=16_000, return_tensors="pt"
            ).input_features.to(device)
            if fp16:
                features = features.half()

            t0 = time.time()
            with torch.no_grad():
                pred_ids = model.generate(
                    features, num_beams=16, max_new_tokens=256,
                    no_repeat_ngram_size=3, repetition_penalty=1.3,
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
        print(f"  WER: {results[split_name]['wer']:.3f}%  RTF: {rtf:.4f}")

    total_params = sum(
        p.numel() for p in (
            model.model.parameters()
            if isinstance(model, WhisperWithTokenSubsampling)
            else model.parameters()
        )
    )
    summary = {
        "model_size":        args.model_size,
        "mode":              args.mode,
        "tokens_per_frame":  tokens_per_frame,
        "total_frames":      total_frames,
        "sparsity_pattern":  sparsity_pattern,
        "qat_mode":          qat_mode,
        "pruned_after_lora": pruned_after_lora,
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


# ─────────────────────────────────────────────────────────────────────────────
# SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(args):
    if getattr(args, "sweep_sparsity", None):
        print(f"\n  [sweep] NOTE: --sweep_sparsity '{args.sweep_sparsity}' is "
              f"IGNORED. Training sweeps build dense LoRA models; run "
              f"post-training pruning afterwards, e.g.:\n"
              f"    python ASR/finetune.py --prune_after_lora "
              f"--dense_checkpoint <run_dir> --sparsity_pattern "
              f"{args.sweep_sparsity} --recovery_steps 500\n")
    sizes      = args.sweep_sizes.split(",")
    modes      = args.sweep_modes.split(",")
    lora_ranks = (
        [int(p.strip()) for p in args.lora_rank_sweep.split(",")]
        if getattr(args, "lora_rank_sweep", None)
        else [getattr(args, "lora_r", "dense")]
    )
    qat_modes = (
        [q.strip() for q in args.sweep_qat.split(",")]
        if getattr(args, "sweep_qat", None)
        else [getattr(args, "qat_mode", "none")]
    )

    all_results = []
    for size in sizes:
        for mode in modes:
            for rank in lora_ranks:
                for xq in qat_modes:
                    if mode == "full" and rank != args.lora_r:
                        continue
                    print(
                        f"\n{'#'*60}\n"
                        f"  SWEEP: whisper-{size} | mode={mode} | r={rank} "
                        f"| xQ={xq}  (dense — xP applied post-hoc separately "
                        f"for lora via --prune_after_lora)\n"
                        f"{'#'*60}"
                    )
                    args.model_size       = size
                    args.mode             = mode
                    args.lora_r           = rank
                    args.qat_mode         = xq

                    checkpoint_dir = train(args)
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                    eval_summary = evaluate_checkpoint(
                        argparse.Namespace(
                            model_size=size, mode=mode,
                            checkpoint=checkpoint_dir,
                            benchmark_dataset=getattr(args, "benchmark_dataset", "librispeech"),
                            max_eval_samples=getattr(args, "max_eval_samples", None),
                            fp16=getattr(args, "fp16", True),
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
            f"  whisper-{r['model_size']:<10s} | {r['mode']:<5s} | "
            f"tpf={r['tokens_per_frame']} tf={r['total_frames']} | "
            f"xQ={r.get('qat_mode','none'):<5s} | "
            f"trainable={r['trainable_params']:>12,} | "
            f"WER test_clean={r.get('test_clean', {}).get('wer', 'N/A')}%"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Whisper Finetuning: full vs LoRA, N:M sparsity (xP, post-hoc for LoRA), QAT (xQ)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Experiment identity
    p.add_argument("--task",              type=str, default="asr",
                   choices=SUPPORTED_TASKS)
    p.add_argument("--benchmark_dataset", type=str, default="librispeech",
                   choices=list(BENCHMARK_REGISTRY.keys()))
    # Defaults are None (NOT 1/1500) so that eval / prune_and_recover can
    # distinguish "user didn't pass it" from an explicit value and fall
    # back to the checkpoint's experiment_cfg.json. With a truthy default,
    # `getattr(args, ...) or cfg.get(...)` NEVER consulted the config —
    # e.g. a tpf=2-trained adapter was silently evaluated at tpf=1.
    p.add_argument("--tokens_per_frame",  type=int, default=None,
                   help="xV: encoder output stride. 1=all 1500 tokens, 2=750. "
                        "If unset, falls back to checkpoint cfg, then 1.")
    p.add_argument("--total_frames",      type=int, default=None,
                   help="xT: clip audio to (total_frames/1500)*30s. 750=15s. "
                        "If unset, falls back to checkpoint cfg, then 1500.")
    p.add_argument("--seed", type=int, default=42)

    # Model
    p.add_argument("--model_size", type=str, default="small",
                   choices=list(WHISPER_SIZES.keys()))
    p.add_argument("--mode",       type=str, default="lora",
                   choices=["full", "lora"])

    # xP: N:M sparsity
    p.add_argument("--sparsity_pattern", type=str, default="dense",
                   choices=VALID_SPARSITY_PATTERNS,
                   help=(
                       "xP axis — N:M semi-structured weight sparsity.\n"
                       "  dense → no sparsity (baseline)\n"
                       "  2:4   → 2 non-zeros per 4\n"
                       "  1:4   → 1 non-zero per 4\n"
                       "NOTE: for mode=lora this flag only matters when used\n"
                       "with --prune_after_lora (unified post-training pruning\n"
                       "strategy). A plain `train()` call with mode=lora always\n"
                       "builds a dense base; use --prune_after_lora afterward.\n"
                       "For mode=full, this is still applied pre-training\n"
                       "(legacy path — base is trainable so pre/post timing is\n"
                       "less meaningful there)."
                   ))

    # ── Post-training pruning + recovery (unified xP strategy for LoRA) ──────
    p.add_argument("--prune_after_lora", action="store_true",
                   help=(
                       "Run the post-training-pruning + recovery workflow "
                       "instead of a normal train()/sweep. Requires "
                       "--dense_checkpoint. Applies --sparsity_pattern "
                       "(2:4 or 1:4) to a converged dense LoRA checkpoint, "
                       "then fine-tunes for --recovery_steps to let the "
                       "adapter recover."
                   ))
    p.add_argument("--dense_checkpoint", type=str, default=None,
                   help="Path to a completed dense LoRA run's output_dir "
                        "(required with --prune_after_lora).")
    p.add_argument("--recovery_steps", type=int, default=500,
                   help="Number of fine-tune steps after pruning to let the "
                        "LoRA adapter recover. Also the 1:4 iterative mask "
                        "ramp-in window.")
    p.add_argument("--recovery_learning_rate", type=float, default=None,
                   help="Learning rate for the recovery phase. Defaults to "
                        "--learning_rate if unset.")
    p.add_argument("--recovery_warmup_steps", type=int, default=50,
                   help="Warmup steps for the recovery phase (capped at "
                        "--recovery_steps).")

    # xQ: QAT
    p.add_argument("--qat_mode", type=str, default="none",
                   choices=VALID_QAT_MODES,
                   help=(
                       "xQ axis — Quantization-Aware Training (Stage 2 only).\n"
                       "  none → no QAT (default); PTQ applied in inference_eval.py\n"
                       "  int8 → INT8 QAT (per-channel symmetric fake-quant on weights)\n"
                       "  int4 → INT4 QAT (per-tensor 4-bit fake-quant on weights)\n"
                       "Run QAT only on Pareto-optimal configs from Stage 1 (Option B)."
                   ))
    p.add_argument("--sweep_qat", type=str, default="int8,int4",help="Comma-separated QAT modes, e.g. none,int8,int4")
    p.add_argument("--sweep_sparsity", type=str, default=None,
                   help="ACCEPTED FOR CLI COMPATIBILITY BUT IGNORED during "
                        "training sweeps. Under the unified post-training "
                        "pruning strategy, sweeps always train DENSE LoRA "
                        "models; apply sparsity afterwards with "
                        "--prune_after_lora --sparsity_pattern 2:4|1:4.")

    # Data
    p.add_argument("--train_split",       type=str,  default=None)
    p.add_argument("--eval_split",        type=str,  default=None)
    p.add_argument("--max_label_len",     type=int,  default=448)
    p.add_argument("--num_proc",          type=int,  default=1)
    p.add_argument("--streaming",         action="store_true")
    p.add_argument("--max_train_samples", type=int,  default=None)
    p.add_argument("--max_eval_samples",  type=int,  default=None)

    # LoRA (xR axis)
    p.add_argument("--lora_r",          type=int,   default=32,
                   choices=[8, 16, 32, 64],
                   help="xR axis: LoRA rank. 0 = no LoRA (use mode=full instead).")
    p.add_argument("--lora_rank_sweep", type=str, default="32",
                   help="If set, run_sweep iterates over all lora_r values.")
    p.add_argument("--lora_alpha",      type=int,   default=64)
    p.add_argument("--lora_dropout",    type=float, default=0.05)

    # Training
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--eval_batch_size", type=int,   default=8)
    p.add_argument("--grad_accum",      type=int,   default=2)
    p.add_argument("--eval_num_beams",  type=int,   default=1,
                   help="Beam width for PERIODIC training-time eval "
                        "(WER used only for checkpoint selection/logging). "
                        "Beam search multiplies KV-cache + activation "
                        "memory roughly by this factor throughout the "
                        "whole run and was a contributing factor in "
                        "OOMs during long full-mode training runs. "
                        "Defaults to greedy (1); use inference2.py with "
                        "beam=4 for final reported WER after training.")
    p.add_argument("--learning_rate",   type=float, default=1e-5)
    p.add_argument("--warmup_steps",    type=int,   default=500)
    p.add_argument("--max_steps",       type=int,   default=4000)
    p.add_argument("--num_epochs",      type=int,   default=3)
    p.add_argument("--eval_steps",      type=int,   default=500)
    p.add_argument("--log_steps",       type=int,   default=25)
    p.add_argument("--fp16",            action="store_true", default=False)

    # Output
    p.add_argument("--output_dir", type=str, default="/scratch/zt1/project/msml604/user/mokshdag/checkpoints")

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
    if args.prune_after_lora:
        prune_and_recover(args)
    elif args.sweep:
        run_sweep(args)
    elif args.eval_only:
        evaluate_checkpoint(args)
    else:
        train(args)