"""
One-off migration for a converted (INT8) mode=full checkpoint saved before
the _save_possibly_converted() double-prefix fix (WhisperWithTokenSubsampling
re-wrapped before .state_dict(), producing "model.model.*" keys instead of
"model.*").

Self-contained -- does not import from finetune.py/inference2.py, since
those are structured as scripts (heavy top-level execution: env vars, cache
dirs) rather than clean importable libraries.

Usage:
    python fix_converted_checkpoint.py \
        --checkpoint_dir /fs/nexus-scratch/vyomwal5/checkpoints/paper_table/eval/whisper-medium-full-32-librispeech-asr-tpf2-tf750-xPdense-xQint8-postprune-xP24-recov1500-xQint8 \
        --model_size medium
"""
import argparse
import os
import shutil
import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration

LOCAL_PATH = {
    "small":    "/fs/nexus-scratch/vyomwal5/models/models/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d",
    "medium":   "/fs/nexus-scratch/vyomwal5/models/models/models--openai--whisper-medium/snapshots/abdf7c39ab9d0397620ccaea8974cc764cd0953e",
    "tiny":     "/fs/nexus-scratch/vyomwal5/models/models/models--openai--whisper-tiny/snapshots/169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
    "large-v3": "/fs/nexus-scratch/vyomwal5/models/models/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1",
    "distil-small":  "/fs/nexus-scratch/vyomwal5/models/distil-small",
    "distil-medium": "/fs/nexus-scratch/vyomwal5/models/distil-medium",
    "distil-large":  "/fs/nexus-scratch/vyomwal5/models/distil-large",
}


class WhisperWithTokenSubsampling(nn.Module):
    """Minimal stand-in matching finetune.py's/inference2.py's class just
    closely enough for state_dict()/load_state_dict() shape-matching --
    this script never calls forward()/generate(), only uses this for its
    module nesting shape."""
    def __init__(self, base_model, tokens_per_frame: int = 1):
        super().__init__()
        self.model            = base_model
        self.tokens_per_frame = tokens_per_frame


def quantize_dynamic_matching_skeleton(target: nn.Module) -> nn.Module:
    """MUST match finetune.py's finalize_and_convert() / inference2.py's
    _quantize_dynamic_matching_skeleton() exactly (proj_out / lora_A /
    lora_B excluded) or shapes won't line up. This checkpoint is mode=full
    (no LoRA), so the lora_A/lora_B exclusion is inert here but kept for
    consistency with the other two copies of this logic."""
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
    return torch.quantization.quantize_dynamic(
        target, qconfig_spec=qspec, dtype=torch.qint8,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--model_size", required=True, choices=list(LOCAL_PATH.keys()))
    ap.add_argument("--tokens_per_frame", type=int, default=2)
    args = ap.parse_args()

    broken_path = os.path.join(args.checkpoint_dir, "pytorch_model_quantized.bin")
    backup_path = broken_path + ".broken_backup"
    assert os.path.exists(broken_path), f"Not found: {broken_path}"

    print(f"Backing up original file to {backup_path}")
    shutil.copy2(broken_path, backup_path)

    print("Loading broken (double-prefixed) state dict ...")
    broken_sd = torch.load(broken_path, map_location="cpu")

    print("Rebuilding a WRAPPED skeleton matching how this was actually saved ...")
    inner_skeleton = WhisperForConditionalGeneration.from_pretrained(
        LOCAL_PATH[args.model_size], torch_dtype=torch.float32, low_cpu_mem_usage=False,
    )
    quantized_skeleton = quantize_dynamic_matching_skeleton(inner_skeleton)
    wrapped_skeleton = WhisperWithTokenSubsampling(quantized_skeleton, args.tokens_per_frame)

    missing, unexpected = wrapped_skeleton.load_state_dict(broken_sd, strict=False)
    if missing or unexpected:
        print(f"FAILED — this checkpoint doesn't match the expected wrapped "
              f"shape ({len(missing)} missing, {len(unexpected)} unexpected). "
              f"Not overwriting anything. Original untouched at {broken_path}; "
              f"backup also at {backup_path}.")
        if missing:
            print("  missing (first 5):", missing[:5])
        if unexpected:
            print("  unexpected (first 5):", unexpected[:5])
        return
    print("Loaded cleanly into the wrapped skeleton (0 missing, 0 unexpected).")

    print("Extracting the correctly-shaped (unwrapped) state dict for re-save ...")
    fixed_sd = wrapped_skeleton.model.state_dict()

    print("Verifying against a fresh UNWRAPPED skeleton (matches load_int8_converted_full) ...")
    verify_inner = WhisperForConditionalGeneration.from_pretrained(
        LOCAL_PATH[args.model_size], torch_dtype=torch.float32, low_cpu_mem_usage=False,
    )
    verify_skeleton = quantize_dynamic_matching_skeleton(verify_inner)
    missing2, unexpected2 = verify_skeleton.load_state_dict(fixed_sd, strict=False)
    if missing2 or unexpected2:
        print(f"FAILED final verification ({len(missing2)} missing, "
              f"{len(unexpected2)} unexpected) — NOT overwriting. "
              f"Original untouched at {broken_path}; backup at {backup_path}.")
        return
    print("Final verification passed (0 missing, 0 unexpected).")

    print(f"Overwriting {broken_path} with the corrected state dict ...")
    torch.save(fixed_sd, broken_path)
    print(f"Done. Original preserved at {backup_path} in case anything looks wrong.")


if __name__ == "__main__":
    main()