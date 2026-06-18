# test_generate.py
import torch
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model
import sys
sys.path.insert(0, '/nfshomes/vyomwal5/SAME/ASR')
from finetune import WhisperWithTokenSubsampling

model_path = "/fs/nexus-scratch/vyomwal5/models/models/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("Loading model...")
processor = WhisperProcessor.from_pretrained(model_path, language="English", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_path)
model.config.forced_decoder_ids = None
model.config.use_cache = False

lora_cfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
                      target_modules=["q_proj", "v_proj"], bias="none")
model = get_peft_model(model, lora_cfg)
model = model.half().to(device)
model = WhisperWithTokenSubsampling(model, tokens_per_frame=2)
model.eval()
base = model.model.get_base_model() if hasattr(model.model, "get_base_model") else model.model
print(f"Base model type : {type(base)}")
print(f"Has _maybe_reduce_batch: {hasattr(base, '_maybe_reduce_batch')}")
# Print dtype info
print(f"Model dtype : {next(model.parameters()).dtype}")
print(f"Model device: {next(model.parameters()).device}")

def make_features(audio):
    return processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.half().to(device)

def run_test(name, features):
    print(f"\nTesting {name} | shape={features.shape} dtype={features.dtype}...")
    try:
        with torch.no_grad():
            out = model.generate(
                features,
                language="en",
                task="transcribe",
                num_beams=1,
                max_new_tokens=10,
            )
        print(f"  {name}: SUCCESS")
        return True
    except Exception as e:
        print(f"  {name}: FAILED — {e}")
        return False

# ── Test cases ────────────────────────────────────────────────────────────────

# 1. Single normal audio
run_test("single normal",
    make_features(np.random.randn(16000 * 10).astype(np.float32))
)

# 2. Batch of normal audio
run_test("batch normal x4",
    make_features(np.random.randn(16000 * 10).astype(np.float32)).repeat(4, 1, 1)
)

# 3. Single silent audio
run_test("single silent",
    make_features(np.zeros(16000 * 3, dtype=np.float32))
)

# 4. Batch of silent audio
run_test("batch silent x8",
    make_features(np.zeros(16000 * 3, dtype=np.float32)).repeat(8, 1, 1)
)

# 5. Mixed batch — most likely trigger for _maybe_reduce_batch
normal = make_features(np.random.randn(16000 * 5).astype(np.float32))
silent = make_features(np.zeros(16000 * 2, dtype=np.float32))
run_test("mixed batch (normal+silent x4)",
    torch.cat([normal, silent, normal, silent], dim=0)
)

# 6. Very short audio
run_test("very short audio (0.5s)",
    make_features(np.random.randn(16000 // 2).astype(np.float32))
)

# 7. Batch of very short audio
run_test("batch short x4",
    make_features(np.random.randn(16000 // 2).astype(np.float32)).repeat(4, 1, 1)
)

print("\n" + "="*50)
print("All tests complete.")