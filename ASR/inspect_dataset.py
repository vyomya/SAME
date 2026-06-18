# inspect_dataset.py
import numpy as np
from datasets import load_dataset, Audio

dataset = load_dataset(
    "librispeech_asr", "clean",
    split="validation",
    streaming=True,
    trust_remote_code=True,
)

print("Scanning for problematic samples...")
problems = []
for i, sample in enumerate(dataset):
    audio  = np.array(sample["audio"]["array"], dtype=np.float32)
    text   = sample.get("text", "").strip()
    max_amp = np.abs(audio).max()

    issues = []
    if not text:
        issues.append("empty transcript")
    if max_amp < 1e-4:
        issues.append(f"silent audio (max_amp={max_amp:.2e})")
    if len(audio) < 16000 * 0.5:
        issues.append(f"very short audio ({len(audio)/16000:.2f}s)")

    if issues:
        problems.append({
            "index": i,
            "id":    sample.get("id", "?"),
            "issues": issues,
            "text":  text[:50],
        })
        print(f"  Sample {i} | {sample.get('id','?')} | {issues} | '{text[:50]}'")

    if i % 500 == 0:
        print(f"  Scanned {i} samples, {len(problems)} problems found so far...")

    if i >= 5000:
        break

print(f"\nTotal problems found: {len(problems)} / {i+1} samples scanned")