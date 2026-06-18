from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import os

# Models to download
models = {
    "distil-small": "distil-whisper/distil-small.en",
    "distil-medium": "distil-whisper/distil-medium.en",
    "distil-large": "distil-whisper/distil-large-v3"
}

# Local base directory
base_dir = "/fs/nexus-scratch/vyomwal5/models/"

os.makedirs(base_dir, exist_ok=True)

for name, repo_id in models.items():
    save_path = os.path.join(base_dir, name)
    os.makedirs(save_path, exist_ok=True)

    print(f"Downloading {repo_id} → {save_path}")

    # Download model
    model = AutoModelForSpeechSeq2Seq.from_pretrained(repo_id)
    model.save_pretrained(save_path)

    # Download processor (tokenizer + feature extractor)
    processor = AutoProcessor.from_pretrained(repo_id)
    processor.save_pretrained(save_path)

print("All models downloaded successfully.")