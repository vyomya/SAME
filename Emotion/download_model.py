from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import os

os.environ["TRANSFORMERS_CACHE"]    = "/scratch/zt1/project/msml604/user/mokshdag/hf_cache/models"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/scratch/zt1/project/msml604/user/mokshdag/hf_cache/hub"

# Create dirs if they don't exist
os.makedirs(os.environ["TRANSFORMERS_CACHE"],    exist_ok=True)
os.makedirs(os.environ["HUGGINGFACE_HUB_CACHE"], exist_ok=True)

model_name = "facebook/wav2vec2-large-robust"

AutoFeatureExtractor.from_pretrained(model_name)
AutoModelForAudioClassification.from_pretrained(model_name)

print("Done.")