#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=ser_sweep
#SBATCH --output=/home/mokshdag/SAME/sweep_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

cd /home/mokshdag/SAME

# ── xN: smaller model ──────────────────────────────────────────────────────
echo "=== xN: wav2vec2-base ==="
python run_emotion_experiment.py \
    --benchmark_dataset cremad \
    --llm_name facebook/wav2vec2-base \
    --max_audio_len 4.0 \
    -- --mode lora --max_steps 3000 --eval_steps 300 \
       --output_dir /home/mokshdag/checkpoints/

# ── LoRA rank sweep ────────────────────────────────────────────────────────
for RANK in 8 32 64; do
    ALPHA=$((RANK * 2))
    echo "=== LoRA rank: r=$RANK ==="
    python run_emotion_experiment.py \
        --benchmark_dataset cremad \
        --llm_name facebook/wav2vec2-large-robust \
        --max_audio_len 4.0 \
        -- --mode lora --lora_r $RANK --lora_alpha $ALPHA \
           --max_steps 3000 --eval_steps 300 \
           --output_dir /home/mokshdag/checkpoints/
done

# ── unfreeze_top_layers sweep ──────────────────────────────────────────────
for LAYERS in 0 8; do
    echo "=== unfreeze_top_layers=$LAYERS ==="
    python run_emotion_experiment.py \
        --benchmark_dataset cremad \
        --llm_name facebook/wav2vec2-large-robust \
        --max_audio_len 4.0 \
        -- --mode lora --unfreeze_top_layers $LAYERS \
           --max_steps 3000 --eval_steps 300 \
           --output_dir /home/mokshdag/checkpoints/
done

echo "=== ALL DONE ==="
