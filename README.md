# Scaling Audio Models Efficiently — ASR Pipeline

**A Joint Study of Compute Constraints and Optimization Behavior**
Vyom Agarwal · Mokshda Gangrade · Group 25 · MSML604/605 · University of Maryland

---

## Table of Contents

1. [Overview](#overview)
2. [Environment Setup](#environment-setup)
3. [Running Experiments](#running-experiments)


---

## Overview

This pipeline fine-tunes OpenAI Whisper models on LibriSpeech using two configurable compute axes:

| Axis | Parameter | Controls |
|------|-----------|----------|
| **xT** (audio duration) | `--total_frames` | Seconds of audio fed to encoder (e.g. 1500 → 30s, 750 → 15s, 375 → 7.5s) |
| **xV** (encoder resolution) | `--tokens_per_frame` | Mel spectrogram subsampling stride (1 = full res, 2 = half, 4 = quarter) |

Both full fine-tuning and LoRA adaptation modes are supported.

---

## Environment Setup

### 1. Create a new Conda environment with Python 3.11

```bash
conda create -n asr_env python=3.11 -y
conda activate asr_env
```

### 2. Install all Python requirements

```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg

```bash
conda install -c conda-forge ffmpeg -y
```

### 4. Install the NVIDIA NPP CUDA 12 library

```bash
pip install --no-cache nvidia-npp-cu12
```

### 5. Set `LD_LIBRARY_PATH`

First, find the paths for your Conda environment and key site-packages:

```bash
# Find your conda env lib path
CONDA_ENV_LIB=$(python -c "import sys; print(sys.prefix)")/lib

# Find torch lib path
TORCH_LIB=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")/lib

# Find NVIDIA lib paths
CUDA_RUNTIME_LIB=$(python -c "import nvidia.cuda_runtime, os; print(os.path.dirname(nvidia.cuda_runtime.__file__))")/lib
CUDA_NVRTC_LIB=$(python -c "import nvidia.cuda_nvrtc, os; print(os.path.dirname(nvidia.cuda_nvrtc.__file__))")/lib
NPP_LIB=$(python -c "import nvidia.npp, os; print(os.path.dirname(nvidia.npp.__file__))")/lib
```

Then export `LD_LIBRARY_PATH` using those resolved paths:

```bash
export LD_LIBRARY_PATH=$CONDA_ENV_LIB:$TORCH_LIB:$CUDA_RUNTIME_LIB:$CUDA_NVRTC_LIB:$NPP_LIB:$LD_LIBRARY_PATH
```

> **Example (NEXUS cluster, user `vyomwal5`):** The resolved paths look like:
> ```
> /scratch/zt1/project/msml604/user/vyomwal5/anaconda3/envs/same/lib
> /scratch/zt1/project/msml604/user/vyomwal5/anaconda3/envs/same/lib/python3.11/site-packages/torch/lib
> /scratch/zt1/project/msml604/user/vyomwal5/anaconda3/envs/same/lib/python3.11/site-packages/nvidia/cuda_runtime/lib
> /scratch/zt1/project/msml604/user/vyomwal5/anaconda3/envs/same/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib
> /scratch/zt1/project/msml604/user/vyomwal5/anaconda3/envs/same/lib/python3.11/site-packages/nvidia/npp/lib
> ```

> **Tip:** Add the `export LD_LIBRARY_PATH=...` line to your `~/.bashrc` or SLURM job script so it persists across sessions.

---

## Running Experiments

### Standard single run (whisper-small, LoRA, full audio, full resolution)

```bash
python run.py \
  --task asr \
  --benchmark_dataset librispeech \
  --llm_size small \
  --tokens_per_frame 1 \
  --total_frames 448 \
  -- \
  --mode lora \
  --lora_r 16 \
  --max_steps 2500
```

### Full model size sweep (recommended)

Sweeps `tiny`, `small`, `medium`, and `large-v3` in LoRA mode in one command:

```bash
python run.py \
  --task asr \
  --benchmark_dataset librispeech \
  --llm_size small \
  --tokens_per_frame 1 \
  --total_frames 448 \
  -- \
  --sweep \
  --sweep_sizes small,medium,tiny,large-v3 \
  --sweep_modes lora
```

### Reduced audio duration (15s budget)

```bash
python run.py \
  --task asr \
  --benchmark_dataset librispeech \
  --llm_size small \
  --tokens_per_frame 1 \
  --total_frames 750 \
  -- \
  --mode lora \
  --lora_r 16 \
  --max_steps 2500
```

### Subsampled encoder resolution (half resolution)

```bash
python run.py \
  --task asr \
  --benchmark_dataset librispeech \
  --llm_size small \
  --tokens_per_frame 2 \
  --total_frames 1500 \
  -- \
  --mode lora \
  --lora_r 16 \
  --max_steps 2500
```

### Full fine-tuning baseline

```bash
python run.py \
  --task asr \
  --benchmark_dataset librispeech \
  --llm_size small \
  --tokens_per_frame 1 \
  --total_frames 1500 \
  -- \
  --mode full \
  --max_steps 2500
```

---

---

# Scaling Audio Models Efficiently — SER Pipeline

**Speech Emotion Recognition with wav2vec2 on CREMA-D**
Vyom Agarwal · Mokshda Gangrade · Group 25 · MSML604/605 · University of Maryland

---

## Table of Contents

1. [Overview](#overview-1)
2. [Environment Setup](#environment-setup-1)
3. [Running Experiments](#running-experiments-1)

---

## Overview

This pipeline fine-tunes `facebook/wav2vec2-large-robust` (and other wav2vec2/WavLM variants) on the CREMA-D dataset for 6-class speech emotion recognition, using a configurable audio-duration compute axis (xT).

| Axis | Parameter | Controls |
|------|-----------|----------|
| **xT** (audio duration) | `--max_audio_len` | Max clip length in seconds fed to the encoder (e.g. 2.0s, 4.0s, 6.0s) |
| **xN** (model size) | `--llm_name` | HuggingFace model ID — swap to compare wav2vec2-base, wav2vec2-large-robust, wavlm-large |

Both full fine-tuning and LoRA adaptation modes are supported. The pipeline uses a lazy PyTorch Dataset (no HF `.map()`) to avoid OOM at scale, and a three-group optimizer with separate learning rates for the classifier head, unfrozen encoder layers, and LoRA adapters.

---

## Environment Setup

### 1. Install all Python requirements

> If you have already created a Conda environment and installed requirements for Part A (ASR), you can reuse the same environment — the SER pipeline shares the same `requirements.txt`.

```bash
conda activate asr_env          # or your existing env
pip install -r requirements.txt
```

If setting up fresh:

```bash
conda create -n ser_env python=3.11 -y
conda activate ser_env
pip install -r requirements.txt
```

### 2. Prepare the CREMA-D dataset

CREMA-D WAV files should be placed on local cluster storage before training begins to avoid repeated network I/O. The pipeline loads audio directly from local paths on Zaratan HPC.

```bash
# Example: copy dataset to scratch space
cp -r /path/to/cremad_wavs /scratch/<your_project>/data/cremad/
```

Update the dataset path in `run_emotion_experiment.py` (or pass it via `--data_dir` if supported) to point to your local copy.

---

## Running Experiments

### Standard single run (wav2vec2-large-robust, LoRA, 4s audio)

This is the recommended starting point — 4s was found to be the optimal clip length for CREMA-D (best unweighted accuracy, faster than 6s).

```bash
python run_emotion_experiment.py \
  --benchmark_dataset cremad \
  --llm_name facebook/wav2vec2-large-robust \
  --max_audio_len 4.0 \
  -- \
  --mode lora \
  --max_steps 3000 \
  --eval_steps 300
```

### xT sweep — vary audio clip length

Sweeps 2s, 4s, and 6s to map the compute-accuracy tradeoff on the xT axis:

```bash
for LEN in 2.0 4.0 6.0; do
  python run_emotion_experiment.py \
    --benchmark_dataset cremad \
    --llm_name facebook/wav2vec2-large-robust \
    --max_audio_len $LEN \
    -- \
    --mode lora \
    --max_steps 3000 \
    --eval_steps 300
done
```

### xN sweep — vary model size (at optimal 4s clip)

```bash
for MODEL in facebook/wav2vec2-base facebook/wav2vec2-large-robust microsoft/wavlm-large; do
  python run_emotion_experiment.py \
    --benchmark_dataset cremad \
    --llm_name $MODEL \
    --max_audio_len 4.0 \
    -- \
    --mode lora \
    --max_steps 3000 \
    --eval_steps 300
done
```

### LoRA rank sweep (at optimal 4s clip)

```bash
for RANK in 8 16 32 64; do
  python run_emotion_experiment.py \
    --benchmark_dataset cremad \
    --llm_name facebook/wav2vec2-large-robust \
    --max_audio_len 4.0 \
    -- \
    --mode lora \
    --lora_r $RANK \
    --max_steps 3000 \
    --eval_steps 300
done
```

### Full fine-tuning baseline

```bash
python run_emotion_experiment.py \
  --benchmark_dataset cremad \
  --llm_name facebook/wav2vec2-large-robust \
  --max_audio_len 4.0 \
  -- \
  --mode full \
  --max_steps 3000 \
  --eval_steps 300
```


