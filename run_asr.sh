#!/bin/bash
#SBATCH --job-name=asr_job
#SBATCH --output=asr_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2
source ~/myvenv/bin/activate

export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8

python run.py --task asr --benchmark_dataset librispeech --llm_size small --tokens_per_frame 16 --total_frames 160 --streaming