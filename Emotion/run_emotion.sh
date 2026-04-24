#!/bin/bash

BASE=/scratch/zt1/project/msml604/user/mokshdag/miniconda3/envs/same

export LD_LIBRARY_PATH=$BASE/lib/python3.11/site-packages/nvidia/nccl/lib:$BASE/lib:$BASE/lib/python3.11/site-packages/torch/lib:$BASE/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$BASE/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$BASE/lib/python3.11/site-packages/nvidia/npp/lib:$LD_LIBRARY_PATH

python run_emotion_experiment.py "$@"