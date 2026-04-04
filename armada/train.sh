#!/bin/bash
{
    source "/path/to/miniconda3/etc/profile.d/conda.sh"; conda activate armada
    export WANDB_API_KEY=${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}
    export HYDRA_FULL_ERROR=1
    export CUDA_VISIBLE_DEVICES=0

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py train_diffusion_transformer_real_hybrid_dino_multi_gpu_workspace
    exit
}