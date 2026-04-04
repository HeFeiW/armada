# ManiSkill Trained Model Testing Guide

This guide explains:
1. where trained checkpoints are saved
2. how to load a checkpoint into ARMADA rollout
3. a minimal command to test in ManiSkill quickly

## 1) Where the trained model is

Training outputs are saved under:
- `outputs/<date>/<time>_train_diffusion_transformer_hybrid_dino_multi_gpu_maniskill_pick/`

For your latest run:
- `outputs/2026.04.04/20.14.11_train_diffusion_transformer_hybrid_dino_multi_gpu_maniskill_pick/checkpoints/latest.ckpt`
- `outputs/2026.04.04/20.14.11_train_diffusion_transformer_hybrid_dino_multi_gpu_maniskill_pick/checkpoints/epoch=0000-train_loss=0.538.ckpt`

`latest.ckpt` is usually the easiest choice for rollout testing.

## 2) How ARMADA uses the checkpoint

`RealEnvRunner` loads `checkpoint_path` and restores policy weights in:
- `armada/env_runner/real_env_runner.py`

Rollout config entry point:
- `armada/config/base_rollout.yaml`

The key fields for testing are:
- `checkpoint_path`: path to the `.ckpt` file
- `train_dataset_path`: training dataset root (contains `replay_buffer.zarr`)
- `save_buffer_path`: where rollout trajectories are saved
- `output_dir`: visualization / failure-detection outputs
- `env_backend`: set to `maniskill` for simulator testing
- `training`: should match your training config family (use `train_maniskill_poc` here)

## 3) Quick ManiSkill rollout test (recommended)

Run from repo root:

```bash
python armada/run_rollout.py --config-name base_rollout \
  training=train_maniskill_poc \
  env_backend=maniskill \
  checkpoint_path=/home/tmp_wanghf/armada/outputs/2026.04.04/20.14.11_train_diffusion_transformer_hybrid_dino_multi_gpu_maniskill_pick/checkpoints/latest.ckpt \
  train_dataset_path=/home/tmp_wanghf/armada/armada_data/maniskill_pick \
  save_buffer_path=/home/tmp_wanghf/armada/armada_data/maniskill_pick_rollout_poc \
  output_dir=/home/tmp_wanghf/armada/outputs_rollout/maniskill_poc
```

## 4) What to check after running

- New rollout data appears in `save_buffer_path`
- Output artifacts appear in `output_dir`
- Console logs show policy inference and environment stepping without checkpoint load errors

## 5) Practical notes

- If the run was interrupted early, checkpoint quality may be low (still fine for pipeline validation).
- For pure pipeline checks, `latest.ckpt` from an early epoch is acceptable.
- If rollout reports shape mismatch, ensure `training=train_maniskill_poc` is passed so runtime config matches the checkpoint family.
