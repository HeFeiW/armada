# ManiSkill + ARMADA Integration Summary

This document reflects the current implementation in code. If there is any mismatch with other docs, use code as the source of truth.

## Scope

- Task: `PickCube-v1` (ManiSkill)
- Goal: collect expert demonstrations and train an imitation learning diffusion policy in ARMADA
- Data format: ARMADA replay buffer (`replay_buffer.zarr`)

## Implemented Pipeline

1. Collect demonstrations
- Entry: `maniskill_armada/collect_data.py`
- Policy: `maniskill_armada/heuristic_policy.py`
- Conversion utilities: `maniskill_armada/data_utils.py`
- Output directory default: `armada_data/maniskill_pick`
- Replay buffer path: `armada_data/maniskill_pick/replay_buffer.zarr`

2. Load dataset for diffusion policy
- Loader: `armada/diffusion_policy/diffusion_policy/dataset/mydataset.py`
- Task config: `armada/config/training/task/maniskill_pick.yaml`
- Train config: `armada/config/training/train_maniskill_poc.yaml`

3. Train imitation learning policy
- Entry: `armada/train.py`
- Command:
  `cd armada && python train.py train_maniskill_poc`

## Current Data/Interface Contract

Each episode contains:
- `wrist_cam`: `(T, H, W, 3)`, `uint8`
- `side_cam`: `(T, H, W, 3)`, `uint8`
- `tcp_pose`: `(T, 7)`, `float32`
- `joint_pos`: `(T, 7)`, `float32`
- `action`: `(T, 8)`, `float32`

Training conversion in `MyDataset`:
- action: 8D (xyz + quaternion + gripper) -> 10D (`rotation_6d` + gripper)
- ee pose: quaternion -> `rotation_6d` (shape `[9]`)

## Confirmed Training Config Behavior

`train_maniskill_poc.yaml` now reuses the existing DINO diffusion workspace config and overrides:
- task: `maniskill_pick`
- training/runtime knobs for a POC run (epochs/batch size/logging mode)

`maniskill_pick.yaml` now provides required fields for workspace execution:
- `env_runner`
- dataset sequence params linked to workspace (`horizon`, `n_obs_steps`, `n_action_steps`)

## Collector CLI (Current)

Supported args in `collect_data.py`:
- `--output`
- `--num-envs`
- `--num-episodes`
- `--max-steps`
- `--no-zarr`
- `--seed`
- `--save-video`

Note: parameters like `--stage1-episodes/--stage2-episodes/--stage3-episodes` are not in current collector code.

## Minimal Runbook

1. Collect data:
`python maniskill_armada/collect_data.py --num-episodes 200 --save-video`

2. Train:
`cd armada && python train.py train_maniskill_poc`

3. Check outputs:
- checkpoints/logs under `armada/outputs/...`
- dataset at `armada_data/maniskill_pick/replay_buffer.zarr`
