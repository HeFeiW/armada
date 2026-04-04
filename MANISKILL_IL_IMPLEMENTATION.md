# ManiSkill PickCube IL Implementation Notes

## Design Principles

- Reuse ARMADA existing diffusion policy training workspace and dataset code.
- Keep ManiSkill integration at data collection and task config layers.
- Keep replay buffer schema compatible with existing `MyDataset`.

## Module Responsibilities

- `maniskill_armada/collect_data.py`
  - creates ManiSkill `PickCube-v1` env
  - runs heuristic policy to generate demonstrations
  - records image/state/action trajectories
  - writes ARMADA `replay_buffer.zarr`

- `maniskill_armada/heuristic_policy.py`
  - produces 8D ARMADA action: `[dx, dy, dz, dqw, dqx, dqy, dqz, gripper]`
  - provides conversion to ManiSkill control action (`to_maniskill_action`)

- `armada/diffusion_policy/diffusion_policy/dataset/mydataset.py`
  - reads `replay_buffer.zarr`
  - converts action/pose quaternion representation to `rotation_6d`
  - outputs model-ready batch with RGB and low-dim observations

- `armada/config/training/task/maniskill_pick.yaml`
  - declares `shape_meta`
  - binds dataset path and loader
  - provides `env_runner` required by workspace

- `armada/config/training/train_maniskill_poc.yaml`
  - reuses existing DINO diffusion workspace config
  - overrides task and practical POC training knobs

## Interfaces

1. Data collection output interface
- episode keys: `wrist_cam`, `side_cam`, `tcp_pose`, `joint_pos`, `action`

2. Dataset interface
- input: `.../replay_buffer.zarr`
- output: `{'obs': {'wrist_img', 'side_img', 'ee_pose'}, 'action'}` as tensors

3. Training interface
- command: `python train.py train_maniskill_poc` (from `armada/`)
- config entry: Hydra config name `train_maniskill_poc`

## Key Compatibility Decisions

- Preserve action representation in collected data as ARMADA 8D; perform representation conversion inside dataset loader.
- Keep `n_obs_steps > 1` behavior in `MyDataset` and set default to 2 to avoid invalid default initialization.
- Use config inheritance to minimize duplication and reduce maintenance cost.
