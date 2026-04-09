# ARMADA to ManiSkill Integration Notes
This document summarizes the current state of ManiSkill integration in ARMADA, focusing on 

- [Environment setup](#environment-setup)
- [Data collection and training](#data-collection-and-training)
- [Human-in-loop rollout execution and debugging](#human-in-loop-rollout-execution-and-debugging)

## Environment Setup
- Follow the official ARMADA setup guide for base environment and dependencies.[ARMADA Setup Guide](./README.md#️-installation)
- Install ManiSkill and its dependencies as per the [ManiSkill Installation Guide](https://maniskill.github.io/maniskill-docs/installation.html).
- Additionally, you should set `setuptool` to version 81.0.0 or earlier to avoid compatibility issues with `pybullet-svl`(already included in the `conda_environment.yaml`, just in case you use the yaml file from original ARMADA repo which may have a newer `setuptools` version). And `protobuf==3.20.0` to adapt to the latest `wandb==0.22.3`.(Original ARMADA repo has earlier `wandb` version which does not support current 86-letters API key format.)

## Data Collection and Training
- Data collection for `PickCube-v1` is implemented in `maniskill_armada/collect_data.py`, which uses a heuristic policy defined in `maniskill_armada/heuristic_policy.py` to generate demonstrations. The collected data is stored in ARMADA's replay buffer format (`replay_buffer.zarr`).
- The dataset loader in `armada/diffusion_policy/diffusion_policy/dataset/mydataset.py` is modified to read the ManiSkill replay buffer and convert the action and pose representations to match the training requirements of the diffusion policy.
- Training is configured through `armada/config/training/train_maniskill_poc.yaml`, which inherits from the existing DINO diffusion workspace config and overrides necessary fields for the ManiSkill task.
- To collect data, run:
```bash
python maniskill_armada/collect_data.py --output /path/to/output_dir
```
- To train the policy, run:
```bash
python armada/train.py train_maniskill_poc
```
- If you want to customize the training config, you can override specific fields through command-line arguments or by creating a new YAML config that inherits from `train_maniskill_poc.yaml`.
- for detailed implementation notes and design decisions, refer to [MANISKILL_IL_IMPLEMENTATION.md](./MANISKILL_IL_IMPLEMENTATION.md).

## Human-in-loop Rollout Execution and Debugging
- The rollout execution for ManiSkill is handled by `armada/env_runner/real_env_runner.py`. This module integrates the trained policy with the ManiSkill environment and manages the human-in-loop interactions based on the configuration specified in `armada/config/maniskill_rollout.yaml`.
- To run a rollout, use the following command:
```bash
python armada/run_rollout.py --config-name maniskill_rollout maniskill.num_envs=1 # or more for parallel rollouts
```
- The rollout will save collected trajectories to the specified `save_buffer_path` and output visualization artifacts to `output_dir`.
- During human intervention (`teleop` mode), keyboard input now directly controls the end-effector 6D pose in simulator via `hardware/maniskill_robot_env.py`.

### Keyboard Teleop Controls (ManiSkill)
- Translation:
	- `W / S`: move end-effector in `+X / -X`
	- `A / Z`: move end-effector in `+Y / -Y`
	- `R / V`: move end-effector in `+Z / -Z`
- Rotation (delta orientation per keypress):
	- `U / J`: `+roll / -roll`
	- `I / K`: `+pitch / -pitch`
	- `O / L`: `+yaw / -yaw`
- Gripper:
	- `N`: close gripper
	- `M`: open gripper
- Intervention flow controls:
	- `C`: return control to policy inference
	- `F`: finish episode and save
	- `D`: discard episode
	- `Q`: quit rollout loop

### Teleop Configuration
- In `armada/config/maniskill_rollout.yaml` under `maniskill`:
	- `teleop_pos_step`: translation step size per keypress (meters)
	- `teleop_rot_step_deg`: rotation step size per keypress (degrees)
	- `teleop_gripper_step`: gripper width step per keypress (meters)
	- `teleop_show_help`: print one-time teleop key map when intervention starts
	- `teleop_use_cv2_keys`: use OpenCV window key polling when display is available

### Implementation Notes
 During teleop, each accepted human step appends: `wrist_cam`, `side_cam`, `tcp_pose`, `joint_pos`, `action`, and `action_mode=INTV` into the same episode trajectory buffer.
 In parallel ManiSkill rollout, teleop state is synchronized back into the episode manager on every accepted human step so policy rollout resumes from the corrected robot state.
 Teleop continues until user issues `C/F/D` (or timeout policy in `human_loop` if configured):
	- `C`: exit teleop and continue policy rollout from corrected state
	- `F`: finish and save current episode
	- `D`: discard current episode

### Headless / Remote Display Troubleshooting
- Symptom: Qt/XCB errors like `could not connect to display ...` and abort.
- Cause: OpenCV GUI backend cannot create windows for dashboard/key polling.
- Fix options:
	- Set `human_loop.visualize=false` when running on headless nodes.
	- Keep rollout running headless and use terminal teleop input fallback.
	- If you need GUI, run with a valid local display or proper X11/Wayland forwarding.

Example headless run:
```bash
python armada/run_rollout.py --config-name maniskill_rollout human_loop.visualize=false maniskill.num_envs=5
```
- For detailed adaption notes, configuration options, and debugging tips related to the human-in-loop rollout, refer to [MANISKILL_HIL_ADAPTATION.md](./MANISKILL_HIL_ADAPTATION.md).