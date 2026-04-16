# ManiSkill Human-in-the-Loop Adaptation Notes

## Goal
Replace real-world robot human-in-loop path with a ManiSkill-native flow while keeping rollout-facing API stable and reusing existing ARMADA modules.

## File Index
- Rollout config:
  - `armada/config/maniskill_rollout.yaml`
  - `armada/config/base_rollout.yaml` (fallback template)
- Runner integration:
  - `armada/env_runner/real_env_runner.py`
- ManiSkill env adapter:
  - `hardware/maniskill_robot_env.py`
- Human-in-loop decision module:
  - `armada/utils/maniskill_hil.py`
- Offline rollout renderer
  - `maniskill_armada/render_rollout_videos.py`
- Debug guide:
  - `MANISKILL_HIL_DEBUG_GUIDE.md`

## Quick Start
1. Prepare rollout config:
  - Use `armada/config/maniskill_rollout.yaml`.
   - Set valid paths for `checkpoint_path`, `train_dataset_path`, `save_buffer_path`, `output_dir`.

2. Command:
  ```python
  # run single-env rollout:
  python armada/run_rollout.py --config-name maniskill_rollout maniskill.num_envs=1
  # run parallel multi-env rollout(recommended for data collection):
  python armada/run_rollout.py --config-name maniskill_rollout maniskill.num_envs=4
  ```

3. Output check:
   - Collected trajectories are saved to `save_buffer_path/replay_buffer.zarr`.
   - Visualization/scene artifacts are saved under `output_dir/seed_<seed>/`.
   - run offline renderer to visualize saved trajectories:
   ```python
    python maniskill_armada/render_rollout_videos.py --replay-buffer /path/to/replay_buffer.zarr --output-dir /path/to/video_output
   ```

## Human-Loop Configuration
Human-in-loop behavior is configured in `armada/config/maniskill_rollout.yaml` under `human_loop`.

- `mode`
  - `auto`: no terminal input, use policy defaults.
  - `manual`: terminal prompt per intervention event.
- `on_failure`
  - default action when failure is reported.
  - suggested value: `teleop`, other options: `continue`, `discard`, `finish`.
- `on_timeout`
  - default action when env truncates or max step is reached.
  - suggested value: `teleop`, other options: `continue`, `discard`, `finish`.
- `prompt_timeout_s`
  - manual mode input timeout; fallback to configured default action.
- `max_teleop_steps`
  - upper bound of one intervention window.
- `max_blocked_rounds`
  - anti-deadlock guard for repeatedly blocked envs.
- `visualize`
  - whether to open the OpenCV dashboard.
- `window_name`
  - OpenCV window title.

## Manual Mode Usage
When `human_loop.mode=manual`, each intervention event prompts:
- `C` / `continue`: continue policy control.
- `T` / `teleop`: enter simulated teleop intervention.
- `D` / `discard`: drop current env trajectory.
- `F` / `finish`: finalize current env trajectory.

When `human_loop.visualize=true`, the rollout opens a local dashboard that shows:
- per-env side and wrist renders
- current step and rollout state
- pending error / timeout reason
- the key legend for decision making

The dashboard accepts the same keys directly in the window, so you can decide without switching back to the terminal.

Recommended run command:
- `python armada/run_rollout.py --config-name maniskill_rollout maniskill.num_envs=4 human_loop.mode=manual`

## Parallel Multi-Env Behavior
- Per-env isolation:
  - each env has its own buffer and timestep.
  - one env entering intervention does not stop other envs.
- Intervention release:
  - intervention ends when simulated teleop sets infer flag, or when anti-block limits are reached.
- Save behavior:
  - discarded env episodes are skipped.
  - non-discarded env episodes are written independently to replay buffer.

## Blocking / Error Handling Recommendations
1. If rollout seems stuck in intervention:
   - reduce `human_loop.max_teleop_steps`.
   - reduce `human_loop.max_blocked_rounds`.

2. If manual mode is unattended:
   - keep `prompt_timeout_s` finite.
   - set `on_failure` and `on_timeout` to deterministic fallback actions.

3. If you want fully unattended runs:
   - use `human_loop.mode=auto`.
   - prefer `on_failure=teleop` and `on_timeout=finish` or `teleop` based on your data policy.

## Minimal Recommended Settings
For stable first use:
- `maniskill.num_envs: 2`
- `human_loop.mode: auto`
- `human_loop.visualize: true`
- `human_loop.on_failure: teleop`
- `human_loop.on_timeout: teleop`
- `human_loop.max_teleop_steps: 64`
- `human_loop.max_blocked_rounds: 16`

Then scale `maniskill.num_envs` after confirming throughput and intervention behavior.

## What Changed
- Added a simulation-only human-in-loop decision controller:
  - `armada/utils/maniskill_hil.py`
  - Supports `auto` and `manual` decision modes.
  - Unified decisions: `continue`, `teleop`, `discard`, `finish`.
- Extended ManiSkill environment adapter for parallel rollout usage:
  - `hardware/maniskill_robot_env.py`
  - Added per-env indexed state access (`get_robot_state(env_idx=...)`).
  - Added batched stepping API (`deploy_action_batch(...)`) for true multi-env progression.
  - Kept original single-env API behavior for backward compatibility.
- Integrated parallel simulation path into existing runner (no external API break):
  - `armada/env_runner/real_env_runner.py`
  - Existing `run_rollout()` now auto-selects parallel path when:
    - `env_backend=maniskill`
    - `maniskill.num_envs > 1`
  - Single-env hardware/sim path remains unchanged.
- Added config knobs for HIL behavior:
  - `armada/config/base_rollout.yaml`
  - New `human_loop` section controls decision mode and anti-block parameters.

## Reused Modules
- `EpisodeManager` is still used for observation history and absolute-action conversion.
- Policy inference and replay buffer writing stay in existing ARMADA flow.
- Action mode semantics are preserved (`ROBOT` for policy, `INTV` for intervention).

## API Compatibility
- Existing methods still valid:
  - `reset_robot(...)`
  - `get_robot_state()`
  - `deploy_action(tcp_action, gripper_action)`
- Extended methods (optional):
  - `get_robot_state(env_idx=...)`
  - `deploy_action(..., env_idx=...)`
  - `deploy_action_batch(tcp_actions, gripper_actions)`
  - `set_active_env(env_idx)`
  - `get_env_done_flags()`

## Parallel Multi-Env Handling
- Policy rollout is advanced across all active envs in batched rounds.
- Each env keeps its own buffers and step counter.
- If one env enters intervention, other envs continue policy rollout.
- Intervention completion returns env back to policy mode without stalling others.

## Error/Timeout/Human Intervention Logic
- Timeout or truncation triggers human-loop decision per env.
- `manual` mode:
  - Prompt in terminal with timeout fallback.
- `auto` mode:
  - Uses configured default actions (`on_failure`, `on_timeout`).
- Blocking prevention:
  - `max_teleop_steps` limits one intervention window.
  - `max_blocked_rounds` forces release if env repeatedly blocks.

## Notes on Different Env Settings
- Current parallel path assumes one ManiSkill vectorized env config per rollout process.
- For strongly heterogeneous env settings, recommended practice is multiple rollout processes with different config overrides.
- Within a single process, per-env asynchronous intervention timing is supported.

## New Config Keys
In `human_loop`:
- `mode`: `auto` or `manual`
- `on_failure`: default decision on failure
- `on_timeout`: default decision on timeout/truncation
- `prompt_timeout_s`: manual input timeout fallback
- `max_teleop_steps`: intervention window cap
- `max_blocked_rounds`: repeated-block safeguard
