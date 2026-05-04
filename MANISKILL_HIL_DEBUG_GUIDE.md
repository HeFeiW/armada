# ManiSkill Rollout Debug Guide

This guide explains the ManiSkill ARMADA rollout output, the human-in-loop execution flow, and what happens when teleop hardware is missing.

## 1. How to render saved rollouts into video

Saved rollout data is stored in `save_buffer_path/replay_buffer.zarr`. Use the offline renderer to turn it into videos:

```bash
python maniskill_armada/render_rollout_videos.py \
  --replay-buffer /home/tmp_wanghf/armada/armada_data/maniskill_pick_rollout_poc/replay_buffer.zarr \
  --output-dir /home/tmp_wanghf/armada/armada_data/maniskill_pick_rollout_poc/videos
```

Optional flags:
- `--episode-index N`: render one episode only
- `--fps 10`: set output video frame rate
- `--no-actions`: hide action text overlay
- `--panel-width` / `--panel-height`: control camera panel size

Each output video contains:
- side camera on the left
- wrist camera on the right
- timestep and action mode
- failure marker when `failure_indices[t] == True`
- a compact state/action summary

## 2. What the rollout output means

The rollout episode saved to `replay_buffer.zarr` contains one or more episodes. Each episode usually has:
- `side_cam`: RGB frames from the side camera, shape `(T, H, W, 3)`
- `wrist_cam`: RGB frames from the wrist camera, shape `(T, H, W, 3)`
- `tcp_pose`: TCP pose sequence, shape `(T, 7)`
- `joint_pos`: joint positions, shape `(T, 7)`
- `action`: action sequence, shape `(T, 8)` or compatible rollout action format
- `action_mode`: per-step source flag
- `failure_indices`: per-step failure markers

Common meanings:
- `episode`: one full task attempt from reset to finish/discard.
- `timestep`: one saved transition index inside an episode, starting at 0.
- `action_mode`:
  - `ROBOT`: policy action
  - `INTV`: human intervention action
- `failure_indices`:
  - `True` means FLOAT marked that chunk as failure-related.
  - It is chunked by `Ta`, then expanded to per-step flags.

Important rollout state terms:
- `Ta`: action chunk length. One policy inference produces `Ta` executed steps.
- `To`: observation history length used by the policy.
- `j`: current global timestep inside an episode.
- `episode_idx`: current episode number in the rollout loop.
- `saved_episode_idx`: index of the latest episode written into the replay buffer.

## 3. Human-in-loop execution logic

The rollout path has three major stages:

### 3.1 Policy stage
The main loop repeatedly does:
- read current ManiSkill observation
- update `EpisodeManager` history
- infer an action chunk with the policy
- execute the first `Ta` steps one by one
- save each step into episode buffers
- after each chunk, let FLOAT consume the latent and update OT state

### 3.2 Failure detection stage
FLOAT works asynchronously:
- `process_step({'step_type': 'policy_step', ...})` submits the current latent to the async queue
- `detect_failure(...)` collects finished OT results and determines whether the current chunk is failing
- if a failure is detected, the rollout transitions into intervention logic
- if the episode reaches the max length, it also transitions to intervention or finish logic

### 3.3 Human intervention stage
When intervention is required:
- the runner rewinds part of the trajectory if the failure detector requests it
- the robot/env state is aligned to a safe point
- the system enters human intervention mode
- in ManiSkill simulation, the current implementation uses a simulated teleop fallback instead of a physical device

## 4. Why the current run did not block for teleop hardware

This is expected with the current ManiSkill path.

The rollout uses a simulated intervention fallback when teleop hardware is not available:
- `hardware/maniskill_robot_env.py` provides a no-op sigma device
- `human_teleop_step(...)` does not require real teleop hardware
- after a small number of simulated intervention steps, it automatically sets `keyboard.infer = True`
- this means the rollout can continue instead of waiting forever for a device that does not exist

So the behavior you saw is not a deadlock. It is a fallback designed to keep the rollout moving when no teleop device is attached.

## 5. Fallback modes and decisions

There are two layers of fallback:

### 5.1 Human-loop decision fallback
Configured in `armada/config/maniskill_rollout.yaml` under `human_loop`:
- `mode: manual` or `auto`
- `prompt_timeout_s`
- `on_failure`
- `on_timeout`
- `max_teleop_steps`
- `max_blocked_rounds`

If no decision is made in time, the system uses the configured default action.

### 5.2 Teleop-device fallback
In ManiSkill simulation:
- if no real teleop device exists, the code does not try to read a physical haptic device
- it uses the simulated teleop branch and automatically returns control after a short window
- this is why you did not see a hard block

## 6. Recommended debug workflow

1. Render a video from the saved replay buffer and inspect the trajectory quality.
2. If a rollout failed, open the dashboard or video and check the failure point.
3. If you need manual decisions, keep `human_loop.mode=manual`.
4. If you want a fully unattended test, switch to `human_loop.mode=auto`.
5. If you want stricter waiting behavior, reduce `max_teleop_steps` and avoid auto fallbacks.

## 7. Common log prefixes

The rollout now prints structured logs for the main decision points:

- `[RUNNER]`
  - rollout state, rewind progress, intervention entry/exit, cleanup, and fallback reasons
- `[HIL]`
  - human-in-loop decision logic, auto/manual decisions, timeout fallbacks, blocked rounds
- `[DASHBOARD]`
  - dashboard frame updates, error screens, and window lifecycle
- `[FLOAT]`
  - human-demo latent cache load/save status, cache miss reasons, and latent preparation progress

Common fallback messages:
- `rewind fallback -> using latest buffered frame as reference`
  - rewind stopped before any frame was popped, so the last saved frame was reused
- `rewind fallback -> using current env render as reference`
  - no buffered frame existed, so the current ManiSkill render was used
- `sim fallback -> entering human intervention`
  - ManiSkill uses the simulated teleop path rather than a real device
- `No valid input in ...s, fallback to ...`
  - manual decision timed out and used the configured default action

## 8. Files involved

- Rollout config: `armada/config/maniskill_rollout.yaml`
- Rollout runner: `armada/env_runner/real_env_runner.py`
- Simulated HIL controller: `armada/utils/maniskill_hil.py`
- Dashboard: `armada/utils/maniskill_dashboard.py`
- Offline renderer: `maniskill_armada/render_rollout_videos.py`
- FLOAT cache: `float/float.py`

## 9. Notes

- The offline renderer reads saved data only. It does not require ManiSkill to be running.
- If your saved episode contains only partial data, the renderer will skip missing fields.
- If the rollout output is not what you expect, first inspect the rendered video before changing training or policy code.


