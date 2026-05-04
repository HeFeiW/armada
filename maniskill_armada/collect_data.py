"""
Data Collection Script for ManiSkill + ARMADA

Collects trajectories using heuristic policy and saves to ARMADA-compatible Zarr format.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import time
import gymnasium as gym
import mani_skill.envs  
from mani_skill.utils.wrappers.record import RecordEpisode

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from maniskill_armada.heuristic_policy import HeuristicPickPolicy
from maniskill_armada.data_utils import (
    maniskill_obs_to_armada_format,
    render_cameras,
    set_env_goal_pos,
    resize_images,
    normalize_image
)


def _pick_mount_link(links_map: Dict[str, Any]):
    # Prefer existing camera mount links if present.
    preferred = [
        'camera_link',
        'hand_camera_link',
        'wrist_camera_link',
        'panda_hand',
        'hand',
        'wrist',
        'link7',
        'ee_link',
    ]
    for name in preferred:
        if name in links_map:
            return links_map[name]
    # Fallback: try fuzzy match.
    for key in links_map.keys():
        lk = key.lower()
        if 'camera' in lk:
            return links_map[key]
    for key in links_map.keys():
        lk = key.lower()
        if 'hand' in lk or 'wrist' in lk or 'ee' in lk:
            return links_map[key]
    # Final fallback: last link object.
    try:
        return list(links_map.values())[-1]
    except Exception:
        return None


def ensure_wrist_camera_sensor(env, uid: str = 'wrist_camera', width: int = 640, height: int = 480) -> bool:
    """Best-effort add a wrist-mounted camera to the ManiSkill env.

    Returns True if a reconfigure was attempted and the camera appears in obs.sensor_data.
    """
    try:
        from mani_skill.sensors.camera import CameraConfig  # type: ignore
        import sapien  # type: ignore
    except Exception:
        return False

    unwrapped = getattr(env, 'unwrapped', env)

    # If it already exists, nothing to do.
    try:
        obs0 = unwrapped.get_obs() if hasattr(unwrapped, 'get_obs') else None
        if isinstance(obs0, dict) and isinstance(obs0.get('sensor_data', None), dict):
            if uid in obs0['sensor_data']:
                return True
            # also accept common names
            for alt in ['hand_camera', 'wrist_cam', 'hand_cam']:
                if alt in obs0['sensor_data']:
                    return True
    except Exception:
        pass

    # Build mount link.
    try:
        robot = unwrapped.agent.robot
        links_map = getattr(robot, 'links_map', None)
        if not isinstance(links_map, dict):
            return False
        mount_link = _pick_mount_link(links_map)
        if mount_link is None:
            return False
    except Exception:
        return False

    # Create a camera config mounted to EE.
    try:
        cam_cfg = CameraConfig(
            uid=uid,
            pose=sapien.Pose(p=[0.0, 0.0, 0.0], q=[1.0, 0.0, 0.0, 0.0]),
            width=int(width),
            height=int(height),
            fov=float(np.pi / 2),
            near=0.01,
            far=100.0,
            mount=mount_link,
        )
    except Exception:
        return False

    # Install as custom sensor config and reconfigure.
    try:
        existing = getattr(unwrapped, '_custom_sensor_configs', None)
        if existing is None:
            existing = []
        # Ensure it's a list we can append.
        existing_list = list(existing)
        # Avoid duplicates.
        existing_uids = set()
        for cfg in existing_list:
            u = getattr(cfg, 'uid', None)
            if u is not None:
                existing_uids.add(str(u))
        if uid not in existing_uids:
            existing_list.append(cam_cfg)
        setattr(unwrapped, '_custom_sensor_configs', existing_list)

        if hasattr(unwrapped, '_reconfigure'):
            try:
                unwrapped._reconfigure()
            except TypeError:
                # Some versions expect a bool or kwargs; fall back to calling without args already tried.
                pass

        # Trigger a fresh observation to confirm.
        try:
            obs1 = unwrapped.get_obs() if hasattr(unwrapped, 'get_obs') else None
            if isinstance(obs1, dict) and isinstance(obs1.get('sensor_data', None), dict):
                return uid in obs1['sensor_data']
        except Exception:
            pass
        return True
    except Exception:
        return False


def get_env_obj_pose(env) -> np.ndarray:
    """Get current object pose [x,y,z,qw,qx,qy,qz] from ManiSkill env."""
    if hasattr(env.unwrapped, 'cube') and hasattr(env.unwrapped.cube, 'pose'):
        cube_pose = env.unwrapped.cube.pose
        p = np.asarray(cube_pose.p).reshape(-1)
        q = np.asarray(cube_pose.q).reshape(-1)
        return np.concatenate([p, q], axis=0).astype(np.float32)
    return None


def collect_episode(
    env,
    policy,
    episode_idx: int,
    max_steps: int = 200,
    reset_seed: Optional[int] = None,
    fixed_env_goal_pos: Optional[np.ndarray] = None,
) -> Dict:
    """
    Collect one episode using heuristic policy.

    Args:
        env: ManiSkill environment
        policy: HeuristicPickPolicy instance
        episode_idx: Episode index for logging
        max_steps: Maximum steps per episode

    Returns:
        dict with episode data {wrist_cam, side_cam, tcp_pose, joint_pos, action}
    """
    obs, info = env.reset(seed=reset_seed)
    # Keep cube spawn random, but optionally override the environment's internal goal.
    if fixed_env_goal_pos is not None:
        ok = set_env_goal_pos(env, fixed_env_goal_pos)
        if not ok:
            print(
                f"  Warning: Failed to override env goal_pos to {np.asarray(fixed_env_goal_pos).reshape(-1)[:3].tolist()}. "
                "Policy lift goal may still be fixed, but env obs/reward goal may vary."
            )
        # Refresh obs so obs['extra']['goal_pos'] reflects the overridden goal.
        try:
            obs = env.unwrapped.get_obs()
        except Exception:
            pass

    policy.reset()

    done = False
    step = 0
    success = False

    episode_data = {
        'wrist_cam': [],
        'side_cam': [],
        'tcp_pose': [],
        'joint_pos': [],
        'action': [],
        # Always store the lift goal used for this episode.
        # If --fixed-goal-pos is provided, this will be constant across episodes.
        'lift_goal_pos': [],
        # ManiSkill env may also expose an internal goal_pos in obs['extra'].
        # This can vary across resets even when the heuristic uses a fixed lift goal.
        'env_goal_pos': [],
    }

    # Store previous pose for action computation
    prev_tcp_pose = None

    while not done and step < max_steps:
        # Render cameras from ManiSkill obs + env
        try:
            # Only enable follow-tcp hack when there is still only one sensor camera.
            sensor_names = []
            if isinstance(obs, dict) and isinstance(obs.get('sensor_data', None), dict):
                sensor_names = list(obs['sensor_data'].keys())
            wrist_follow = len(sensor_names) <= 1
            side_img, wrist_img = render_cameras(obs, env, resolution=(640, 480), wrist_follow_tcp=wrist_follow)
        except Exception as e:
            print(f"  Warning: Camera rendering failed at step {step}: {e}")
            side_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
            wrist_img = np.ones((480, 640, 3), dtype=np.uint8) * 128

        # Extract robot state
        try:
            armada_obs = maniskill_obs_to_armada_format(obs, env)
            tcp_pose = armada_obs['tcp_pose']
            joint_pos = armada_obs['joint_pos']
        except Exception as e:
            print(f"  Warning: State extraction failed at step {step}: {e}")
            tcp_pose = np.zeros(7, dtype=np.float32)
            joint_pos = np.zeros(7, dtype=np.float32)

        # Resolve lift goal used by heuristic policy (preferred), plus env goal for reference.
        env_goal_pos = None
        try:
            extra = obs.get('extra', {}) if isinstance(obs, dict) else {}
            env_goal_pos = extra.get('goal_pos', None)
            if env_goal_pos is not None:
                env_goal_pos = np.asarray(env_goal_pos, dtype=np.float32)
                if env_goal_pos.ndim == 2:
                    env_goal_pos = env_goal_pos[0]
                env_goal_pos = env_goal_pos.reshape(-1)[:3]
        except Exception:
            env_goal_pos = None

        lift_goal_pos = getattr(policy, 'fixed_goal_pos', None)
        if lift_goal_pos is not None:
            lift_goal_pos = np.asarray(lift_goal_pos, dtype=np.float32).reshape(-1)[:3]
        else:
            # Fallback to env-reported goal_pos if present; otherwise keep zeros.
            lift_goal_pos = env_goal_pos if env_goal_pos is not None else np.zeros(3, dtype=np.float32)

        # Get action from policy
        try:
            obj_pose = get_env_obj_pose(env)
            action = policy.get_action(obs, obj_pose=obj_pose)  # (8,)
        except Exception as e:
            print(f"  Warning: Policy inference failed at step {step}: {e}")
            action = np.zeros(8, dtype=np.float32)

        # Store data
        episode_data['wrist_cam'].append(wrist_img)
        episode_data['side_cam'].append(side_img)
        episode_data['tcp_pose'].append(tcp_pose)
        episode_data['joint_pos'].append(joint_pos)
        episode_data['action'].append(action)
        episode_data['lift_goal_pos'].append(lift_goal_pos)
        episode_data['env_goal_pos'].append(env_goal_pos if env_goal_pos is not None else np.zeros(3, dtype=np.float32))

        # Execute action in environment
        try:
            env_action = policy.to_maniskill_action(action)
            obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            success = float(reward) > 0.5
        except Exception as e:
            print(f"  Warning: Environment step failed at step {step}: {e}")
            done = True
            success = False

        step += 1

    # Convert lists to numpy arrays and stack
    if len(episode_data['wrist_cam']) == 0:
        print(f"  Error: No data collected for episode {episode_idx}")
        return None

    episode_dict = {}
    for key, values in episode_data.items():
        stacked = np.stack(values, axis=0)
        episode_dict[key] = stacked.astype(np.float32) if key != 'wrist_cam' and key != 'side_cam' else stacked.astype(np.uint8)

    return {
        'data': episode_dict,
        'steps': step,
        'success': success
    }


def main():
    parser = argparse.ArgumentParser(description='Collect data for ManiSkill PickCube task')
    parser.add_argument('--output', type=str, default=str(REPO_ROOT / 'armada_data' / 'maniskill_pick_fixed_goal_min'),
                        help='Output directory for zarr data')
    # data collection parameters
    parser.add_argument('--num-envs', type=int, default=1, help='Number of parallel environments')
    parser.add_argument('--num-episodes', type=int, default=50,
                        help='Total number of episodes (default 50)')
    parser.add_argument('--max-steps', type=int, default=200,
                        help='Maximum steps per episode (default 200)')
    parser.add_argument('--no-zarr', action='store_true',
                        help='Skip zarr saving (debug mode)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for environment')
    parser.add_argument('--save-video', action='store_true', help='Whether to save episode videos')
    parser.add_argument(
        '--fixed-reset-seed',
        type=int,
        default=None,
        help='Use the same reset seed for every episode (minimal validation for fixed target/initial state).',
    )
    parser.add_argument(
        '--fixed-goal-pos',
        type=float,
        nargs=3,
        default=None,
        metavar=('X', 'Y', 'Z'),
        help='Override heuristic lift target goal_pos with a fixed world position [x y z].',
    )
    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")

    # Initialize environment
    print("Initializing ManiSkill environment...")
    np.random.seed(args.seed)
    try:
        env = gym.make(
        "PickCube-v1",
        num_envs=args.num_envs,
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        max_episode_steps=args.max_steps,
        )
        if args.save_video:
            env = RecordEpisode(
                env,
                output_dir=str(output_path / 'videos'),
                save_trajectory=True,
                trajectory_name="pickcube_single_trajectory",
                save_video=True,
                video_fps=30,
            )
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("Trying alternative environment creation...")
        # try:
        #     env = gym.make('PickCube-v1', obs_mode='state_dict', control_mode='pd_ee_delta_pose', render_mode='rgb_array')
        # except Exception as e2:
        #     print(f"Failed to create environment: {e2}")
        #     return
        return

    print(f"Environment created successfully")

    # One-time: try to inject a true wrist camera mounted to end-effector.
    # Do a warmup reset so robot links exist; then install config + reconfigure.
    try:
        _obs0, _info0 = env.reset(seed=args.seed)
    except Exception:
        _obs0, _info0 = None, None

    wrist_ok = ensure_wrist_camera_sensor(env, uid='wrist_camera', width=640, height=480)
    setattr(env, '_armada_wrist_sensor_installed', bool(wrist_ok))
    if wrist_ok:
        print("[collect_data] Installed EE-mounted wrist_camera sensor via CameraConfig")
    else:
        print("[collect_data][WARN] Could not install EE-mounted wrist camera; will fall back to single-camera mode")

    # Initialize heuristic policy
    policy_cfg = {}
    fixed_env_goal_pos = None
    if args.fixed_goal_pos is not None:
        policy_cfg['fixed_goal_pos'] = list(args.fixed_goal_pos)
        # For fixed-goal dataset collection, keep lift as position-only.
        policy_cfg['lift_position_only'] = True
        print(f"Using fixed heuristic goal_pos: {policy_cfg['fixed_goal_pos']}")
        print(
            "Note: This script will also try to override ManiSkill env goal_pos after each reset to match --fixed-goal-pos. "
            "It saves both lift_goal_pos (used by policy) and env_goal_pos (from obs) per step for verification."
        )
        # Also request env-level fixed goal by default to keep task goal consistent.
        fixed_env_goal_pos = np.asarray(args.fixed_goal_pos, dtype=np.float32).reshape(3)

    policy = HeuristicPickPolicy(env, config=policy_cfg)

    total_episodes = 0
    total_steps = 0
    total_successes = 0

    episode_buffer = []  # Buffer to collect all episodes before saving

    # Collect data
    for ep_idx in range(args.num_episodes):
        print(f"\nEpisode {ep_idx + 1}/{args.num_episodes}...")
        try:
            reset_seed = args.fixed_reset_seed
            result = collect_episode(
                env,
                policy,
                ep_idx,
                max_steps=args.max_steps,
                reset_seed=reset_seed,
                fixed_env_goal_pos=fixed_env_goal_pos,
            )

            if result is None:
                print("FAILED (no data)")
                continue

            episode_data = result['data']
            steps = result['steps']
            success = result['success']

            status = "[SUCCESS]" if success else "[FAILED]"
            print(f"{status} ({steps} steps)")

            episode_buffer.append(episode_data)
            total_episodes += 1
            total_steps += steps
            total_successes += int(success)

        except Exception as e:
            print(f"ERROR: {e}")
    env.close()
    print(f"\n{'='*60}")
    print(f"Collection Complete!")
    print(f"{'='*60}")
    print(f"Total episodes collected: {total_episodes}")
    print(f"Total steps: {total_steps}")
    print(f"Success rate: {total_successes / max(total_episodes, 1) * 100:.1f}%")

    # Save to Zarr (if not skipped)
    if not args.no_zarr and total_episodes > 0:
        print(f"\nSaving to Zarr format...")

        try:
            # Import ReplayBuffer from ARMADA
            from armada.diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer

            zarr_path = str(output_path / 'replay_buffer.zarr')
            buffer = ReplayBuffer.create_from_path(zarr_path, mode='a')

            for episode_data in episode_buffer:
                buffer.add_episode(episode_data, compressors='disk')

            print(f"✓ Saved {total_episodes} episodes to {zarr_path}")
            print(f"  Total transitions: {total_steps}")

        except Exception as e:
            print(f"Error saving to Zarr: {e}")
            print("Attempting to save state as numpy...")

            # Fallback: save as numpy
            for i, episode_data in enumerate(episode_buffer):
                np.savez_compressed(
                    output_path / f'episode_{i:04d}.npz',
                    **episode_data
                )
            print(f"✓ Saved episodes as numpy files")

    env.close()
    print("\nData collection finished!")


if __name__ == '__main__':
    main()
