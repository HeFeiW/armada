"""
Data Collection Script for ManiSkill + ARMADA

Collects trajectories using heuristic policy and saves to ARMADA-compatible Zarr format.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Dict
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
    resize_images,
    normalize_image
)


def get_env_obj_pose(env) -> np.ndarray:
    """Get current object pose [x,y,z,qw,qx,qy,qz] from ManiSkill env."""
    if hasattr(env.unwrapped, 'cube') and hasattr(env.unwrapped.cube, 'pose'):
        cube_pose = env.unwrapped.cube.pose
        p = np.asarray(cube_pose.p).reshape(-1)
        q = np.asarray(cube_pose.q).reshape(-1)
        return np.concatenate([p, q], axis=0).astype(np.float32)
    return None


def collect_episode(env, policy, episode_idx: int, max_steps: int = 200) -> Dict:
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
    obs, info = env.reset()
    policy.reset()

    done = False
    step = 0
    success = False

    episode_data = {
        'wrist_cam': [],
        'side_cam': [],
        'tcp_pose': [],
        'joint_pos': [],
        'action': []
    }

    # Store previous pose for action computation
    prev_tcp_pose = None

    while not done and step < max_steps:
        # Render cameras from ManiSkill obs + env
        try:
            side_img, wrist_img = render_cameras(obs, env, resolution=(640, 480))
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
    parser.add_argument('--output', type=str, default=str(REPO_ROOT / 'armada_data' / 'maniskill_pick'),
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

    # Initialize heuristic policy
    policy = HeuristicPickPolicy(env)

    total_episodes = 0
    total_steps = 0
    total_successes = 0

    episode_buffer = []  # Buffer to collect all episodes before saving

    # Collect data
    for ep_idx in range(args.num_episodes):
        print(f"\nEpisode {ep_idx + 1}/{args.num_episodes}...")
        try:
            result = collect_episode(env, policy, ep_idx)

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
