"""
Quick data collection script for ManiSkill PickCube POC.

Goal:
- prioritize pipeline verification over SOTA performance
- collect a moderate number of successful demos quickly
- save ARMADA-compatible replay_buffer.zarr
"""

import argparse
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np

from collect_data import collect_episode
from heuristic_policy import HeuristicPickPolicy


def save_episode_batch(zarr_path: Path, batch: List[Dict]) -> None:
    """Append a batch of episodes to replay buffer."""
    if len(batch) == 0:
        return

    from armada.diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer

    buffer = ReplayBuffer.create_from_path(str(zarr_path), mode="a")
    for episode_data in batch:
        buffer.add_episode(episode_data, compressors="disk")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick ManiSkill PickCube demo collection for POC training")
    parser.add_argument("--output", type=str, default="armada_data/maniskill_pick")
    parser.add_argument("--target-successes", type=int, default=120,
                        help="Stop when this many successful episodes are collected")
    parser.add_argument("--max-attempts", type=int, default=220,
                        help="Hard cap on total attempts to avoid very long runs")
    parser.add_argument("--max-steps", type=int, default=180,
                        help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=20,
                        help="Flush to disk every N successful episodes")
    parser.add_argument("--keep-failed", action="store_true",
                        help="Also keep failed episodes (default: only keep successful demos)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete existing replay_buffer.zarr before collection")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = output_dir / "replay_buffer.zarr"

    if args.overwrite and zarr_path.exists():
        import shutil
        shutil.rmtree(zarr_path)

    np.random.seed(args.seed)

    env = gym.make(
        "PickCube-v1",
        num_envs=args.num_envs,
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        max_episode_steps=args.max_steps,
    )

    policy = HeuristicPickPolicy(env)

    successes = 0
    attempts = 0
    total_steps = 0
    to_save: List[Dict] = []

    print("=" * 72)
    print("Quick POC data collection")
    print(f"output: {output_dir}")
    print(f"target_successes: {args.target_successes}, max_attempts: {args.max_attempts}")
    print(f"keep_failed: {args.keep_failed}, save_every: {args.save_every}")
    print("=" * 72)

    while attempts < args.max_attempts and successes < args.target_successes:
        attempts += 1
        result = collect_episode(env, policy, attempts - 1, max_steps=args.max_steps)
        if result is None:
            print(f"[{attempts:04d}] no data, skip")
            continue

        episode = result["data"]
        step_n = int(result["steps"])
        ok = bool(result["success"])
        total_steps += step_n

        if ok:
            successes += 1

        if ok or args.keep_failed:
            to_save.append(episode)

        tag = "SUCCESS" if ok else "FAILED"
        print(f"[{attempts:04d}] {tag:<7} steps={step_n:<4} | success={successes}/{args.target_successes}")

        if len(to_save) >= args.save_every:
            save_episode_batch(zarr_path, to_save)
            print(f"  flushed {len(to_save)} episodes to {zarr_path}")
            to_save = []

    if len(to_save) > 0:
        save_episode_batch(zarr_path, to_save)
        print(f"flushed final {len(to_save)} episodes to {zarr_path}")

    env.close()

    success_rate = successes / max(attempts, 1)
    print("=" * 72)
    print("Collection finished")
    print(f"attempts={attempts}, successes={successes}, success_rate={success_rate:.2%}, total_steps={total_steps}")
    print(f"dataset: {zarr_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
