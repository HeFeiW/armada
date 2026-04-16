"""
Render previously saved ManiSkill ARMADA rollouts into videos.

Input:
- replay_buffer.zarr produced by rollout or data collection.

Output:
- One mp4 per episode, with side/wrist view, timestep, action mode,
  failure markers, and basic episode summary overlays.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT))

from armada.diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255.0).clip(0, 255)
        image = image.astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image with 3 channels, got shape {image.shape}")
    return image


def _draw_text_lines(canvas: np.ndarray, lines: Iterable[str], origin=(14, 28), color=(255, 255, 255), scale=0.55):
    x, y = origin
    for line in lines:
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += int(22 * max(scale / 0.55, 1.0))


def _make_episode_frame(
    episode: Dict[str, np.ndarray],
    t: int,
    show_actions: bool = True,
    panel_size: Tuple[int, int] = (480, 360),
) -> np.ndarray:
    side = _as_uint8_rgb(episode["side_cam"][t])
    wrist = _as_uint8_rgb(episode["wrist_cam"][t])

    side_bgr = cv2.cvtColor(cv2.resize(side, panel_size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
    wrist_bgr = cv2.cvtColor(cv2.resize(wrist, panel_size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)

    montage = np.hstack([side_bgr, wrist_bgr])
    header_h = 72
    header = np.zeros((header_h, montage.shape[1], 3), dtype=np.uint8)
    action_mode = int(episode["action_mode"][t]) if "action_mode" in episode else -1
    failure = bool(episode["failure_indices"][t]) if "failure_indices" in episode else False
    total = episode["action"].shape[0]
    progress = f"{t + 1}/{total}"
    lines = [
        f"Episode video | step {progress} | action_mode={action_mode} | failure={failure}",
        "Left: side camera | Right: wrist camera",
    ]
    _draw_text_lines(header, lines, origin=(14, 28), color=(255, 255, 255), scale=0.58)

    if "tcp_pose" in episode and t < episode["tcp_pose"].shape[0]:
        tcp_pose = np.asarray(episode["tcp_pose"][t]).reshape(-1)
        state_lines = [
            f"tcp_pose: [{tcp_pose[0]:+.3f}, {tcp_pose[1]:+.3f}, {tcp_pose[2]:+.3f}, ...]",
        ]
        _draw_text_lines(header, state_lines, origin=(860, 28), color=(200, 240, 255), scale=0.45)

    if show_actions and "action" in episode and t < episode["action"].shape[0]:
        action = np.asarray(episode["action"][t]).reshape(-1)
        action_lines = [
            f"action: [{', '.join(f'{x:+.3f}' for x in action[:4])}, ...]",
        ]
        _draw_text_lines(header, action_lines, origin=(860, 52), color=(220, 220, 180), scale=0.43)

    if failure:
        cv2.rectangle(montage, (0, 0), (montage.shape[1], montage.shape[0]), (0, 0, 120), thickness=4)

    return np.vstack([header, montage])


def render_episode_to_video(
    episode: Dict[str, np.ndarray],
    output_path: Path,
    fps: int = 10,
    show_actions: bool = True,
    panel_size: Tuple[int, int] = (480, 360),
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_frame = _make_episode_frame(episode, 0, show_actions=show_actions, panel_size=panel_size)
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    try:
        for t in range(episode["action"].shape[0]):
            frame = _make_episode_frame(episode, t, show_actions=show_actions, panel_size=panel_size)
            # _make_episode_frame already returns a BGR frame for OpenCV writer.
            writer.write(frame)
    finally:
        writer.release()


def _load_replay_buffer(zarr_path: Path) -> ReplayBuffer:
    if not zarr_path.exists():
        raise FileNotFoundError(f"Replay buffer not found: {zarr_path}")
    return ReplayBuffer.copy_from_path(str(zarr_path), keys=None)


def _iter_episode_indices(buffer: ReplayBuffer, episode_index: Optional[int]) -> List[int]:
    if episode_index is not None:
        return [episode_index]
    return list(range(buffer.n_episodes))


def main():
    parser = argparse.ArgumentParser(description="Render ManiSkill ARMADA rollout trajectories to mp4 videos")
    parser.add_argument("--replay-buffer", type=str, required=True, help="Path to replay_buffer.zarr")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save mp4 videos")
    parser.add_argument("--episode-index", type=int, default=None, help="Render only one episode")
    parser.add_argument("--fps", type=int, default=10, help="Video fps")
    parser.add_argument("--no-actions", action="store_true", help="Hide action overlay")
    parser.add_argument("--panel-width", type=int, default=480, help="Single camera panel width")
    parser.add_argument("--panel-height", type=int, default=360, help="Single camera panel height")
    args = parser.parse_args()

    zarr_path = Path(args.replay_buffer)
    output_dir = Path(args.output_dir)

    buffer = _load_replay_buffer(zarr_path)
    episode_indices = _iter_episode_indices(buffer, args.episode_index)

    rendered = 0
    for ep_idx in episode_indices:
        episode = buffer.get_episode(ep_idx, copy=True)
        # debug: 
        print(f"Episode {ep_idx} keys: {list(episode.keys())}")
        for key in episode.keys():
            print(f"  {key}: shape={episode[key].shape} dtype={episode[key].dtype}")
        if "side_cam" not in episode or "wrist_cam" not in episode:
            print(f"Skipping episode {ep_idx}: missing side_cam/wrist_cam")
            continue
        if "action" not in episode:
            print(f"Skipping episode {ep_idx}: missing action")
            continue

        video_path = output_dir / f"episode_{ep_idx:04d}.mp4"
        print(f"Rendering episode {ep_idx} -> {video_path}")
        render_episode_to_video(
            episode,
            video_path,
            fps=args.fps,
            show_actions=not args.no_actions,
            panel_size=(args.panel_width, args.panel_height),
        )
        rendered += 1

    print(f"Rendered {rendered} episode(s) to {output_dir}")


if __name__ == "__main__":
    main()
