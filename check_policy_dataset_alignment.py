#!/usr/bin/env python3
"""
Minimal offline test for policy behavior on training dataset observations.

What this script does:
1. Loads policy checkpoint using the same workspace/payload path as rollout.
2. Samples observation windows from training replay buffer.
3. Runs policy.predict_action(obs_window) on those windows.
4. Compares predicted actions against dataset ground-truth actions.

This is a quick sanity check for behavior cloning alignment, not a full benchmark.
"""

import argparse
import os
import random
import socket
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Resize


REPO_ROOT = Path(__file__).resolve().parent
# Match training entrypoint import behavior so hydra target
# `diffusion_policy.workspace.*` can be imported.
sys.path.insert(0, str(REPO_ROOT / "armada"))
sys.path.insert(0, str(REPO_ROOT / "armada" / "diffusion_policy"))

from armada.diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer
from armada.diffusion_policy.diffusion_policy.model.common.rotation_transformer import RotationTransformer


def _find_free_port(start_port: int = 30123) -> int:
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                port += 1


def _init_single_process_group(device_id: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_find_free_port()))
    torch.distributed.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(device_id)


def _destroy_process_group_safely():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _as_tensor_image_window(
    imgs_hwc: np.ndarray,
    image_shape: Tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    # imgs_hwc: (T, H, W, 3), uint8
    # output: (1, T, 3, Ht, Wt), float32 in [0, 1]
    _, target_h, target_w = image_shape
    proc = Compose(
        [
            Resize((target_h + 8, target_w + 8), interpolation=InterpolationMode.BICUBIC, antialias=True),
            CenterCrop((target_h, target_w)),
        ]
    )
    tchw = torch.from_numpy(imgs_hwc).permute(0, 3, 1, 2).float() / 255.0
    tchw = proc(tchw)
    return tchw.unsqueeze(0).to(device)


def _quat_wxyz_to_rot6d(quat_wxyz: np.ndarray) -> np.ndarray:
    rot = R.from_quat(quat_wxyz, scalar_first=True)
    mat = rot.as_matrix()
    return mat[:, :2].reshape(-1).astype(np.float32)


def _convert_tcp_pose_to_ee_pose_9d(tcp_pose: np.ndarray) -> np.ndarray:
    # tcp_pose: (T, 7) as [x, y, z, qw, qx, qy, qz]
    out = np.zeros((tcp_pose.shape[0], 9), dtype=np.float32)
    out[:, :3] = tcp_pose[:, :3]
    for i in range(tcp_pose.shape[0]):
        out[i, 3:] = _quat_wxyz_to_rot6d(tcp_pose[i, 3:7])
    return out


def _convert_action_8d_to_10d(action_8d: np.ndarray, rot_tf: RotationTransformer) -> np.ndarray:
    # action_8d: (T, 8) as [dx, dy, dz, qw, qx, qy, qz, grip]
    out = np.zeros((action_8d.shape[0], 10), dtype=np.float32)
    out[:, :3] = action_8d[:, :3]
    out[:, 3:9] = rot_tf.forward(action_8d[:, 3:7])
    out[:, 9] = action_8d[:, 7]
    return out


@dataclass
class SampleWindow:
    ep_idx: int
    t: int


def _episode_bounds(episode_ends: np.ndarray, ep_idx: int) -> Tuple[int, int]:
    start = 0 if ep_idx == 0 else int(episode_ends[ep_idx - 1])
    end = int(episode_ends[ep_idx])
    return start, end


def _collect_valid_windows(
    replay_buffer: ReplayBuffer,
    n_obs_steps: int,
    ta: int,
) -> List[SampleWindow]:
    windows: List[SampleWindow] = []
    episode_ends = np.asarray(replay_buffer.episode_ends, dtype=np.int64)
    for ep_idx in range(replay_buffer.n_episodes):
        start, end = _episode_bounds(episode_ends, ep_idx)
        length = end - start
        min_t = n_obs_steps - 1
        max_t = length - ta
        if max_t < min_t:
            continue
        for t in range(min_t, max_t + 1):
            windows.append(SampleWindow(ep_idx=ep_idx, t=t))
    return windows


def _build_obs_and_gt(
    replay_buffer: ReplayBuffer,
    win: SampleWindow,
    n_obs_steps: int,
    ta: int,
    image_shape: Tuple[int, int, int],
    device: torch.device,
    rot_tf: RotationTransformer,
) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
    episode = replay_buffer.get_episode(win.ep_idx, copy=True)
    t = win.t

    obs_start = t - n_obs_steps + 1
    obs_end = t + 1

    side_window = episode["side_cam"][obs_start:obs_end]
    wrist_window = episode["wrist_cam"][obs_start:obs_end]
    tcp_window = episode["tcp_pose"][obs_start:obs_end]

    obs_dict = {
        "side_img": _as_tensor_image_window(side_window, image_shape=image_shape, device=device),
        "wrist_img": _as_tensor_image_window(wrist_window, image_shape=image_shape, device=device),
        "ee_pose": torch.from_numpy(_convert_tcp_pose_to_ee_pose_9d(tcp_window)).unsqueeze(0).to(device),
    }

    gt_8d = episode["action"][t : t + ta]
    gt_10d = _convert_action_8d_to_10d(gt_8d, rot_tf)
    return obs_dict, gt_10d


def _compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    # pred/gt shape: (Ta, 10)
    diff = pred - gt
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff**2))
    mae = float(np.mean(abs_diff))

    # Split metrics for interpretability.
    pos_mae = float(np.mean(abs_diff[:, :3]))
    rot6d_mae = float(np.mean(abs_diff[:, 3:9]))
    grip_mae = float(np.mean(abs_diff[:, 9]))

    return {
        "mse": mse,
        "mae": mae,
        "pos_mae": pos_mae,
        "rot6d_mae": rot6d_mae,
        "grip_mae": grip_mae,
    }


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    # rot6d shape: (..., 6)
    v1 = rot6d[..., 0:3]
    v2 = rot6d[..., 3:6]

    # Gram-Schmidt orthonormalization
    e1 = v1 / np.clip(np.linalg.norm(v1, axis=-1, keepdims=True), 1e-8, None)
    v2_proj = np.sum(e1 * v2, axis=-1, keepdims=True) * e1
    u2 = v2 - v2_proj
    e2 = u2 / np.clip(np.linalg.norm(u2, axis=-1, keepdims=True), 1e-8, None)
    e3 = np.cross(e1, e2)

    mat = np.stack([e1, e2, e3], axis=-1)
    return mat.astype(np.float32)


def _rotation_geodesic_deg(pred_rot6d: np.ndarray, gt_rot6d: np.ndarray) -> np.ndarray:
    # input shape: (T, 6), return shape: (T,)
    pred_m = _rot6d_to_matrix(pred_rot6d)
    gt_m = _rot6d_to_matrix(gt_rot6d)
    rel = np.matmul(np.transpose(pred_m, (0, 2, 1)), gt_m)
    tr = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    rad = np.arccos(tr)
    return np.degrees(rad).astype(np.float32)


def _gripper_binary_accuracy(pred_grip: np.ndarray, gt_grip: np.ndarray, threshold: float = 0.5) -> float:
    pred_bin = pred_grip >= threshold
    gt_bin = gt_grip >= threshold
    return float(np.mean(pred_bin == gt_bin))


def _fmt_mean_std(vals: List[float]) -> str:
    arr = np.asarray(vals, dtype=np.float64)
    return f"{arr.mean():.6f} +- {arr.std(ddof=0):.6f}"


def main():
    parser = argparse.ArgumentParser(description="Minimal offline policy-vs-dataset alignment test")
    parser.add_argument("--training-config", type=str, default="train_maniskill_poc")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset-zarr", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--ta", type=int, default=8, help="Number of rollout steps compared against GT")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--use-ema", action="store_true", help="Force EMA model for inference")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this minimal test because workspace uses DDP on GPU.")

    with hydra.initialize(config_path="./armada/config/training", version_base=None):
        cfg = hydra.compose(config_name=args.training_config)

    dataset_zarr = args.dataset_zarr or cfg.task.dataset.zarr_path
    print(f"[TEST] training_config={args.training_config}")
    print(f"[TEST] checkpoint={args.checkpoint}")
    print(f"[TEST] dataset_zarr={dataset_zarr}")

    _init_single_process_group(args.device_id)
    device = torch.device(f"cuda:{args.device_id}")

    try:
        payload = torch.load(open(args.checkpoint, "rb"), pickle_module=dill)

        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, rank=0, world_size=1, device_id=args.device_id, device=str(device))
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        policy = workspace.model.module
        use_ema = bool(args.use_ema or cfg.training.use_ema)
        if use_ema:
            policy = workspace.ema_model.module
        policy.to(device)
        policy.eval()

        n_obs_steps = int(cfg.n_obs_steps)
        image_shape = tuple(cfg.task.image_shape)
        action_rot_tf = RotationTransformer(from_rep="quaternion", to_rep=cfg.shape_meta.action.rotation_rep)

        replay_buffer = ReplayBuffer.copy_from_path(
            dataset_zarr,
            keys=["side_cam", "wrist_cam", "tcp_pose", "action"],
        )
        valid_windows = _collect_valid_windows(replay_buffer, n_obs_steps=n_obs_steps, ta=args.ta)
        if len(valid_windows) == 0:
            raise RuntimeError("No valid windows found. Check n_obs_steps/ta and dataset episode lengths.")

        sample_n = min(args.num_samples, len(valid_windows))
        sampled = random.sample(valid_windows, sample_n)

        metrics = {
            "mse": [],
            "mae": [],
            "pos_mae": [],
            "rot6d_mae": [],
            "grip_mae": [],
            "rot_geodesic_deg": [],
            "grip_bin_acc": [],
        }
        hardest = []

        for i, win in enumerate(sampled):
            obs_dict, gt_10d = _build_obs_and_gt(
                replay_buffer=replay_buffer,
                win=win,
                n_obs_steps=n_obs_steps,
                ta=args.ta,
                image_shape=image_shape,
                device=device,
                rot_tf=action_rot_tf,
            )

            with torch.no_grad():
                pred_dict = policy.predict_action(obs_dict)
            pred = pred_dict["action"][0, : args.ta].detach().cpu().numpy()

            m = _compute_metrics(pred, gt_10d)
            for k in metrics:
                if k in m:
                    metrics[k].append(m[k])

            rot_deg = _rotation_geodesic_deg(pred[:, 3:9], gt_10d[:, 3:9])
            metrics["rot_geodesic_deg"].append(float(np.mean(rot_deg)))
            metrics["grip_bin_acc"].append(_gripper_binary_accuracy(pred[:, 9], gt_10d[:, 9]))

            hardest.append((m["mae"], win.ep_idx, win.t))

            if (i + 1) % 10 == 0 or (i + 1) == sample_n:
                print(f"[TEST] processed {i + 1}/{sample_n}")

        hardest.sort(reverse=True)

        print("\n===== Policy vs Dataset Alignment Summary =====")
        print(f"samples: {sample_n}")
        print(f"n_obs_steps: {n_obs_steps}")
        print(f"ta(compared steps): {args.ta}")
        print(f"use_ema: {use_ema}")
        print(f"MAE(all dims): {_fmt_mean_std(metrics['mae'])}")
        print(f"MSE(all dims): {_fmt_mean_std(metrics['mse'])}")
        print(f"MAE(pos xyz): {_fmt_mean_std(metrics['pos_mae'])}")
        print(f"MAE(rot6d): {_fmt_mean_std(metrics['rot6d_mae'])}")
        print(f"Rotation geodesic error (deg): {_fmt_mean_std(metrics['rot_geodesic_deg'])}")
        print(f"MAE(gripper): {_fmt_mean_std(metrics['grip_mae'])}")
        print(f"Gripper binary accuracy: {_fmt_mean_std(metrics['grip_bin_acc'])}")

        print("\nTop-5 hardest windows by MAE:")
        for rank, (err, ep_idx, t) in enumerate(hardest[:5], start=1):
            print(f"  {rank}. episode={ep_idx}, t={t}, mae={err:.6f}")

        print("\nInterpretation tips:")
        print("- Lower MAE/MSE means behavior cloning alignment is better on dataset observations.")
        print("- If grip MAE is much larger than pos/rot, check gripper convention consistency.")
        print("- If all metrics are high, prioritize data/action representation checks before rollout debugging.")

    finally:
        _destroy_process_group_safely()


if __name__ == "__main__":
    main()
