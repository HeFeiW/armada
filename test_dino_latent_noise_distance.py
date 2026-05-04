#!/usr/bin/env python3
"""
Test whether DINO-based latent is discriminative by comparing:
1) Expert trajectory first frame latent
2) Pure noise image latent

The script loads the same policy checkpoint/workspace path used by rollout,
then computes cosine similarity/distance between flattened latents.
"""

import argparse
import os
import random
import socket
import sys
from pathlib import Path
from typing import Dict, Tuple

import dill
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Resize

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "armada"))
sys.path.insert(0, str(REPO_ROOT / "armada" / "diffusion_policy"))

from armada.diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer
from armada.diffusion_policy.diffusion_policy.model.common.rotation_transformer import RotationTransformer
from hardware.my_device.macros import HUMAN


def _find_free_port(start_port: int = 30123) -> int:
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                port += 1


def _init_single_process_group(device_id: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_find_free_port()))
    torch.distributed.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(device_id)


def _destroy_process_group_safely() -> None:
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


def _episode_bounds(episode_ends: np.ndarray, ep_idx: int) -> Tuple[int, int]:
    start = 0 if ep_idx == 0 else int(episode_ends[ep_idx - 1])
    end = int(episode_ends[ep_idx])
    return start, end


def _find_first_expert_episode(replay_buffer: ReplayBuffer) -> int:
    if "action_mode" not in replay_buffer.data:
        print("[WARN] action_mode not found in replay buffer; fallback to episode 0")
        return 0

    # action_mode = replay_buffer.data["action_mode"]
    episode_ends = np.asarray(replay_buffer.episode_ends, dtype=np.int64)
    for ep_idx in range(replay_buffer.n_episodes):
        start, end = _episode_bounds(episode_ends, ep_idx)
        # if np.any(action_mode[start:end] == HUMAN):
        #     return ep_idx
        return ep_idx  # For now, just return the first episode regardless of action_mode

    print("[WARN] no HUMAN-labeled episode found; fallback to episode 0")
    return 0


def _build_ee_pose_from_tcp_first_frame(
    tcp_pose_first: np.ndarray,
    ee_pose_dim: int,
    rot_rep: str,
) -> np.ndarray:
    # tcp_pose_first: [x, y, z, qw, qx, qy, qz]
    out = np.zeros((ee_pose_dim,), dtype=np.float32)
    out[:3] = tcp_pose_first[:3].astype(np.float32)
    rot_tf = RotationTransformer(from_rep="quaternion", to_rep=rot_rep)
    out[3:] = rot_tf.forward(tcp_pose_first[3:7][None, :])[0].astype(np.float32)
    return out


def _build_obs_dict(
    side_frame: np.ndarray,
    wrist_frame: np.ndarray,
    state_first: np.ndarray,
    n_obs_steps: int,
    target_image_shape: Tuple[int, int, int],
    state_key: str,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    side_window = np.repeat(side_frame[None, ...], n_obs_steps, axis=0)
    wrist_window = np.repeat(wrist_frame[None, ...], n_obs_steps, axis=0)
    state_window = np.repeat(state_first[None, ...], n_obs_steps, axis=0)

    return {
        "side_img": _as_tensor_image_window(side_window, image_shape=target_image_shape, device=device),
        "wrist_img": _as_tensor_image_window(wrist_window, image_shape=target_image_shape, device=device),
        state_key: torch.from_numpy(state_window).unsqueeze(0).to(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare expert first-frame latent vs pure-noise latent")
    parser.add_argument("--training-config", type=str, default="train_maniskill_poc")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset-zarr", type=str, default=None)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=123)
    parser.add_argument("--use-ema", action="store_true", help="Force EMA model for inference")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because workspace model loading uses DDP on GPU.")

    with hydra.initialize(config_path="./armada/config/training", version_base=None):
        cfg = hydra.compose(config_name=args.training_config)

    if "policy" in cfg and "obs_encoder" in cfg.policy:
        cfg.policy.obs_encoder.pretrained_path = None

    dataset_zarr = args.dataset_zarr or cfg.task.dataset.zarr_path
    n_obs_steps = int(cfg.n_obs_steps)
    dataset_image_shape = tuple(cfg.task.image_shape)
    target_image_shape = (dataset_image_shape[0], 518, 518)

    if "ee_pose" in cfg.shape_meta.obs:
        state_key = "ee_pose"
        ee_pose_dim = int(cfg.shape_meta.obs.ee_pose.shape[0])
        rot_rep = str(cfg.shape_meta.obs.ee_pose.rotation_rep)
        required_keys = ["side_cam", "wrist_cam", "tcp_pose"]
    elif "qpos" in cfg.shape_meta.obs:
        state_key = "qpos"
        ee_pose_dim = int(cfg.shape_meta.obs.qpos.shape[0])
        rot_rep = ""
        required_keys = ["side_cam", "wrist_cam", "joint_pos", "action_mode"]
    else:
        raise RuntimeError(f"Unsupported observation state keys: {list(cfg.shape_meta.obs.keys())}")

    print(f"[TEST] training_config={args.training_config}")
    print(f"[TEST] checkpoint={args.checkpoint}")
    print(f"[TEST] dataset_zarr={dataset_zarr}")
    print(
        f"[TEST] n_obs_steps={n_obs_steps}, dataset_image_shape={dataset_image_shape}, "
        f"target_image_shape={target_image_shape}, state_key={state_key}"
    )

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

        replay_buffer = ReplayBuffer.copy_from_path(dataset_zarr, keys=required_keys)
        expert_ep_idx = _find_first_expert_episode(replay_buffer)
        episode = replay_buffer.get_episode(expert_ep_idx, copy=True)

        if episode["side_cam"].shape[0] < 1:
            raise RuntimeError(f"Episode {expert_ep_idx} is empty")

        expert_side_frame = episode["side_cam"][0]
        expert_wrist_frame = episode["wrist_cam"][0]

        if state_key == "ee_pose":
            tcp_first = episode["tcp_pose"][0].astype(np.float32)
            expert_state_first = _build_ee_pose_from_tcp_first_frame(
                tcp_pose_first=tcp_first,
                ee_pose_dim=ee_pose_dim,
                rot_rep=rot_rep,
            )
        else:
            if "joint_pos" in episode:
                expert_state_first = episode["joint_pos"][0].astype(np.float32)
            else:
                expert_state_first = episode["qpos"][0].astype(np.float32)

        rng = np.random.default_rng(args.noise_seed)
        noise_side_frame = rng.integers(0, 256, size=expert_side_frame.shape, dtype=np.uint8)
        noise_wrist_frame = rng.integers(0, 256, size=expert_wrist_frame.shape, dtype=np.uint8)
        noise_side_frame_2  = rng.integers(0, 256, size=expert_side_frame.shape, dtype=np.uint8)
        noise_wrist_frame_2 = rng.integers(0, 256, size=expert_wrist_frame.shape, dtype=np.uint8)
        expert_obs = _build_obs_dict(
            side_frame=expert_side_frame,
            wrist_frame=expert_wrist_frame,
            state_first=expert_state_first,
            n_obs_steps=n_obs_steps,
            target_image_shape=target_image_shape,
            state_key=state_key,
            device=device,
        )
        noise_obs = _build_obs_dict(
            side_frame=noise_side_frame,
            wrist_frame=noise_wrist_frame,
            state_first=expert_state_first,
            n_obs_steps=n_obs_steps,
            target_image_shape=target_image_shape,
            state_key=state_key,
            device=device,
        )
        noise_obs_2 = _build_obs_dict(
            side_frame=noise_side_frame_2,
            wrist_frame=noise_wrist_frame_2,
            state_first=expert_state_first,
            n_obs_steps=n_obs_steps,
            target_image_shape=target_image_shape,
            state_key=state_key,
            device=device,
        )
        # save the expert vs noise images for sanity check
        # import cv2
        # cv2.imwrite("expert_side.png", cv2.cvtColor(expert_side_frame, cv2.COLOR_RGB2BGR))
        # cv2.imwrite("expert_wrist.png", cv2.cvtColor(expert_wrist_frame, cv2.COLOR_RGB2BGR))
        # cv2.imwrite("noise_side.png", cv2.cvtColor(noise_side_frame, cv2.COLOR_RGB2BGR))
        # cv2.imwrite("noise_wrist.png", cv2.cvtColor(noise_wrist_frame, cv2.COLOR_RGB2BGR))
        
        with torch.no_grad():
            latent_expert = policy.extract_latent(expert_obs)  # (B, To, D)
            latent_noise = policy.extract_latent(noise_obs)    # (B, To, D)
            latent_noise_2 = policy.extract_latent(noise_obs_2)  # (B, To, D)

        flat_expert = latent_expert.reshape(1, -1)
        flat_noise = latent_noise.reshape(1, -1)
        flat_noise_2 = latent_noise_2.reshape(1, -1)
        cosine_sim = float(F.cosine_similarity(flat_expert, flat_noise, dim=1).item())
        cosine_dist = 1.0 - cosine_sim

        per_step_sim = F.cosine_similarity(latent_expert[0], latent_noise[0], dim=1).detach().cpu().numpy()
        per_step_dist = 1.0 - per_step_sim

        print("\n===== DINO Latent Discriminability Test =====")
        print(f"expert_episode_idx: {expert_ep_idx}")
        print(f"use_ema: {use_ema}")
        print(f"latent_shape: expert={tuple(latent_expert.shape)}, noise={tuple(latent_noise.shape)}, noise_2={tuple(latent_noise_2.shape)}")
        print(f"flattened_cosine_similarity: {cosine_sim:.6f}")
        print(f"flattened_cosine_distance: {cosine_dist:.6f}")
        print(f"per_step_cosine_similarity: {np.array2string(per_step_sim, precision=6, separator=', ')}")
        print(f"per_step_cosine_distance: {np.array2string(per_step_dist, precision=6, separator=', ')}")
        print(f"expert_latent_l2_norm: {float(torch.norm(flat_expert, p=2).item()):.6f}")
        print(f"noise_latent_l2_norm: {float(torch.norm(flat_noise, p=2).item()):.6f}")
        print(f"noise_2_latent_l2_norm: {float(torch.norm(flat_noise_2, p=2).item()):.6f}")

    finally:
        _destroy_process_group_safely()


if __name__ == "__main__":
    main()
