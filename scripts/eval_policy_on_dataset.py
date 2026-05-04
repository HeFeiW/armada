#!/usr/bin/env python3
"""Evaluate trained policy predictions on dataset episodes.

Usage:
  python3 scripts/eval_policy_on_dataset.py --cfg config.yaml --checkpoint path/to/checkpoint.pt \
      --dataset-path /path/to/train_dataset --device cpu --filter-action-mode all --max-episodes 50

This script loads the training workspace (using the same class in `cfg.training._target_`),
loads the policy, iterates dataset episodes and for each episode runs one prediction
from the initial `To` observation window and compares the predicted `Ta` actions to
the dataset ground-truth actions. It reports MSE per-episode and aggregate stats.
"""
import argparse
import os
import sys
from typing import Optional

import dill
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from armada.diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer
from armada.diffusion_policy.diffusion_policy.common.pytorch_util import dict_apply
from armada.utils.episode_manager import EpisodeManager
from hardware.my_device.macros import HUMAN, ROBOT

import hydra


def load_workspace_and_policy(cfg, checkpoint_path, device):
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)

    cls = hydra.utils.get_class(cfg.training._target_)
    # minimal args: (training_cfg, rank, world_size, device_id, device)
    workspace = cls(cfg.training, 0, 1, 0, device)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model.module
    if cfg.training.training.use_ema:
        policy = workspace.ema_model.module

    policy.to(device)
    policy.eval()

    # extract some params needed for EpisodeManager
    obs_feature_dim = policy.obs_feature_dim
    img_shape = cfg.training.task.image_shape
    return workspace, policy, obs_feature_dim, img_shape


def build_episode_manager(cfg, obs_feature_dim, img_shape, device, num_samples=1):
    action_dim = cfg.training.shape_meta.action.shape[0]
    action_rot_transformer = None
    obs_rot_transformer = None
    state_type = 'qpos'
    state_shape = cfg.training.shape_meta.obs.qpos.shape

    if 'rotation_rep' in cfg.training.shape_meta.action:
        from armada.diffusion_policy.diffusion_policy.model.common.rotation_transformer import RotationTransformer
        action_rot_transformer = RotationTransformer(from_rep='quaternion', to_rep=cfg.training.shape_meta.action.rotation_rep)

    if 'ee_pose' in cfg.training.shape_meta.obs:
        state_type = 'ee_pose'
        state_shape = cfg.training.shape_meta.obs.ee_pose.shape
        if 'rotation_rep' in cfg.training.shape_meta.obs.ee_pose:
            from armada.diffusion_policy.diffusion_policy.model.common.rotation_transformer import RotationTransformer
            obs_rot_transformer = RotationTransformer(from_rep='quaternion', to_rep=cfg.training.shape_meta.obs.ee_pose.rotation_rep)

    To = cfg.training.n_obs_steps
    Ta = cfg.Ta

    mgr = EpisodeManager(
        obs_rot_transformer=obs_rot_transformer,
        action_rot_transformer=action_rot_transformer,
        obs_feature_dim=obs_feature_dim,
        img_shape=img_shape,
        state_type=state_type,
        state_shape=state_shape,
        action_dim=action_dim,
        To=To,
        Ta=Ta,
        device=device,
        num_samples=num_samples,
    )
    return mgr


def evaluate_on_dataset(cfg, policy, episode_manager, replay_buffer: ReplayBuffer, device, filter_mode='all', max_episodes: Optional[int]=None):
    To = cfg.training.n_obs_steps
    Ta = cfg.Ta

    mses = []
    count = 0

    for i in range(int(replay_buffer.n_episodes)):
        if max_episodes is not None and count >= max_episodes:
            break

        ep = replay_buffer.get_episode(i)
        action_mode = ep.get('action_mode', None)
        if action_mode is not None and filter_mode in ('human', 'robot'):
            modes = np.array(action_mode)
            # decide majority mode of first Ta chunk
            first_mode = modes[:Ta]
            has_human = np.any(first_mode == HUMAN)
            has_robot = np.any(first_mode == ROBOT)
            if filter_mode == 'human' and not has_human:
                continue
            if filter_mode == 'robot' and not has_robot:
                continue

        # skip episodes too short
        if ep['action'].shape[0] < Ta or ep['side_cam'].shape[0] < To:
            continue

        # reset episode manager and fill initial To observations
        episode_manager.reset_observation_history()
        for t in range(To):
            side = ep['side_cam'][t] / 255.0
            wrist = ep['wrist_cam'][t] / 255.0
            state = ep['tcp_pose'][t] if episode_manager.state_type == 'ee_pose' else ep['joint_pos'][t]
            episode_manager.update_observation(side, wrist, state)

        policy_obs = episode_manager.get_policy_observation()
        with torch.no_grad():
            pred = policy.predict_action(policy_obs)

        # handle tuple returns
        if isinstance(pred, tuple) or isinstance(pred, list):
            pred = pred[0]

        np_pred = dict_apply(pred, lambda x: x.detach().to('cpu').numpy())
        pred_actions = np_pred['action']
        # pred_actions shape: (num_samples, Ta, action_dim) or (Ta, action_dim)
        if pred_actions.ndim == 3:
            pred_actions = pred_actions[0]

        gt_actions = ep['action'][:pred_actions.shape[0]]
        mse = float(np.mean((pred_actions - gt_actions) ** 2))
        mses.append(mse)
        count += 1

    mses = np.array(mses) if len(mses) > 0 else np.array([])
    return mses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, help='Path to hydra config YAML used for training')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint file')
    parser.add_argument('--dataset-path', required=True, help='Path to training dataset directory (contains replay_buffer.zarr)')
    parser.add_argument('--device', default='cpu', help='Device: cpu or cuda:0')
    parser.add_argument('--filter-action-mode', choices=['all', 'human', 'robot'], default='all')
    parser.add_argument('--max-episodes', type=int, default=50)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    device = torch.device(args.device)

    workspace, policy, obs_feature_dim, img_shape = load_workspace_and_policy(cfg, args.checkpoint, device)
    episode_manager = build_episode_manager(cfg, obs_feature_dim, img_shape, device)

    base_zarr_path = os.path.join(args.dataset_path, 'replay_buffer.zarr')
    replay_buffer = ReplayBuffer.copy_from_path(base_zarr_path, keys=None)

    mses = evaluate_on_dataset(cfg, policy, episode_manager, replay_buffer, device, filter_mode=args.filter_action_mode, max_episodes=args.max_episodes)

    if mses.size == 0:
        print('No episodes evaluated (check dataset, filter and lengths).')
        return

    print(f'Evaluated {mses.size} episodes')
    print(f'MSE mean: {mses.mean():.6f}, std: {mses.std():.6f}, min: {mses.min():.6f}, max: {mses.max():.6f}')

    # save results
    out_path = os.path.join(os.getcwd(), 'eval_policy_on_dataset_results.npz')
    np.savez(out_path, mses=mses)
    print(f'Saved results to {out_path}')


if __name__ == '__main__':
    main()
