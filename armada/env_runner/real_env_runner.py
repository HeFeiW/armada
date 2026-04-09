import sys
import os
import re
import time
import numpy as np
import torch
import dill
import cv2
from typing import Dict, List, Any, Optional, Tuple
from scipy.spatial.transform import Rotation as R
from omegaconf import DictConfig
import hydra

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from armada.diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer
from armada.diffusion_policy.diffusion_policy.common.pytorch_util import dict_apply
from armada.diffusion_policy.diffusion_policy.model.common.rotation_transformer import RotationTransformer
from hardware.my_device.macros import CAM_SERIAL, HUMAN, ROBOT
from armada.utils.episode_manager import EpisodeManager
from armada.utils.maniskill_dashboard import ManiSkillRolloutDashboard, SimpleConsoleRolloutUI
from armada.utils.maniskill_hil import ManiSkillHumanInLoopController
from armada.env_runner.base_env_runner import BaseEnvRunner

class RealEnvRunner(BaseEnvRunner):
    """Environment runner for single-robot case"""
    
    def __init__(self, 
                 cfg: DictConfig, 
                 rank: int, 
                 device_ids: List[int]):
        super().__init__(cfg, rank, device_ids)
        
        # Initialize components
        self._load_policy()
        self._setup_transformers()
        self._initialize_robot_env()
        self._initialize_episode_manager()
        self._initialize_replay_buffer()
        self.is_sim = getattr(self.cfg, 'env_backend', 'hardware') == 'maniskill'
        self.num_envs = int(getattr(getattr(self.cfg, 'maniskill', {}), 'num_envs', 1)) if self.is_sim else 1
        self.is_parallel_sim = self.is_sim and self.num_envs > 1
        self.sim_hil_controller = ManiSkillHumanInLoopController(
            dict(getattr(self.cfg, 'human_loop', {})) if self.is_sim else {}
        )
        self.console_ui = None
        human_loop_cfg = getattr(self.cfg, 'human_loop', None)
        self.sim_dashboard = None
        if self.is_sim and human_loop_cfg is not None and bool(getattr(human_loop_cfg, 'visualize', True)):
            self.sim_dashboard = ManiSkillRolloutDashboard(
                enabled=True,
                window_name=str(getattr(human_loop_cfg, 'window_name', 'ARMADA ManiSkill Rollout')),
            )

        print(
            f"[RUNNER] init env_backend={getattr(self.cfg, 'env_backend', 'hardware')} "
            f"num_envs={self.num_envs} parallel={self.is_parallel_sim} "
            f"human_loop_mode={getattr(human_loop_cfg, 'mode', 'n/a') if human_loop_cfg is not None else 'n/a'}",
        )

        self.max_episode_length = self._calculate_max_episode_length()
        
        # Initialize failure detection module if specified
        self.failure_detection_module = None
        if hasattr(cfg, 'failure_detection'):
            self._initialize_failure_detection_module()

        # Set random seed
        self.seed = cfg.seed
        np.random.seed(self.seed)
        
        # Set up output directory (for scene setup visualization)
        self._setup_output_directory()
        
        # Extract deployment round number, contained in save_buffer_path
        self.num_round = self._extract_round_number()
        
        # Episode number management
        self.episode_idx = 0 # rollout episode index
        self.saved_episode_idx = 0 # replay buffer episode index (only those saved to replay buffer)
        
    def _load_policy(self):
        """Load policy from checkpoint"""
        payload = torch.load(open(self.cfg.checkpoint_path, 'rb'), pickle_module=dill)
        
        # Initialize trained workspace
        cls = hydra.utils.get_class(self.cfg.training._target_)
        workspace = cls(self.cfg.training, self.rank, self.world_size, self.device_id, self.device)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        # Get trained policy from workspace
        self.policy = workspace.model.module
        if self.cfg.training.training.use_ema:
            self.policy = workspace.ema_model.module
        
        self.policy.to(self.device)
        self.policy.eval()
        
        # Extract policy parameters
        self.To = self.cfg.training.n_obs_steps
        self.Ta = self.cfg.Ta
        self.obs_feature_dim = self.policy.obs_feature_dim
        self.img_shape = self.cfg.training.task.image_shape
    
    def _setup_transformers(self):
        """Setup rotation transformers for action and observation spaces"""
        self.action_dim = self.cfg.training.shape_meta.action.shape[0]
        self.action_rot_transformer = None
        self.obs_rot_transformer = None
        
        # Check if there's need for transforming rotation representation
        if 'rotation_rep' in self.cfg.training.shape_meta.action:
            self.action_rot_transformer = RotationTransformer(
                from_rep='quaternion', # The robot env uses quaternion representation by default
                to_rep=self.cfg.training.shape_meta.action.rotation_rep
            )
        if 'ee_pose' in self.cfg.training.shape_meta.obs:
            self.ee_pose_dim = self.cfg.training.shape_meta.obs.ee_pose.shape[0]
            self.state_type = 'ee_pose'
            self.state_shape = self.cfg.training.shape_meta.obs.ee_pose.shape
            if 'rotation_rep' in self.cfg.training.shape_meta.obs.ee_pose:
                self.obs_rot_transformer = RotationTransformer(
                    from_rep='quaternion', 
                    to_rep=self.cfg.training.shape_meta.obs.ee_pose.rotation_rep
                )
        else:
            self.ee_pose_dim = self.cfg.training.shape_meta.obs.qpos.shape[0]
            self.state_type = 'qpos'
            self.state_shape = self.cfg.training.shape_meta.obs.qpos.shape
    
    def _initialize_robot_env(self):
        """Initialize robot environment"""
        env_backend = getattr(self.cfg, 'env_backend', 'hardware')
        if env_backend == 'maniskill':
            from hardware.maniskill_robot_env import ManiSkillRobotEnv

            self.robot_env = ManiSkillRobotEnv(
                camera_serial=CAM_SERIAL,
                img_shape=self.img_shape,
                fps=self.fps,
                maniskill_cfg=getattr(self.cfg, 'maniskill', None)
            )
        else:
            from hardware.robot_env import RobotEnv

            self.robot_env = RobotEnv(
                camera_serial=CAM_SERIAL,
                img_shape=self.img_shape,
                fps=self.fps
            )
    
    def _initialize_episode_manager(self):
        """Initialize episode manager"""
        self.episode_manager = self._build_episode_manager(
            num_samples=self.cfg.failure_detection.num_samples
        )

    def _build_episode_manager(self, num_samples: int = 1):
        return EpisodeManager(
            obs_rot_transformer=self.obs_rot_transformer,
            action_rot_transformer=self.action_rot_transformer,
            obs_feature_dim=self.obs_feature_dim,
            img_shape=self.img_shape,
            state_type=self.state_type,
            state_shape=self.state_shape,
            action_dim=self.action_dim,
            To=self.To,
            Ta=self.Ta,
            device=self.device,
            num_samples=num_samples
        )

    def _ui(self) -> Optional[SimpleConsoleRolloutUI]:
        return getattr(self, 'console_ui', None)

    def _ui_set_episode(self, episode_idx: int):
        ui = self._ui()
        if ui is not None:
            ui.set_episode(episode_idx)

    def _ui_update_env(self, env_idx: int, step: int, state: str, decision: str, mode: str = "", round_idx: Optional[int] = None):
        ui = self._ui()
        if ui is not None:
            ui.update_env(env_idx, step=step, state=state, decision=decision, mode=mode, round_idx=round_idx)

    def _ui_message(self, text: str, level: str = "info"):
        ui = self._ui()
        if ui is not None:
            ui.push_message(text, level=level)

    def _run_parallel_sim_episode(self) -> List[Optional[Dict[str, Any]]]:
        """Run one vectorized ManiSkill episode with per-env human-in-loop decisions."""
        print(f"Running parallel simulation episode with num_envs={self.num_envs}")
        self._ui_set_episode(self.episode_idx)

        self.robot_env.reset_robot(getattr(self.cfg, 'random_init', False), None)
        managers = [self._build_episode_manager(num_samples=1) for _ in range(self.num_envs)]
        episode_buffers = [
            {
                'tcp_pose': [],
                'joint_pos': [],
                'action': [],
                'action_mode': [],
                'wrist_cam': [],
                'side_cam': []
            }
            for _ in range(self.num_envs)
        ]

        steps = np.zeros((self.num_envs,), dtype=np.int32)
        env_finished = np.zeros((self.num_envs,), dtype=np.bool_)
        env_discard = np.zeros((self.num_envs,), dtype=np.bool_)
        env_teleop = np.zeros((self.num_envs,), dtype=np.bool_)
        teleop_steps = np.zeros((self.num_envs,), dtype=np.int32)
        teleop_last_p = [None for _ in range(self.num_envs)]
        teleop_last_r = [None for _ in range(self.num_envs)]

        for env_idx in range(self.num_envs):
            state = self.robot_env.get_robot_state(env_idx=env_idx)
            managers[env_idx].reset_observation_history()
            for _ in range(self.To):
                managers[env_idx].update_observation(
                    state['policy_side_img'] / 255.0,
                    state['policy_wrist_img'] / 255.0,
                    state['tcp_pose'] if self.state_type == 'ee_pose' else state['joint_pos']
                )
            managers[env_idx].initialize_pose(state['tcp_pose'][:3], state['tcp_pose'][3:])
            self._ui_update_env(env_idx, int(steps[env_idx]), 'rollout', 'policy', mode='policy')

        while not np.all(env_finished):
            active_envs = [
                i for i in range(self.num_envs)
                if (not env_finished[i]) and (not env_teleop[i]) and steps[i] < self.max_episode_length
            ]

            env_action_seq = {}

            for env_idx in active_envs:
                state = self.robot_env.get_robot_state(env_idx=env_idx)
                managers[env_idx].update_observation(
                    state['policy_side_img'] / 255.0,
                    state['policy_wrist_img'] / 255.0,
                    state['tcp_pose'] if self.state_type == 'ee_pose' else state['joint_pos']
                )
                policy_obs = managers[env_idx].get_policy_observation()
                with torch.no_grad():
                    curr_action = self.policy.predict_action(policy_obs)
                np_action_dict = dict_apply(curr_action, lambda x: x.detach().to('cpu').numpy())
                env_action_seq[env_idx] = np_action_dict['action']

            if self.sim_dashboard is not None:
                payloads = {}
                statuses = {}
                for env_idx in range(self.num_envs):
                    current_state = self.robot_env.get_robot_state(env_idx=env_idx)
                    payloads[env_idx] = {
                        'side_img': current_state['side_img_raw'],
                        'wrist_img': current_state['wrist_img_raw'],
                    }
                    statuses[env_idx] = {
                        'step': int(steps[env_idx]),
                        'mode': 'teleop' if env_teleop[env_idx] else 'policy',
                        'state': 'finished' if env_finished[env_idx] else ('discarded' if env_discard[env_idx] else 'running'),
                        'decision': 'manual' if self.sim_hil_controller.mode == 'manual' else 'auto',
                    }
                self.sim_dashboard.show(payloads, statuses, banner=f"Episode {self.episode_idx}")

            for step in range(self.Ta):
                if not env_action_seq:
                    break
                tcp_actions = {}
                gripper_actions = {}
                per_step_actions = {}

                for env_idx, action_seq in env_action_seq.items():
                    if env_finished[env_idx]:
                        continue
                    deployed_action, gripper_action, _curr_p, _curr_r, curr_p_action, curr_r_action = \
                        managers[env_idx].get_absolute_action_for_step(action_seq, step)
                    tcp_actions[env_idx] = deployed_action
                    gripper_actions[env_idx] = gripper_action[0]
                    per_step_actions[env_idx] = (curr_p_action, curr_r_action, gripper_action[0])

                if not tcp_actions:
                    continue

                self.robot_env.deploy_action_batch(tcp_actions, gripper_actions)

                for env_idx in list(tcp_actions.keys()):
                    state_data = self.robot_env.get_robot_state(env_idx=env_idx)
                    curr_p_action, curr_r_action, grip = per_step_actions[env_idx]
                    episode_buffers[env_idx]['wrist_cam'].append(state_data['demo_wrist_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                    episode_buffers[env_idx]['side_cam'].append(state_data['demo_side_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                    episode_buffers[env_idx]['tcp_pose'].append(state_data['tcp_pose'])
                    episode_buffers[env_idx]['joint_pos'].append(state_data['joint_pos'])
                    episode_buffers[env_idx]['action'].append(np.concatenate((curr_p_action, curr_r_action, [grip])))
                    episode_buffers[env_idx]['action_mode'].append(ROBOT)
                    steps[env_idx] += 1

                    if step >= self.Ta - self.To + 1:
                        managers[env_idx].update_observation(
                            state_data['policy_side_img'] / 255.0,
                            state_data['policy_wrist_img'] / 255.0,
                            state_data['tcp_pose'] if self.state_type == 'ee_pose' else state_data['joint_pos']
                        )

            terminated, truncated = self.robot_env.get_env_done_flags()
            for env_idx in range(self.num_envs):
                if env_finished[env_idx]:
                    continue
                if terminated[env_idx]:
                    env_finished[env_idx] = True
                    continue
                if truncated[env_idx] or steps[env_idx] >= self.max_episode_length:
                    if self.sim_dashboard is not None:
                        current_state = self.robot_env.get_robot_state(env_idx=env_idx)
                        self.sim_dashboard.show(
                            {env_idx: {'side_img': current_state['side_img_raw'], 'wrist_img': current_state['wrist_img_raw']}},
                            {env_idx: {
                                'step': int(steps[env_idx]),
                                'mode': 'policy',
                                'state': 'timeout' if truncated[env_idx] else 'max_step',
                                'decision': 'awaiting',
                                'error_reason': 'timeout / max step reached',
                            }},
                            banner=f"Episode {self.episode_idx} needs decision",
                            error_text=f"env={env_idx}, step={int(steps[env_idx])}, reason=timeout_or_max_step",
                        )
                    decision = self.sim_hil_controller.decide(
                        env_idx,
                        'timeout',
                        int(steps[env_idx]),
                        key_provider=(self.sim_dashboard.wait_for_key if self.sim_dashboard is not None else None),
                    )
                    self._ui_update_env(env_idx, int(steps[env_idx]), 'waiting for decision', decision.action, mode='decision')
                    if decision.action == 'continue':
                        env_teleop[env_idx] = False
                        self._ui_update_env(env_idx, int(steps[env_idx]), 'rollout', 'continue', mode='policy')
                        continue
                    if decision.action == 'discard':
                        env_discard[env_idx] = True
                        env_finished[env_idx] = True
                        self._ui_update_env(env_idx, int(steps[env_idx]), 'discarded', 'discard', mode='decision')
                        continue
                    if decision.action == 'finish':
                        env_finished[env_idx] = True
                        self._ui_update_env(env_idx, int(steps[env_idx]), 'finished', 'finish', mode='decision')
                        continue
                    # Enter teleop and initialize per-env teleop pose tracking.
                    self.robot_env.keyboard.infer = False
                    self.robot_env.keyboard.finish = False
                    self.robot_env.keyboard.discard = False
                    teleop_last_p[env_idx] = managers[env_idx].last_p[0].copy()
                    teleop_last_r[env_idx] = managers[env_idx].last_r[0]
                    teleop_steps[env_idx] = 0
                    env_teleop[env_idx] = True
                    self._ui_update_env(env_idx, int(steps[env_idx]), 'on decision', 'teleop', mode='teleop')

            # Progress teleop envs while others continue policy rollout.
            for env_idx in np.where(env_teleop)[0].tolist():
                self.robot_env.set_active_env(env_idx)
                curr_pose = teleop_last_p[env_idx] if teleop_last_p[env_idx] is not None else managers[env_idx].last_p[0]
                curr_rot = teleop_last_r[env_idx] if teleop_last_r[env_idx] is not None else managers[env_idx].last_r[0]
                teleop_data, new_last_p, new_last_r = self.robot_env.human_teleop_step(curr_pose, curr_rot)
                if teleop_data is not None:
                    teleop_last_p[env_idx] = new_last_p
                    teleop_last_r[env_idx] = new_last_r
                    episode_buffers[env_idx]['wrist_cam'].append(teleop_data['demo_wrist_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                    episode_buffers[env_idx]['side_cam'].append(teleop_data['demo_side_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                    episode_buffers[env_idx]['tcp_pose'].append(teleop_data['tcp_pose'])
                    episode_buffers[env_idx]['joint_pos'].append(teleop_data['joint_pos'])
                    episode_buffers[env_idx]['action'].append(teleop_data['action'])
                    episode_buffers[env_idx]['action_mode'].append(teleop_data['action_mode'])
                    managers[env_idx].update_observation(
                        teleop_data['policy_side_img'] / 255.0,
                        teleop_data['policy_wrist_img'] / 255.0,
                        teleop_data['tcp_pose'] if self.state_type == 'ee_pose' else teleop_data['joint_pos']
                    )
                    managers[env_idx].initialize_pose(new_last_p, new_last_r.as_quat(scalar_first=True))
                    steps[env_idx] += 1
                    self._ui_update_env(env_idx, int(steps[env_idx]), 'on decision', 'teleop', mode='teleop')
                elif self.robot_env.keyboard.quit:
                    env_finished[:] = True
                    break
                elif self.robot_env.keyboard.infer:
                    self.robot_env.keyboard.infer = False
                    env_teleop[env_idx] = False
                    teleop_steps[env_idx] = 0
                    self.sim_hil_controller.clear_blocked(env_idx)
                    self._ui_update_env(env_idx, int(steps[env_idx]), 'rollout', 'continue', mode='policy')
                    continue
                elif self.robot_env.keyboard.discard:
                    self.robot_env.keyboard.discard = False
                    env_discard[env_idx] = True
                    env_finished[env_idx] = True
                    env_teleop[env_idx] = False
                    teleop_steps[env_idx] = 0
                    self.sim_hil_controller.clear_blocked(env_idx)
                    self._ui_update_env(env_idx, int(steps[env_idx]), 'discarded', 'discard', mode='decision')
                    continue
                elif self.robot_env.keyboard.finish:
                    self.robot_env.keyboard.finish = False
                    env_finished[env_idx] = True
                    env_teleop[env_idx] = False
                    teleop_steps[env_idx] = 0
                    self.sim_hil_controller.clear_blocked(env_idx)
                    self._ui_update_env(env_idx, int(steps[env_idx]), 'finished', 'finish', mode='decision')
                    continue

                teleop_steps[env_idx] += 1
                if teleop_steps[env_idx] >= self.sim_hil_controller.max_teleop_steps:
                    force_release = self.sim_hil_controller.mark_blocked_round(env_idx)
                    if force_release:
                        env_teleop[env_idx] = False
                        teleop_steps[env_idx] = 0
                        self.sim_hil_controller.clear_blocked(env_idx)

                if self.sim_dashboard is not None:
                    current_state = self.robot_env.get_robot_state(env_idx=env_idx)
                    self.sim_dashboard.show(
                        {env_idx: {'side_img': current_state['side_img_raw'], 'wrist_img': current_state['wrist_img_raw']}},
                        {env_idx: {
                            'step': int(steps[env_idx]),
                            'mode': 'teleop',
                            'state': 'intervening',
                            'decision': 'teleop',
                        }},
                        banner=f"Episode {self.episode_idx} | env {env_idx} intervention",
                    )

                if steps[env_idx] >= self.max_episode_length:
                    env_finished[env_idx] = True

        episode_list = []
        for env_idx in range(self.num_envs):
            if env_discard[env_idx] or len(episode_buffers[env_idx]['action_mode']) == 0:
                episode_list.append(None)
                continue

            episode = dict()
            episode['wrist_cam'] = np.stack(episode_buffers[env_idx]['wrist_cam'], axis=0)
            episode['side_cam'] = np.stack(episode_buffers[env_idx]['side_cam'], axis=0)
            episode['tcp_pose'] = np.stack(episode_buffers[env_idx]['tcp_pose'], axis=0)
            episode['joint_pos'] = np.stack(episode_buffers[env_idx]['joint_pos'], axis=0)
            episode['action'] = np.stack(episode_buffers[env_idx]['action'], axis=0)
            episode['action_mode'] = np.array(episode_buffers[env_idx]['action_mode'])
            if episode['action_mode'].shape[0] % self.Ta != 0:
                trim = episode['action_mode'].shape[0] % self.Ta
                keep = episode['action_mode'].shape[0] - trim
                for key in ['wrist_cam', 'side_cam', 'tcp_pose', 'joint_pos', 'action', 'action_mode']:
                    episode[key] = episode[key][:keep]
            episode['failure_indices'] = np.zeros((episode['action_mode'].shape[0],), dtype=np.bool_)
            episode_list.append(episode)

        return episode_list
    
    def _initialize_replay_buffer(self):
        """Initialize replay buffer for data collection"""
        base_zarr_path = os.path.join(self.cfg.train_dataset_path, 'replay_buffer.zarr')
        self.replay_buffer = ReplayBuffer.copy_from_path(base_zarr_path, keys=None) # Build upon previous training set

        # Add action_mode if not present
        if 'action_mode' not in self.replay_buffer.keys():
            self.replay_buffer.data['action_mode'] = np.full((self.replay_buffer.n_steps, ), HUMAN)
        
        # Add failure_indices if not present
        if 'failure_indices' not in self.replay_buffer.keys():
            self.replay_buffer.data['failure_indices'] = np.zeros((self.replay_buffer.n_steps, ), dtype=np.bool_)
    
    def _initialize_failure_detection_module(self):
        """Initialize failure detection module based on config"""
        self.failure_detection_module = hydra.utils.instantiate(self.cfg.failure_detection)
        
        # Initialize the module using runtime variables
        self.failure_detection_module.runtime_initialize(
            device=self.device,
            policy=self.policy,
            replay_buffer=self.replay_buffer,
            episode_manager=self.episode_manager,
            max_episode_length=self.max_episode_length
        )
    
    def _setup_output_directory(self):
        """Setup output directory for scene configuration visualization, usually used for fair comparison between multiple policies"""
        self.save_img = False
        self.output_dir = os.path.join(self.cfg.output_dir, f"seed_{self.seed}")
        if os.path.isdir(self.output_dir): # Precise scene reset for fair policy comparison
            print(f"Output directory {self.output_dir} already exists, will not overwrite it.")
        else: # Start another round of deployment without reference scene setup
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
            self.save_img = True
        
        # Create save buffer directory
        os.makedirs(self.cfg.save_buffer_path, exist_ok=True)
    
    def _extract_round_number(self) -> int:
        """Extract deployment round number from save buffer path """
        match_round = re.search(r'round(\d)', self.cfg.save_buffer_path)
        if match_round:
            return int(match_round.group(1))
        return 0

    def _validate_episode_shapes(self, episode: Dict[str, Any]):
        """Validate one episode against replay buffer schema before appending."""
        for key, value in episode.items():
            if key not in self.replay_buffer.data:
                continue
            arr = self.replay_buffer.data[key]
            if value.shape[1:] != arr.shape[1:]:
                raise ValueError(
                    f"Episode key '{key}' shape mismatch: "
                    f"got tail={value.shape[1:]}, expected tail={arr.shape[1:]}, full={value.shape}"
                )
    
    def _calculate_max_episode_length(self) -> int:
        """Calculate maximum episode length based on expert demonstrations"""
        human_demo_indices = []
        for i in range(self.replay_buffer.n_episodes):
            episode_start = self.replay_buffer.episode_ends[i-1] if i > 0 else 0
            if np.any(self.replay_buffer.data['action_mode'][episode_start: self.replay_buffer.episode_ends[i]] == HUMAN):
                human_demo_indices.append(i)
        
        human_eps_len = []
        for i in human_demo_indices:
            human_episode = self.replay_buffer.get_episode(i)
            human_eps_len.append(human_episode['side_cam'].shape[0])
        
        return int(torch.max(torch.tensor(human_eps_len)) // self.Ta * self.Ta) # Truncate to the nearest multiple of Ta
    
    def run_rollout(self):
        """Main rollout loop"""
        try:
            while True:
                if self.robot_env.keyboard.quit:
                    print("[RUNNER] quit flag detected, stopping rollout loop")
                    break
                
                print(f"Rollout episode: {self.episode_idx}")
                
                # Run single episode
                try:
                    if self.is_parallel_sim:
                        episode_data = self._run_parallel_sim_episode()
                    else:
                        episode_data = self._run_single_episode()
                except Exception:
                    if self.sim_dashboard is not None:
                        import traceback

                        error_text = traceback.format_exc()
                        self.sim_dashboard.show_error(error_text, banner=f"Episode {self.episode_idx} failed")
                        print(error_text)
                    raise
                
                if episode_data is not None:
                    if isinstance(episode_data, list):
                        print(f"[RUNNER] collected {len([ep for ep in episode_data if ep is not None])} parallel episodes")
                        for ep in episode_data:
                            if ep is None:
                                print("[RUNNER] skipped discarded parallel episode")
                                continue
                            self._validate_episode_shapes(ep)
                            self.replay_buffer.add_episode(ep, compressors='disk')
                            self.saved_episode_idx = self.replay_buffer.n_episodes - 1
                            print(f'Saved episode {self.saved_episode_idx}')
                    else:
                        # Save episode to replay buffer
                        self._validate_episode_shapes(episode_data)
                        self.replay_buffer.add_episode(episode_data, compressors='disk')
                        self.saved_episode_idx = self.replay_buffer.n_episodes - 1
                        print(f'Saved episode {self.saved_episode_idx}')
                
                # Reset robot between episodes
                self.robot_env.reset_robot()
                print("Reset!")
                
                self.episode_idx += 1
                
                # For scene configuration reset
                time.sleep(5)
        
        finally:
            self._cleanup()
    
    def _run_single_episode(self) -> Optional[Dict[str, Any]]:
        """Run a single episode and return episode data"""
        self._ui_set_episode(self.episode_idx)
        # Reset keyboard states
        self.robot_env.keyboard.finish = False
        self.robot_env.keyboard.help = False
        self.robot_env.keyboard.infer = False
        self.robot_env.keyboard.discard = False
        time.sleep(1)
        
        # Initialize episode buffers
        self.episode_buffers = {
            'tcp_pose': [],
            'joint_pos': [],
            'action': [],
            'action_mode': [],
            'wrist_cam': [],
            'side_cam': []
        }
        
        # Reset robot
        random_init_pose = None
        if getattr(self.cfg, 'random_init', False):
            random_init_pose = self.robot_env.robot.init_pose + np.random.uniform(-0.1, 0.1, size=7)
            print(f"[RUNNER] random_init enabled, pose_offset={np.round(random_init_pose - self.robot_env.robot.init_pose, 4)}")
        
        robot_state = self.robot_env.reset_robot(getattr(self.cfg, 'random_init', False), random_init_pose)
        print(f"[RUNNER] episode={self.episode_idx} reset complete, initial j={self.j if hasattr(self, 'j') else 0}")
        self._ui_update_env(0, 0, 'rollout', 'policy', mode='policy')
        
        # Initialize episode manager
        self.episode_manager.reset_observation_history()
        
        # Update initial observations for policy obs buffer
        for _ in range(self.To):
            self.episode_manager.update_observation(
                robot_state['policy_side_img'] / 255.0,
                robot_state['policy_wrist_img'] / 255.0,
                robot_state['tcp_pose'] if self.state_type == 'ee_pose' else robot_state['joint_pos']
            )
        
        # Initialize target pose tracking
        if getattr(self.cfg, 'random_init', False) and random_init_pose is not None:
            self.episode_manager.initialize_pose(random_init_pose[:3], random_init_pose[3:])
        else:
            self.episode_manager.initialize_pose(self.robot_env.robot.init_pose[:3], self.robot_env.robot.init_pose[3:])
        
        # Scene alignment if there are reference scene setups
        if self.save_img or not os.path.isfile(os.path.join(self.output_dir, f"side_{self.episode_idx}.png")):
            self.robot_env.save_scene_images(self.output_dir, self.episode_idx)
        else:
            self.robot_env.align_scene_with_file(self.output_dir, self.episode_idx)
        
        # Initialize failure detection module step data
        if self.failure_detection_module:
            init_policy_obs = self.episode_manager.get_policy_observation() # Keep dim but only require the first sample
            for key, value in init_policy_obs.items():
                init_policy_obs[key] = value[0:1]
            with torch.no_grad():
                init_latent = self.policy.extract_latent(init_policy_obs)
                init_latent = init_latent.reshape(-1).to(dtype=torch.float32)
            self.failure_detection_module.process_step({
                'step_type': 'episode_start',
                'episode_idx': self.episode_idx,
                'rollout_init_latent': init_latent.unsqueeze(0) # For determining expert candidates
            })
        
        # Detach teleop device
        detach_pos, detach_rot = self.robot_env.detach_sigma()
        
        self.j = 0  # Episode timestep
        
        while True:
            if self.j >= self.max_episode_length:
                print("Maximum episode length reached, turning to human for help.")
                self.robot_env.keyboard.help = True
                self._ui_update_env(0, int(self.j), 'waiting for decision', 'teleop', mode='decision')
            
            # Policy inference loop
            self._run_policy_inference_loop()
            
            # Human intervention if requested
            if self.robot_env.keyboard.help:
                intervention_result = self._run_human_intervention(detach_pos, detach_rot)
                detach_pos, detach_rot = intervention_result['detach_pos'], intervention_result['detach_rot']

            if self.robot_env.keyboard.quit:
                print("[RUNNER] quit flag detected during human intervention, stopping episode")
                self.robot_env.keyboard.finish = True
                self._ui_update_env(0, int(self.j), 'finished', 'quit', mode='decision')
                break
            
            # Check if episode should finish
            if self.robot_env.keyboard.discard:
                return None
            
            if self.robot_env.keyboard.finish:
                break
        
        # Finalize episode
        if self.robot_env.keyboard.finish:
            episode_data = self._finalize_episode()
            return episode_data
        
        return None
    
    def _run_policy_inference_loop(self) -> Dict[str, Any]:
        """Run policy inference loop"""
        print("=========== Policy inference ============")
        
        while not self.robot_env.keyboard.finish and not self.robot_env.keyboard.discard and not self.robot_env.keyboard.help:
            start_time = time.time()
            
            # Get robot state and observations
            robot_state = self.robot_env.get_robot_state()

            if self.sim_dashboard is not None:
                self.sim_dashboard.show(
                    {0: {'side_img': robot_state['side_img_raw'], 'wrist_img': robot_state['wrist_img_raw']}},
                    {0: {
                        'step': int(self.j),
                        'mode': 'policy',
                        'state': 'running',
                        'decision': 'single-env',
                    }},
                    banner=f"Episode {self.episode_idx} | single-env policy rollout",
                )
            
            # Update observation history
            self.episode_manager.update_observation(
                robot_state['policy_side_img'] / 255.0,
                robot_state['policy_wrist_img'] / 255.0,
                robot_state['tcp_pose'] if self.state_type == 'ee_pose' else robot_state['joint_pos']
            )
            
            # Get action sequence for execution
            policy_obs = self.episode_manager.get_policy_observation()
            with torch.no_grad():
                if self.failure_detection_module and hasattr(self.failure_detection_module, 'should_stop_rewinding'):
                    curr_action, curr_latent = self.policy.predict_action(policy_obs, return_latent=True)
                else:
                    curr_action = self.policy.predict_action(policy_obs)
                    curr_latent = None
            
            # Get first Ta actions and execute on robot
            np_action_dict = dict_apply(curr_action, lambda x: x.detach().to('cpu').numpy())
            action_seq = np_action_dict['action']
            
            # Execute action sequence
            for step in range(self.Ta):
                if step > 0:
                    start_time = time.time()
                
                # Get robot state
                state_data = robot_state if step == 0 else self.robot_env.get_robot_state()
                
                # Get absolute action for this step
                deployed_action, gripper_action, curr_p, curr_r, curr_p_action, curr_r_action = \
                    self.episode_manager.get_absolute_action_for_step(action_seq, step)
                
                # Execute action on robot
                self.robot_env.deploy_action(deployed_action, gripper_action[0])
                
                # Save to episode buffers
                self.episode_buffers['wrist_cam'].append(state_data['demo_wrist_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                self.episode_buffers['side_cam'].append(state_data['demo_side_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                self.episode_buffers['tcp_pose'].append(state_data['tcp_pose'])
                self.episode_buffers['joint_pos'].append(state_data['joint_pos'])
                self.episode_buffers['action'].append(np.concatenate((curr_p_action, curr_r_action, [gripper_action[0]])))
                self.episode_buffers['action_mode'].append(ROBOT)
                
                # Update policy observation for the last few steps
                if step >= self.Ta - self.To + 1:
                    self.episode_manager.update_observation(
                        state_data['policy_side_img'] / 255.0,
                        state_data['policy_wrist_img'] / 255.0,
                        state_data['tcp_pose'] if self.state_type == 'ee_pose' else state_data['joint_pos']
                    )
                
                time.sleep(max(1 / self.fps - (time.time() - start_time), 0))
                self.j += 1
            
            # ================Detect failure===============
            if self.failure_detection_module:
                step_data = {
                    'step_type': 'policy_step',
                    'curr_latent': curr_latent,
                    'timestep': self.j,
                    'robot_state': robot_state
                }
                
                self.failure_detection_module.process_step(step_data)
                
                failure_flag, failure_reason, _ = self.failure_detection_module.detect_failure(
                    timestep=self.j,
                    max_episode_length=self.max_episode_length
                )
                print(f"[RUNNER] failure_detection step={self.j} flag={failure_flag} reason={failure_reason}")

                if self.j >= self.max_episode_length: # Making sure that every failure detection result is processed before raising timeout
                    failure_flag, failure_reason, _ = self.failure_detection_module.wait_for_final_results(self.j)

                print(f"=========== Global timestep: {self.j // self.Ta - 1} =============")
                
                if failure_flag or self.j >= self.max_episode_length:
                    if failure_flag:
                        print(f"Failure detected! Due to {failure_reason}")
                    else:
                        print("Maximum episode length reached!")

                    if self.sim_dashboard is not None:
                        self.sim_dashboard.show(
                            {0: {'side_img': robot_state['side_img_raw'], 'wrist_img': robot_state['wrist_img_raw']}},
                            {0: {
                                'step': int(self.j),
                                'mode': 'policy',
                                'state': 'failure' if failure_flag else 'max_step',
                                'decision': 'awaiting intervention',
                                'error_reason': failure_reason or 'maximum episode length reached',
                            }},
                            banner=f"Episode {self.episode_idx} requires intervention",
                            error_text=failure_reason or 'maximum episode length reached',
                        )

                    if self.is_sim:
                        # In simulation we avoid blocking for hardware keyboard input.
                        if self.j < self.max_episode_length and failure_flag:
                            print("[RUNNER] sim fallback -> entering human intervention due to failure")
                            self.robot_env.keyboard.help = True
                            break
                        print("[RUNNER] sim fallback -> entering human intervention due to timeout/max step")
                        self.robot_env.keyboard.help = True
                        break
                    else:
                        print("Press 'c' to continue; Press 'd' to discard the demo; Press 'h' to request human intervention; Press 'f' to finish the episode.")
                        while not self.robot_env.keyboard.ctn and not self.robot_env.keyboard.discard and not self.robot_env.keyboard.help and not self.robot_env.keyboard.finish:
                            time.sleep(0.1)

                        if self.robot_env.keyboard.ctn and self.j < self.max_episode_length:
                            print("False Positive failure! Continue policy rollout.")
                            self.robot_env.keyboard.ctn = False
                        elif self.robot_env.keyboard.ctn and self.j >= self.max_episode_length:
                            print("Cannot continue policy rollout, maximum episode length reached. Calling for human intervention.")
                            self.robot_env.keyboard.ctn = False
                            self.robot_env.keyboard.help = True
                            break
            # ================Detect failure end===============
            
            # Check for maximum episode length without failure detection
            elif self.j >= self.max_episode_length:
                print("Maximum episode length reached, turning to human for help.")
                self.robot_env.keyboard.help = True
                break
        
        return
    
    def _run_human_intervention(self, detach_pos: np.ndarray, detach_rot: R) -> Dict[str, Any]:
        """Run human intervention loop"""
        print("============ Human intervention =============")
        print(f"[RUNNER] intervention entry j={self.j} detach_pos={np.round(detach_pos, 4)}")
        self._ui_update_env(0, int(self.j), 'waiting for decision', 'teleop', mode='decision')
        
        # Reset intervention signals to avoid stale state skipping teleop loop.
        self.robot_env.keyboard.help = False
        self.robot_env.keyboard.infer = False
        self.robot_env.keyboard.finish = False
        self.robot_env.keyboard.discard = False
        
        # Perform rewinding if needed
        if self.failure_detection_module and hasattr(self.failure_detection_module, 'should_stop_rewinding'):
            curr_pos, curr_rot = self._rewind_robot()
            print("[RUNNER] rewind completed by failure detector / fallback path")
        
        # Get current pose for human teleop
        last_p = curr_pos if 'curr_pos' in locals() else self.episode_manager.last_p[0]
        last_r = curr_rot if 'curr_rot' in locals() else self.episode_manager.last_r[0]
        
        # Transform sigma device from current robot pose
        translate = last_p - detach_pos
        rotation = detach_rot.inv() * last_r
        self.robot_env.sigma.resume()
        self.robot_env.sigma.transform_from_robot(translate, rotation)
        print(f"[RUNNER] teleop transform translate={np.round(translate, 4)}")
        self._ui_update_env(0, int(self.j), 'on decision', 'teleop', mode='teleop')
        
        # Human intervention loop
        while not (self.robot_env.keyboard.finish or self.robot_env.keyboard.discard or self.robot_env.keyboard.infer or self.robot_env.keyboard.quit):
            # Execute one step of human teleop
            teleop_data, last_p, last_r = self.robot_env.human_teleop_step(last_p, last_r)
            print(f"[RUNNER] Teleop data is None: {teleop_data is None}")
            if teleop_data is None:
                if self.robot_env.keyboard.quit or self.robot_env.keyboard.finish or self.robot_env.keyboard.discard or self.robot_env.keyboard.infer:
                    print("[RUNNER] teleop exit requested -> leaving human intervention")
                    break
                print("[RUNNER] teleop step returned None -> simulated teleop fallback / waiting")
                if self.sim_dashboard is not None:
                    self.sim_dashboard.show(
                        {0: {'side_img': self.robot_env.get_robot_state()['side_img_raw'], 'wrist_img': self.robot_env.get_robot_state()['wrist_img_raw']}},
                        {0: {
                            'step': int(self.j),
                            'mode': 'teleop',
                            'state': 'waiting',
                            'decision': 'auto fallback',
                        }},
                        banner=f"Episode {self.episode_idx} | waiting for simulated teleop",
                    )
                continue
            
            # Update observation history with latest state
            self.episode_manager.update_observation(
                teleop_data['policy_side_img'] / 255.0,
                teleop_data['policy_wrist_img'] / 255.0,
                teleop_data['tcp_pose'] if self.state_type == 'ee_pose' else teleop_data['joint_pos']
            )
            
            # Store demo data
            self.episode_buffers['wrist_cam'].append(teleop_data['demo_wrist_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
            self.episode_buffers['side_cam'].append(teleop_data['demo_side_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
            self.episode_buffers['tcp_pose'].append(teleop_data['tcp_pose'])
            self.episode_buffers['joint_pos'].append(teleop_data['joint_pos'])
            self.episode_buffers['action'].append(teleop_data['action'])
            self.episode_buffers['action_mode'].append(teleop_data['action_mode'])
            
            self.j += 1
            print(f"[RUNNER] teleop step accepted, new j={self.j}")
            self._ui_update_env(0, int(self.j), 'on decision', 'teleop', mode='teleop')

            if self.sim_dashboard is not None:
                self.sim_dashboard.show(
                    {0: {'side_img': teleop_data['demo_side_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8), 'wrist_img': teleop_data['demo_wrist_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8)}},
                    {0: {
                        'step': int(self.j),
                        'mode': 'teleop',
                        'state': 'intervening',
                        'decision': 'teleop',
                    }},
                    banner=f"Episode {self.episode_idx} | human intervention",
                )
        
        # Reset target pose tracking after human intervention
        self.episode_manager.initialize_pose(last_p, last_r.as_quat(scalar_first=True))

        # Drop stale async failure-detection tasks/results generated before intervention.
        if self.failure_detection_module and hasattr(self.failure_detection_module, 'empty_queue'):
            self.failure_detection_module.empty_queue()
        if self.failure_detection_module and hasattr(self.failure_detection_module, 'empty_result_queue'):
            self.failure_detection_module.empty_result_queue()
        
        # Reset signals
        self.robot_env.keyboard.infer = False
        new_detach_pos, new_detach_rot = self.robot_env.detach_sigma()
        print(f"[RUNNER] intervention exit new_detach_pos={np.round(new_detach_pos, 4)}")
        
        return {'detach_pos': new_detach_pos, 'detach_rot': new_detach_rot}
    
    def _rewind_robot(self) -> Tuple[np.ndarray, R]:
        """Rewind the robot for human intervention"""
        print("Rewinding robot...")
        
        # Use the last target action for rewinding
        curr_pos = self.episode_manager.last_p[0]
        curr_rot = self.episode_manager.last_r[0]
        
        # Let failure detection module determine rewinding behavior
        if self.failure_detection_module and hasattr(self.failure_detection_module, 'should_stop_rewinding'):
            prev_side_cam, prev_wrist_cam, curr_pos, curr_rot = self._rewind_with_failure_detection(curr_pos, curr_rot)
        else:
            prev_side_cam, prev_wrist_cam, curr_pos, curr_rot = self._rewind_simple(curr_pos, curr_rot)
        print("[RUNNER] rewind finished, preparing reference alignment")
        
        # Prepare for human intervention by showing reference scene
        if "prev_side_cam" in locals():
            print("Please reset the scene and press 'c' to go on to human intervention")
            ref_side_img = cv2.cvtColor(prev_side_cam, cv2.COLOR_RGB2BGR)
            ref_wrist_img = cv2.cvtColor(prev_wrist_cam, cv2.COLOR_RGB2BGR)
            self.robot_env.align_with_reference(ref_side_img, ref_wrist_img)
        
        return curr_pos, curr_rot
    
    def _rewind_with_failure_detection(self, curr_pos: np.ndarray, curr_rot: R) -> Tuple[np.ndarray, np.ndarray, np.ndarray, R]:
        """Rewind with failure detection module guidance"""
        curr_timestep = self.j
        prev_side_cam = None
        prev_wrist_cam = None
        print(f"[RUNNER] rewind_with_failure_detection start j={self.j}")
        
        for _ in range(curr_timestep):
            if not self.failure_detection_module.rewind_step(self.j, self.episode_buffers, curr_timestep):
                print(f"[RUNNER] rewind stopped by failure detector at j={self.j}")
                break
            # Rewind one step on robot
            curr_pos, curr_rot, prev_side_cam, prev_wrist_cam = self._rewind_single_step(curr_pos, curr_rot)
            self.j -= 1
            print(f"[RUNNER] rewind step complete, j={self.j}")

        if prev_side_cam is None or prev_wrist_cam is None:
            if len(self.episode_buffers['side_cam']) > 0 and len(self.episode_buffers['wrist_cam']) > 0:
                prev_side_cam = self.episode_buffers['side_cam'][-1]
                prev_wrist_cam = self.episode_buffers['wrist_cam'][-1]
                print("[RUNNER] rewind fallback -> using latest buffered frame as reference")
            else:
                state_data = self.robot_env.get_robot_state()
                prev_side_cam = state_data['side_img_raw']
                prev_wrist_cam = state_data['wrist_img_raw']
                print("[RUNNER] rewind fallback -> using current env render as reference")
        
        return prev_side_cam, prev_wrist_cam, curr_pos, curr_rot
    
    def _rewind_simple(self, curr_pos: np.ndarray, curr_rot: R) -> Tuple[np.ndarray, np.ndarray, np.ndarray, R]:
        """Simple rewinding with step limit"""
        curr_timestep = self.j
        prev_side_cam = None
        prev_wrist_cam = None
        print(f"[RUNNER] rewind_simple start j={self.j}")
        
        for i in range(curr_timestep):
            # Simple stop condition: limit to 3 Ta-step chunks
            if self.j % self.Ta == 0 and self.j > 0:
                if i // self.Ta >= 3:
                    print("Stop rewinding (reached 3 Ta-step limit).")
                    print(f"[RUNNER] rewind_simple stop reason=step_limit i={i} j={self.j}")
                    break
            
            # Rewind one step
            curr_pos, curr_rot, prev_side_cam, prev_wrist_cam = self._rewind_single_step(curr_pos, curr_rot)
            self.j -= 1
            print(f"[RUNNER] rewind_simple step complete, j={self.j}")

        if prev_side_cam is None or prev_wrist_cam is None:
            if len(self.episode_buffers['side_cam']) > 0 and len(self.episode_buffers['wrist_cam']) > 0:
                prev_side_cam = self.episode_buffers['side_cam'][-1]
                prev_wrist_cam = self.episode_buffers['wrist_cam'][-1]
                print("[RUNNER] rewind_simple fallback -> using latest buffered frame as reference")
            else:
                state_data = self.robot_env.get_robot_state()
                prev_side_cam = state_data['side_img_raw']
                prev_wrist_cam = state_data['wrist_img_raw']
                print("[RUNNER] rewind_simple fallback -> using current env render as reference")
        
        return prev_side_cam, prev_wrist_cam, curr_pos, curr_rot
    
    def _rewind_single_step(self, curr_pos: np.ndarray, curr_rot: R) -> Tuple[np.ndarray, R, np.ndarray, np.ndarray]:
        """Rewind a single step"""
        # Get previous action data to rewind
        prev_wrist_cam = self.episode_buffers['wrist_cam'].pop()
        prev_side_cam = self.episode_buffers['side_cam'].pop()
        self.episode_buffers['tcp_pose'].pop()
        self.episode_buffers['joint_pos'].pop()
        prev_action = self.episode_buffers['action'].pop()
        self.episode_buffers['action_mode'].pop()
        
        # Rewind robot by applying inverse action
        curr_pos, curr_rot = self.robot_env.rewind_robot(curr_pos, curr_rot, prev_action)
        print(f"[RUNNER] rewind_single_step applied inverse action, remaining buffers={len(self.episode_buffers['action'])}")
        
        return curr_pos, curr_rot, prev_side_cam, prev_wrist_cam
    
    def _finalize_episode(self) -> Dict[str, Any]:
        """Finalize episode and return episode data"""
        episode = dict()
        episode['wrist_cam'] = np.stack(self.episode_buffers['wrist_cam'], axis=0)
        episode['side_cam'] = np.stack(self.episode_buffers['side_cam'], axis=0)
        episode['tcp_pose'] = np.stack(self.episode_buffers['tcp_pose'], axis=0)
        episode['joint_pos'] = np.stack(self.episode_buffers['joint_pos'], axis=0)
        episode['action'] = np.stack(self.episode_buffers['action'], axis=0)
        episode['action_mode'] = np.array(self.episode_buffers['action_mode'])
        
        assert episode['action_mode'].shape[0] % self.Ta == 0, "A Ta-step chunking is required for the entire demo"
        
        # Finalize failure detection
        if self.failure_detection_module:
            failure_episode_data = self.failure_detection_module.finalize_episode(episode)
            episode.update(failure_episode_data)
        else:
            # Default: no failure indices
            episode['failure_indices'] = np.zeros((episode['action_mode'].shape[0],), dtype=np.bool_)
        
        return episode
    
    def _cleanup(self):
        """Cleanup resources"""
        # Save the replay buffer
        save_zarr_path = os.path.join(self.cfg.save_buffer_path, 'replay_buffer.zarr')
        self.replay_buffer.save_to_path(save_zarr_path)
        
        # Cleanup failure detection module
        if self.failure_detection_module and hasattr(self.failure_detection_module, 'cleanup'):
            self.failure_detection_module.cleanup()

        if self.sim_dashboard is not None:
            self.sim_dashboard.close()
        
        print("Saved replay buffer to", save_zarr_path)
        print("[RUNNER] cleanup complete")
        torch.distributed.destroy_process_group() 