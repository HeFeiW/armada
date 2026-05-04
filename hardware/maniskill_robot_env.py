import time
import os
import select
import sys
import re
from itertools import permutations, product
from types import SimpleNamespace
from typing import Any, Dict, Optional

import cv2
import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Resize
from maniskill_armada.data_utils import extract_action_from_poses
from maniskill_armada.heuristic_teleop_node import HeuristicTeleopNode

from hardware.my_device.macros import INTV


class _SimSigma:
    def reset(self):
        return

    def detach(self):
        return

    def resume(self):
        return

    def transform_from_robot(self, _translate, _rotation):
        return


class ManiSkillRobotEnv:
    """RobotEnv-compatible ManiSkill adapter for policy rollout simulation."""

    def __init__(
        self,
        camera_serial=None,
        img_shape=None,
        fps=10,
        maniskill_cfg: Optional[Any] = None,
    ):
        self.camera_serial = camera_serial
        self.fps = fps
        self.img_shape = img_shape

        cfg = maniskill_cfg or {}
        self.env_id = cfg.get("env_id", "PickCube-v1")
        self.num_envs = int(cfg.get("num_envs", 1))
        self.obs_mode = cfg.get("obs_mode", "state_dict")
        self.control_mode = cfg.get("control_mode", "pd_ee_delta_pose")
        self.render_mode = cfg.get("render_mode", "rgb_array")
        self.max_episode_steps = int(cfg.get("max_episode_steps", 300))
        self.seed = int(cfg.get("seed", 0))
        self.side_camera_name = cfg.get("side_camera_name", "base_camera")
        self.wrist_camera_name = cfg.get("wrist_camera_name", "base_camera")
        self.auto_select_cameras = bool(cfg.get("auto_select_cameras", True))
        self.wrist_camera_follow_tcp = bool(cfg.get("wrist_camera_follow_tcp", False))
        self.wrist_camera_tcp_pos_offset = np.asarray(
            cfg.get("wrist_camera_tcp_pos_offset", [0.0, 0.0, 0.0]), dtype=np.float32
        ).reshape(3)
        self.wrist_camera_tcp_quat_wxyz_offset = np.asarray(
            cfg.get("wrist_camera_tcp_quat_wxyz_offset", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32
        ).reshape(4)
        self.max_snapshot_steps = int(cfg.get("max_snapshot_steps", 5000))
        self.auto_intervention_steps = int(cfg.get("auto_intervention_steps", 8))
        self.gripper_max_width = float(cfg.get("gripper_max_width", 0.09))
        self.teleop_pos_step = float(cfg.get("teleop_pos_step", 0.01))
        self.teleop_rot_step_deg = float(cfg.get("teleop_rot_step_deg", 6.0))
        self.teleop_gripper_step = float(cfg.get("teleop_gripper_step", 0.01))
        self.teleop_rotation_world_frame = bool(cfg.get("teleop_rotation_world_frame", True))
        self.teleop_feedback_from_measured_pose = bool(cfg.get("teleop_feedback_from_measured_pose", True))
        self.teleop_show_help = bool(cfg.get("teleop_show_help", True))
        self.teleop_use_cv2_keys = bool(cfg.get("teleop_use_cv2_keys", True))
        self.teleop_input_mode = str(cfg.get("teleop_input_mode", "keyboard")).lower()
        self.teleop_heuristic_cfg = cfg.get("teleop_heuristic", {}) or {}
        self.teleop_terminal_poll_timeout_s = float(cfg.get("teleop_terminal_poll_timeout_s", 0.2))
        self.teleop_no_input_log_interval_s = float(cfg.get("teleop_no_input_log_interval_s", 2.0))
        self.success_use_env_info = bool(cfg.get("success_use_env_info", True))
        self.success_use_pose_check = bool(cfg.get("success_use_pose_check", True))
        self.success_require_static = bool(cfg.get("success_require_static", False))
        self.success_pos_tolerance = float(cfg.get("success_pos_tolerance", 0.03))
        self.success_rot_tolerance_deg = float(cfg.get("success_rot_tolerance_deg", 20.0))
        self.success_target_pose_wxyz = np.asarray(
            cfg.get(
                "success_target_pose_wxyz",
                [0.02510096, -0.00136661, 0.2814997, 0.7723232, 0.61468965, 0.15892147, 0.02043187],
            ),
            dtype=np.float32,
        ).reshape(7)
        self.debug_camera_enable = bool(cfg.get("debug_camera_enable", False))
        self.debug_side_camera_offset = np.asarray(
            cfg.get("debug_side_camera_offset", [0.0, 0.0, 0.0]), dtype=np.float32
        ).reshape(3)
        self.debug_wrist_camera_offset = np.asarray(
            cfg.get("debug_wrist_camera_offset", [0.0, 0.0, 0.0]), dtype=np.float32
        ).reshape(3)
        self._debug_camera_applied = False
        self._has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

        self.env = gym.make(
            self.env_id,
            num_envs=self.num_envs,
            obs_mode=self.obs_mode,
            control_mode=self.control_mode,
            render_mode=self.render_mode,
            max_episode_steps=self.max_episode_steps,
        )
        self.single_action_space = getattr(self.env, "single_action_space", self.env.action_space)

        self.keyboard = SimpleNamespace(
            quit=False,
            finish=False,
            help=False,
            infer=False,
            discard=False,
            ctn=False,
        )
        self.sigma = _SimSigma()

        self.gripper = SimpleNamespace(max_width=self.gripper_max_width)
        self.robot = SimpleNamespace(init_pose=np.zeros(7, dtype=np.float32))

        # Match hardware image preprocessing path used by rollout policy.
        bicubic = InterpolationMode.BICUBIC
        self.policy_side_image_processor = Compose(
            [
                Resize((img_shape[1] + 8, img_shape[2] + 8), interpolation=bicubic, antialias=True),
                CenterCrop((img_shape[1], img_shape[2])),
            ]
        )
        self.policy_wrist_image_processor = Compose(
            [
                Resize((img_shape[1] + 8, img_shape[2] + 8), interpolation=bicubic, antialias=True),
                CenterCrop((img_shape[1], img_shape[2])),
            ]
        )
        self.demo_image_processor = Compose(
            [
                Resize((img_shape[1] + 8, img_shape[2] + 8), interpolation=bicubic, antialias=True),
                CenterCrop((img_shape[1], img_shape[2])),
            ]
        )

        self.last_obs = None
        self.last_info = None
        self.last_done = False
        self.last_render = None
        self.last_tcp_pose = None
        self.last_joint_pos = None
        self.last_gripper_width = self.gripper_max_width
        self._teleop_steps = 0
        self._snapshots = []
        self.active_env_idx = 0
        self.last_terminated = np.zeros((self.num_envs,), dtype=np.bool_)
        self.last_truncated = np.zeros((self.num_envs,), dtype=np.bool_)
        self._teleop_help_printed = False
        self._stdin_hint_printed = False
        self._stdin_unavailable_warned = False
        self._last_no_input_log_time = 0.0
        self._cube_symmetry_rotations = self._build_cube_symmetry_rotations()
        self._heuristic_teleop_node = None
        self._camera_names_auto_selected = False
        self._available_camera_names_last = []

    def _build_cube_symmetry_rotations(self):
        symmetries = []
        basis = np.eye(3, dtype=np.float32)
        for perm in permutations(range(3)):
            permuted = basis[:, perm]
            for signs in product((-1.0, 1.0), repeat=3):
                mat = permuted * np.asarray(signs, dtype=np.float32)
                if np.isclose(np.linalg.det(mat), 1.0, atol=1e-5):
                    symmetries.append(R.from_matrix(mat))

        unique = []
        seen = set()
        for rot in symmetries:
            key = tuple(np.round(rot.as_quat(scalar_first=True), 6))
            if key not in seen:
                seen.add(key)
                unique.append(rot)
        return unique

    def _try_apply_sensor_pose_offset(self, sensor_obj: Any, offset: np.ndarray) -> bool:
        if sensor_obj is None:
            return False

        pose_owner = sensor_obj
        pose = getattr(sensor_obj, "pose", None)
        if pose is None and hasattr(sensor_obj, "camera"):
            pose_owner = getattr(sensor_obj, "camera")
            pose = getattr(pose_owner, "pose", None)
        if pose is None:
            return False

        try:
            pos = np.asarray(getattr(pose, "p"), dtype=np.float32)
            quat = np.asarray(getattr(pose, "q"), dtype=np.float32)
            if pos.ndim == 1:
                pos_new = pos + offset
            else:
                pos_new = pos + offset.reshape(1, 3)

            pose_cls = type(pose)
            new_pose = None
            if hasattr(pose_cls, "create_from_pq"):
                new_pose = pose_cls.create_from_pq(pos_new, quat)
            else:
                try:
                    new_pose = pose_cls(pos_new, quat)
                except Exception:
                    return False

            if hasattr(pose_owner, "set_pose"):
                pose_owner.set_pose(new_pose)
                return True
            if hasattr(pose_owner, "pose"):
                pose_owner.pose = new_pose
                return True
        except Exception:
            return False
        return False

    def _apply_debug_camera_offsets_once(self):
        if not self.debug_camera_enable or self._debug_camera_applied:
            return

        sensors = getattr(self.env.unwrapped, "_sensors", None)
        if not isinstance(sensors, dict):
            sensors = getattr(self.env.unwrapped, "sensors", None)
        if not isinstance(sensors, dict):
            print("[SIM TELEOP] Debug camera offset requested but sensor dictionary was not found.")
            return

        side_ok = self._try_apply_sensor_pose_offset(sensors.get(self.side_camera_name), self.debug_side_camera_offset)
        wrist_ok = self._try_apply_sensor_pose_offset(sensors.get(self.wrist_camera_name), self.debug_wrist_camera_offset)
        self._debug_camera_applied = side_ok or wrist_ok
        if self._debug_camera_applied:
            print(
                f"[SIM TELEOP] Applied debug camera offsets: side={self.debug_side_camera_offset.tolist()}, "
                f"wrist={self.debug_wrist_camera_offset.tolist()}"
            )
        else:
            print("[SIM TELEOP] Debug camera offsets could not be applied on current ManiSkill sensor objects.")

    def _get_sensor_dict(self) -> Optional[Dict[str, Any]]:
        sensors = getattr(self.env.unwrapped, "_sensors", None)
        if not isinstance(sensors, dict):
            sensors = getattr(self.env.unwrapped, "sensors", None)
        if isinstance(sensors, dict):
            return sensors
        return None

    def _list_available_camera_names_from_obs(self, obs: Any) -> list:
        if not isinstance(obs, dict):
            return []
        sensor_data = obs.get("sensor_data", {})
        if not isinstance(sensor_data, dict):
            return []
        names = []
        for name, payload in sensor_data.items():
            if isinstance(payload, dict) and ("rgb" in payload):
                names.append(str(name))
        return names

    def _pick_camera_by_keywords(self, names: list, keywords: list, exclude: Optional[str] = None) -> Optional[str]:
        if not names:
            return None
        exclude = str(exclude) if exclude is not None else None
        pattern = re.compile("|".join([re.escape(k) for k in keywords]), flags=re.IGNORECASE)

        scored = []
        for name in names:
            if exclude is not None and name == exclude:
                continue
            m = pattern.search(name)
            if m:
                scored.append((m.start(), len(name), name))
        if scored:
            scored.sort()
            return scored[0][2]

        for name in names:
            if exclude is not None and name == exclude:
                continue
            return name
        return None

    def _auto_select_camera_names_once(self):
        if self._camera_names_auto_selected or (not self.auto_select_cameras):
            return

        available = self._list_available_camera_names_from_obs(self.last_obs)
        self._available_camera_names_last = available
        if not available:
            # Can't auto-detect without sensor_data; keep user-provided names.
            self._camera_names_auto_selected = True
            return

        # If both names resolve to the same camera, try to pick a better wrist camera.
        side_name = str(self.side_camera_name)
        wrist_name = str(self.wrist_camera_name)

        if side_name not in available:
            side_candidate = self._pick_camera_by_keywords(
                available,
                keywords=["base", "side", "front", "external", "main", "viewer"],
                exclude=None,
            )
            if side_candidate is not None:
                side_name = side_candidate

        if (wrist_name not in available) or (wrist_name == side_name):
            wrist_candidate = self._pick_camera_by_keywords(
                available,
                keywords=["wrist", "hand", "ee", "tcp", "gripper", "end", "effector"],
                exclude=side_name,
            )
            if wrist_candidate is not None:
                wrist_name = wrist_candidate

        # Persist selection.
        self.side_camera_name = side_name
        self.wrist_camera_name = wrist_name
        self._camera_names_auto_selected = True

        if self.side_camera_name == self.wrist_camera_name:
            print(
                "[SIM TELEOP][WARN] side_camera_name and wrist_camera_name resolve to the same sensor "
                f"('{self.side_camera_name}'). Available cameras: {available}. "
                "If you expect an end-effector camera, ensure the ManiSkill env defines a second camera sensor "
                "(often named like 'hand_camera' / 'wrist_camera') and set wrist_camera_name accordingly."
            )
        else:
            print(
                "[SIM TELEOP] Camera selection: "
                f"side='{self.side_camera_name}', wrist='{self.wrist_camera_name}'. "
                f"Available cameras: {available}"
            )

    def _set_sensor_pose(self, sensor_obj: Any, pos: np.ndarray, quat_wxyz: np.ndarray) -> bool:
        if sensor_obj is None:
            return False
        pose_owner = sensor_obj
        pose = getattr(sensor_obj, "pose", None)
        if pose is None and hasattr(sensor_obj, "camera"):
            pose_owner = getattr(sensor_obj, "camera")
            pose = getattr(pose_owner, "pose", None)
        if pose is None:
            return False

        pos = np.asarray(pos, dtype=np.float32)
        quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)

        try:
            pose_cls = type(pose)
            if hasattr(pose_cls, "create_from_pq"):
                new_pose = pose_cls.create_from_pq(pos, quat_wxyz)
            else:
                new_pose = pose_cls(pos, quat_wxyz)
        except Exception:
            return False

        try:
            if hasattr(pose_owner, "set_pose"):
                pose_owner.set_pose(new_pose)
                return True
            if hasattr(pose_owner, "pose"):
                pose_owner.pose = new_pose
                return True
        except Exception:
            return False
        return False

    def _maybe_update_wrist_camera_pose_from_tcp(self):
        if not self.wrist_camera_follow_tcp:
            return
        sensors = self._get_sensor_dict()
        if not isinstance(sensors, dict):
            return
        wrist_sensor = sensors.get(self.wrist_camera_name)
        if wrist_sensor is None:
            return

        # Avoid mutating the same sensor used for side view.
        if str(self.wrist_camera_name) == str(self.side_camera_name):
            return

        # Prefer batch TCP pose from obs for vectorized envs.
        tcp_pose = None
        if isinstance(self.last_obs, dict):
            extra = self.last_obs.get("extra", {})
            tcp_pose = extra.get("tcp_pose", None)

        # Fallback: read directly from the agent TCP pose.
        if tcp_pose is None:
            try:
                tcp = getattr(getattr(self.env.unwrapped.agent, "tcp", None), "pose", None)
                if tcp is not None:
                    p = self._to_numpy(getattr(tcp, "p"))
                    q = self._to_numpy(getattr(tcp, "q"))
                    tcp_pose = np.concatenate([p, q], axis=-1)
            except Exception:
                tcp_pose = None

        if tcp_pose is None:
            return

        tcp_pose = self._to_numpy(tcp_pose).astype(np.float32)
        if tcp_pose.ndim == 1:
            tcp_pose = tcp_pose.reshape(1, 7)

        # Compute camera world pose = tcp_pose * offset.
        tcp_pos = tcp_pose[:, :3]
        tcp_quat_wxyz = tcp_pose[:, 3:7]
        tcp_rot = R.from_quat(tcp_quat_wxyz.astype(np.float64), scalar_first=True)
        offset_pos = self.wrist_camera_tcp_pos_offset.astype(np.float64)
        offset_rot = R.from_quat(self.wrist_camera_tcp_quat_wxyz_offset.astype(np.float64), scalar_first=True)

        cam_pos = tcp_pos.astype(np.float64) + tcp_rot.apply(offset_pos.reshape(1, 3))
        cam_rot = (tcp_rot * offset_rot)
        cam_quat_wxyz = cam_rot.as_quat(scalar_first=True).astype(np.float32)

        # Apply pose (best-effort). Many ManiSkill sensor poses are not batched;
        # in that case, just use the active env pose.
        pose = getattr(wrist_sensor, "pose", None)
        if pose is None and hasattr(wrist_sensor, "camera"):
            pose = getattr(getattr(wrist_sensor, "camera"), "pose", None)

        try:
            pose_p = np.asarray(getattr(pose, "p"), dtype=np.float32) if pose is not None else None
        except Exception:
            pose_p = None

        if pose_p is not None and pose_p.ndim == 2 and pose_p.shape[0] == cam_pos.shape[0]:
            ok = self._set_sensor_pose(wrist_sensor, cam_pos.astype(np.float32), cam_quat_wxyz)
        else:
            idx = int(np.clip(self.active_env_idx, 0, cam_pos.shape[0] - 1))
            ok = self._set_sensor_pose(
                wrist_sensor,
                cam_pos[idx].astype(np.float32),
                cam_quat_wxyz[idx].astype(np.float32),
            )

        if not ok:
            # Silent failure to avoid log spam; user can disable follow-tcp.
            return

    def _to_numpy(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _quat_wxyz_to_xyzw(self, quat_wxyz: np.ndarray) -> np.ndarray:
        quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
        return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)

    def _quat_xyzw_to_wxyz(self, quat_xyzw: np.ndarray) -> np.ndarray:
        quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)

    def _pose_wxyz_to_xyzw(self, pose_wxyz: np.ndarray) -> np.ndarray:
        pose_wxyz = np.asarray(pose_wxyz, dtype=np.float32).reshape(7)
        return np.concatenate([pose_wxyz[:3], self._quat_wxyz_to_xyzw(pose_wxyz[3:7])], axis=0)

    def _clone_state_dict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        def _clone(v):
            if isinstance(v, dict):
                return {k: _clone(x) for k, x in v.items()}
            if isinstance(v, torch.Tensor):
                return v.clone()
            if isinstance(v, np.ndarray):
                return v.copy()
            return v

        return _clone(state_dict)

    def _index_env(self, value, env_idx: int):
        if isinstance(value, dict):
            return {k: self._index_env(v, env_idx) for k, v in value.items()}
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value
            return value[env_idx]
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return value
            return value[env_idx]
        return value

    def _capture_snapshot(self):
        if not hasattr(self.env.unwrapped, "get_state_dict"):
            return
        state = self.env.unwrapped.get_state_dict()
        self._snapshots.append(self._clone_state_dict(state))
        if len(self._snapshots) > self.max_snapshot_steps:
            self._snapshots = self._snapshots[-self.max_snapshot_steps :]

    def _restore_snapshot(self, idx: int):
        if idx < 0 or idx >= len(self._snapshots):
            return False
        if not hasattr(self.env.unwrapped, "set_state_dict"):
            return False
        self.env.unwrapped.set_state_dict(self._snapshots[idx])
        self.last_obs = self.env.unwrapped.get_obs()
        self.last_done = False
        return True

    def _extract_tcp_pose(self, obs: Dict[str, Any], env_idx: Optional[int] = None) -> np.ndarray:
        if env_idx is None:
            env_idx = self.active_env_idx
        if obs is not None and isinstance(obs, dict):
            extra = obs.get("extra", {})
            tcp_pose = extra.get("tcp_pose", None)
            if tcp_pose is not None:
                tcp = self._to_numpy(tcp_pose)[env_idx]
                return tcp.astype(np.float32)

        p = self._to_numpy(self.env.unwrapped.agent.tcp.pose.p)[env_idx]
        q = self._to_numpy(self.env.unwrapped.agent.tcp.pose.q)[env_idx]
        return np.concatenate([p, q], axis=0).astype(np.float32)

    def _extract_joint_pos(self, obs: Dict[str, Any], env_idx: Optional[int] = None) -> np.ndarray:
        if env_idx is None:
            env_idx = self.active_env_idx
        if obs is not None and isinstance(obs, dict):
            agent = obs.get("agent", {})
            qpos = agent.get("qpos", None)
            if qpos is not None:
                return self._to_numpy(qpos)[env_idx][:7].astype(np.float32)
        return self._to_numpy(self.env.unwrapped.agent.robot.get_qpos())[env_idx][:7].astype(
            np.float32
        )

    def _extract_cube_pose(self, env_idx: Optional[int] = None) -> Optional[np.ndarray]:
        if env_idx is None:
            env_idx = self.active_env_idx

        cube = getattr(self.env.unwrapped, "cube", None)
        if cube is None or not hasattr(cube, "pose"):
            return None

        cube_pose = getattr(cube.pose, "raw_pose", None)
        if cube_pose is None:
            return None

        cube_pose = self._to_numpy(cube_pose)
        if cube_pose.ndim == 1:
            return cube_pose.astype(np.float32)
        return cube_pose[env_idx].astype(np.float32)

    def _get_env_success_flag(self, env_idx: Optional[int] = None) -> bool:
        if env_idx is None:
            env_idx = self.active_env_idx

        if self.success_use_env_info and isinstance(self.last_info, dict) and "success" in self.last_info:
            success_info = self._to_numpy(self.last_info["success"])
            if success_info.ndim == 0:
                return bool(success_info)
            return bool(success_info[env_idx])

        return False

    def _pose_matches_with_cube_symmetry(self, current_pose: np.ndarray, target_pose: np.ndarray) -> bool:
        current_pose = np.asarray(current_pose, dtype=np.float32).reshape(7)
        target_pose = np.asarray(target_pose, dtype=np.float32).reshape(7)

        pos_error = np.linalg.norm(current_pose[:3] - target_pose[:3])
        if pos_error > self.success_pos_tolerance:
            return False

        current_rot = R.from_quat(self._quat_wxyz_to_xyzw(current_pose[3:7]))
        target_rot = R.from_quat(self._quat_wxyz_to_xyzw(target_pose[3:7]))
        tolerance_rad = np.deg2rad(self.success_rot_tolerance_deg)

        if not self._cube_symmetry_rotations:
            rel = current_rot.inv() * target_rot
            return np.linalg.norm(rel.as_rotvec()) <= tolerance_rad

        for symmetry in self._cube_symmetry_rotations:
            sym_target = target_rot * symmetry
            rel = current_rot.inv() * sym_target
            if np.linalg.norm(rel.as_rotvec()) <= tolerance_rad:
                return True

        return False

    def is_task_success(self, env_idx: Optional[int] = None) -> bool:
        if env_idx is None:
            env_idx = self.active_env_idx

        if self._get_env_success_flag(env_idx):
            return True

        if not self.success_use_pose_check:
            return False

        cube_pose = self._extract_cube_pose(env_idx)
        if cube_pose is None:
            return False

        if self.success_require_static:
            info = self.last_info if isinstance(self.last_info, dict) else {}
            if "is_robot_static" in info:
                robot_static = self._to_numpy(info["is_robot_static"])
                if robot_static.ndim == 0:
                    if not bool(robot_static):
                        return False
                elif not bool(robot_static[env_idx]):
                    return False

        return self._pose_matches_with_cube_symmetry(cube_pose, self.success_target_pose_wxyz)

    def get_task_success_flags(self) -> np.ndarray:
        flags = np.zeros((self.num_envs,), dtype=np.bool_)
        for env_idx in range(self.num_envs):
            flags[env_idx] = self.is_task_success(env_idx)
        return flags

    def _render_image(self, camera_name: str, env_idx: Optional[int] = None) -> np.ndarray:
        if env_idx is None:
            env_idx = self.active_env_idx
        img = None
        if self.last_obs is not None and isinstance(self.last_obs, dict):
            sensor_data = self.last_obs.get("sensor_data", {})
            if camera_name in sensor_data and "rgb" in sensor_data[camera_name]:
                rgb_tensor = sensor_data[camera_name]["rgb"]
                if isinstance(rgb_tensor, torch.Tensor):
                    rgb = rgb_tensor.detach().cpu().numpy()[env_idx]
                else:
                    rgb = np.asarray(rgb_tensor)[env_idx]
                img = rgb

        if img is None:
            rendered = self.env.render()
            if isinstance(rendered, torch.Tensor):
                rendered = rendered.detach().cpu().numpy()
            if rendered.ndim == 4:
                rendered = rendered[env_idx]
            img = rendered

        self.last_render = np.asarray(img, dtype=np.uint8)
        return self.last_render

    def reset_robot(self, random_init=False, random_init_pose=None):
        self._teleop_steps = 0
        self.keyboard.ctn = False
        self.keyboard.help = False
        self.keyboard.infer = False
        self.keyboard.finish = False
        self.keyboard.discard = False

        self.last_obs, self.last_info = self.env.reset(seed=self.seed)
        self._auto_select_camera_names_once()
        tcp_pose = self._extract_tcp_pose(self.last_obs, self.active_env_idx)
        self.last_tcp_pose = tcp_pose
        self.robot.init_pose = tcp_pose.copy()
        self.last_joint_pos = self._extract_joint_pos(self.last_obs, self.active_env_idx)
        self.last_gripper_width = self.gripper_max_width
        self.last_terminated[:] = False
        self.last_truncated[:] = False
        self._teleop_help_printed = False
        self._stdin_hint_printed = False
        self._stdin_unavailable_warned = False
        self._last_no_input_log_time = 0.0
        if self._heuristic_teleop_node is not None:
            self._heuristic_teleop_node.reset()

        self._snapshots = []
        self._capture_snapshot()
        self._apply_debug_camera_offsets_once()
        self._maybe_update_wrist_camera_pose_from_tcp()

        # random_init_pose is accepted for API compatibility but not enforced in simulator.
        _ = random_init
        _ = random_init_pose

        return self.get_robot_state()

    def get_robot_state(self, env_idx: Optional[int] = None):
        if env_idx is None:
            env_idx = self.active_env_idx

        tcp_pose = self._extract_tcp_pose(self.last_obs, env_idx)
        joint_pos = self._extract_joint_pos(self.last_obs, env_idx)
        self.last_tcp_pose = tcp_pose
        self.last_joint_pos = joint_pos

        side_rgb = self._render_image(self.side_camera_name, env_idx)
        wrist_rgb = self._render_image(self.wrist_camera_name, env_idx)

        policy_side_img = self.policy_side_image_processor(
            torch.from_numpy(side_rgb.copy()).permute(2, 0, 1)
        )
        policy_wrist_img = self.policy_wrist_image_processor(
            torch.from_numpy(wrist_rgb.copy()).permute(2, 0, 1)
        )
        demo_side_img = self.demo_image_processor(
            torch.from_numpy(side_rgb.copy()).permute(2, 0, 1)
        )
        demo_wrist_img = self.demo_image_processor(
            torch.from_numpy(wrist_rgb.copy()).permute(2, 0, 1)
        )
        # debug: show teleop camera images in OpenCV windows
        if self.debug_camera_enable and not self._has_display:
            # save the processed policy images to disk for inspection
            Image.fromarray(policy_side_img.permute(1, 2, 0).numpy()).save("debug_policy_side.png")
            Image.fromarray(policy_wrist_img.permute(1, 2, 0).numpy()).save("debug_policy_wrist.png")
            print('image saved')
        if self.debug_camera_enable and self._has_display:
            cv2.imshow("Debug Side Camera", side_rgb)
            cv2.imshow("Debug Wrist Camera", wrist_rgb)
            cv2.waitKey(1)
            
        return {
            "tcp_pose": tcp_pose,
            "joint_pos": joint_pos,
            "policy_side_img": policy_side_img,
            "policy_wrist_img": policy_wrist_img,
            "demo_side_img": demo_side_img,
            "demo_wrist_img": demo_wrist_img,
            "side_img_raw": side_rgb.copy(),
            "wrist_img_raw": wrist_rgb.copy(),
        }

    def _abs_target_to_delta_action(
        self,
        tcp_action: np.ndarray,
        gripper_action: float,
        curr_pose: Optional[np.ndarray] = None,
    ):
        if curr_pose is None:
            curr_pose = self.last_tcp_pose if self.last_tcp_pose is not None else tcp_action.copy()

        curr_pose = np.asarray(curr_pose, dtype=np.float32).reshape(7)
        tgt_pose = np.asarray(tcp_action, dtype=np.float32).reshape(7)
        pose_delta_xyzw = extract_action_from_poses(
            self._pose_wxyz_to_xyzw(curr_pose),
            self._pose_wxyz_to_xyzw(tgt_pose),
            float(self.last_gripper_width),
            float(gripper_action),
        )
        dp = pose_delta_xyzw[:3].astype(np.float32)
        dr = R.from_quat(pose_delta_xyzw[3:7]).as_rotvec().astype(np.float32)

        if self.gripper_max_width > 1e-6:
            g = float(np.clip((2.0 * (gripper_action / self.gripper_max_width)) - 1.0, -1.0, 1.0))
        else:
            g = float(np.clip(gripper_action, -1.0, 1.0))

        action = np.concatenate([dp.astype(np.float32), dr, np.array([g], dtype=np.float32)])
        return np.clip(action, self.single_action_space.low, self.single_action_space.high)

    def deploy_action(self, tcp_action, gripper_action, env_idx: Optional[int] = None):
        if env_idx is None:
            env_idx = self.active_env_idx

        curr_pose = None
        if self.last_obs is not None:
            curr_pose = self._extract_tcp_pose(self.last_obs, env_idx)
        action = self._abs_target_to_delta_action(np.asarray(tcp_action), float(gripper_action), curr_pose=curr_pose)
        batched_action = np.zeros((self.num_envs, action.shape[0]), dtype=np.float32)
        batched_action[env_idx] = action
        obs, _reward, terminated, truncated, info = self.env.step(batched_action)
        self.last_obs = obs
        self.last_info = info
        self.last_terminated = self._to_numpy(terminated).astype(np.bool_)
        self.last_truncated = self._to_numpy(truncated).astype(np.bool_)
        self.last_done = bool(self.last_terminated[env_idx] or self.last_truncated[env_idx])
        self.last_gripper_width = float(gripper_action)
        self._capture_snapshot()
        self._auto_select_camera_names_once()
        self._maybe_update_wrist_camera_pose_from_tcp()

    def deploy_action_batch(self, tcp_actions: Dict[int, np.ndarray], gripper_actions: Dict[int, float]):
        action_dim = self.env.action_space.shape[-1]
        batched_action = np.zeros((self.num_envs, action_dim), dtype=np.float32)
        for env_idx, tcp_action in tcp_actions.items():
            gripper_action = gripper_actions[env_idx]
            curr_pose = None
            if self.last_obs is not None:
                curr_pose = self._extract_tcp_pose(self.last_obs, env_idx)
            batched_action[env_idx] = self._abs_target_to_delta_action(
                np.asarray(tcp_action),
                float(gripper_action),
                curr_pose=curr_pose,
            )

        obs, _reward, terminated, truncated, info = self.env.step(batched_action)
        self.last_obs = obs
        self.last_info = info
        self.last_terminated = self._to_numpy(terminated).astype(np.bool_)
        self.last_truncated = self._to_numpy(truncated).astype(np.bool_)
        self.last_done = bool(np.any(self.last_terminated | self.last_truncated))
        self._capture_snapshot()
        self._auto_select_camera_names_once()
        self._maybe_update_wrist_camera_pose_from_tcp()

    def save_scene_images(self, output_dir, episode_idx):
        state = self.get_robot_state()
        side_img = state["side_img_raw"]
        wrist_img = state["wrist_img_raw"]
        Image.fromarray(side_img).save(f"{output_dir}/side_{episode_idx}.png")
        Image.fromarray(wrist_img).save(f"{output_dir}/wrist_{episode_idx}.png")
        return side_img, wrist_img

    def align_with_reference(self, ref_side_img, ref_wrist_img, raw=False):
        # Keep API behavior, but non-blocking in simulation.
        _ = ref_side_img
        _ = ref_wrist_img
        _ = raw
        self.keyboard.ctn = False

    def align_scene_with_file(self, output_dir, episode_idx):
        ref_side_img = cv2.imread(f"{output_dir}/side_{episode_idx}.png")
        ref_wrist_img = cv2.imread(f"{output_dir}/wrist_{episode_idx}.png")
        self.align_with_reference(ref_side_img, ref_wrist_img, raw=True)

    def detach_sigma(self):
        curr_pose = self.last_tcp_pose if self.last_tcp_pose is not None else self.robot.init_pose
        detach_pos = np.array(curr_pose[:3], dtype=np.float32)
        detach_rot = R.from_quat(np.array(curr_pose[3:], dtype=np.float32), scalar_first=True)
        return detach_pos, detach_rot

    def set_active_env(self, env_idx: int):
        self.active_env_idx = int(np.clip(env_idx, 0, self.num_envs - 1))

    def get_env_done_flags(self):
        return self.last_terminated.copy(), self.last_truncated.copy()

    def _poll_keyboard_key(self) -> Optional[str]:
        if self.teleop_use_cv2_keys and self._has_display:
            try:
                key_code = cv2.waitKey(0)  # Wait indefinitely for a key press
                if key_code != -1:
                    try:
                        return chr(key_code & 0xFF).lower()
                    except ValueError:
                        pass
            except Exception as exc:
                self._has_display = False
                print(f"[SIM TELEOP] OpenCV key polling disabled due to display error: {exc}")

        if sys.stdin.isatty():
            if not self._stdin_hint_printed:
                print("[SIM TELEOP] Reading keyboard input from terminal. Type key followed by Enter.")
                self._stdin_hint_printed = True
            try:
                timeout = self.teleop_terminal_poll_timeout_s if not self._has_display else 0.0
                if select.select([sys.stdin], [], [], timeout)[0]:
                    text = sys.stdin.readline().strip().lower()
                    if text:
                        return text[0]
            except Exception:
                return None
        else:
            if not self._stdin_unavailable_warned and not self._has_display:
                print("[SIM TELEOP] No interactive stdin available in headless mode; keyboard teleop input is unavailable.")
                self._stdin_unavailable_warned = True

        now = time.time()
        if now - self._last_no_input_log_time >= self.teleop_no_input_log_interval_s:
            print("[SIM TELEOP] No keyboard input detected.")
            self._last_no_input_log_time = now
        return None

    def _print_teleop_help_once(self):
        if not self.teleop_show_help or self._teleop_help_printed:
            return
        input_suffix = " (type key then Enter in terminal)" if (not self._has_display) else ""
        print(
            "[SIM TELEOP] Keyboard controls:\n"
            "  Translation: W/S (+/-X), A/Z (+/-Y), R/V (+/-Z)\n"
            "  Rotation:    U/J (+/-Roll), I/K (+/-Pitch), O/L (+/-Yaw)\n"
            "  Gripper:     N (close), M (open)\n"
            f"  Rotation frame: {'world' if self.teleop_rotation_world_frame else 'tcp-local'}\n"
            f"  Control:     C (return to policy), F (finish), D (discard), Q (quit){input_suffix}",
            flush=True,
        )
        self._teleop_help_printed = True

    def _get_heuristic_teleop_node(self):
        if self._heuristic_teleop_node is None:
            heuristic_cfg = dict(self.teleop_heuristic_cfg)
            heuristic_cfg.setdefault("gripper_max_width", self.gripper_max_width)
            heuristic_cfg.setdefault("success_target_pose_wxyz", self.success_target_pose_wxyz)
            self._heuristic_teleop_node = HeuristicTeleopNode(self.env, config=heuristic_cfg)
        return self._heuristic_teleop_node

    def human_teleop_step(self, last_p, last_r, target_tcp_pose=None, target_gripper_width=None):
        start_time = time.time()
        self._teleop_steps += 1
        self._print_teleop_help_once()

        prev_p = np.asarray(last_p, dtype=np.float32).copy()
        prev_q_wxyz = last_r.as_quat(scalar_first=True).astype(np.float32)
        dp = np.zeros(3, dtype=np.float32)
        d_rotvec = np.zeros(3, dtype=np.float32)
        gripper_action = float(self.last_gripper_width)

        if target_tcp_pose is None and self.teleop_input_mode == "heuristic":
            heuristic_command = self._get_heuristic_teleop_node().next_command(last_p, last_r)
            target_tcp_pose = heuristic_command.target_tcp_pose
            target_gripper_width = heuristic_command.target_gripper_width
            if heuristic_command.should_exit:
                if heuristic_command.exit_reason == "success":
                    self.keyboard.finish = True
                else:
                    self.keyboard.infer = True

        if target_tcp_pose is not None:
            target_pose = np.asarray(target_tcp_pose, dtype=np.float32).reshape(7)
            target_p = target_pose[:3]
            target_r = R.from_quat(target_pose[3:], scalar_first=True)
            if target_gripper_width is not None:
                gripper_action = float(np.clip(target_gripper_width, 0.0, self.gripper_max_width))
        else:
            key = self._poll_keyboard_key()
            if key == "q":
                self.keyboard.quit = True
            elif key == "f":
                self.keyboard.finish = True
            elif key == "d":
                self.keyboard.discard = True
            elif key == "c":
                self.keyboard.infer = True
            elif key == "w":
                dp[0] += self.teleop_pos_step
            elif key == "s":
                dp[0] -= self.teleop_pos_step
            elif key == "a":
                dp[1] += self.teleop_pos_step
            elif key == "z":
                dp[1] -= self.teleop_pos_step
            elif key == "r":
                dp[2] += self.teleop_pos_step
            elif key == "v":
                dp[2] -= self.teleop_pos_step
            elif key == "u":
                d_rotvec[0] += np.deg2rad(self.teleop_rot_step_deg)
            elif key == "j":
                d_rotvec[0] -= np.deg2rad(self.teleop_rot_step_deg)
            elif key == "i":
                d_rotvec[1] += np.deg2rad(self.teleop_rot_step_deg)
            elif key == "k":
                d_rotvec[1] -= np.deg2rad(self.teleop_rot_step_deg)
            elif key == "o":
                d_rotvec[2] += np.deg2rad(self.teleop_rot_step_deg)
            elif key == "l":
                d_rotvec[2] -= np.deg2rad(self.teleop_rot_step_deg)
            elif key == "n":
                gripper_action = float(np.clip(self.last_gripper_width - self.teleop_gripper_step, 0.0, self.gripper_max_width))
            elif key == "m":
                gripper_action = float(np.clip(self.last_gripper_width + self.teleop_gripper_step, 0.0, self.gripper_max_width))

            target_p = np.asarray(last_p, dtype=np.float32) + dp
            delta_r = R.from_rotvec(d_rotvec.astype(np.float64))
            if self.teleop_rotation_world_frame:
                # Apply increments in world axes so keys map to world-frame orientation updates.
                target_r = delta_r * last_r
            else:
                # Optional fallback to TCP-local increment behavior.
                target_r = last_r * delta_r

        control_signal_set = self.keyboard.infer or self.keyboard.finish or self.keyboard.discard or self.keyboard.quit
        has_motion = target_tcp_pose is not None or (np.linalg.norm(dp) > 0) or (np.linalg.norm(d_rotvec) > 0)
        has_gripper_update = abs(gripper_action - float(self.last_gripper_width)) > 1e-8
        curr_p_action = np.zeros(3, dtype=np.float32)
        curr_r_action = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if has_motion or has_gripper_update:
            target_pose = np.concatenate((target_p, target_r.as_quat(scalar_first=True).astype(np.float32)), axis=0)
            target_pose_xyzw = self._pose_wxyz_to_xyzw(target_pose)
            prev_pose_xyzw = np.concatenate((prev_p, self._quat_wxyz_to_xyzw(prev_q_wxyz)), axis=0)
            teleop_delta_xyzw = extract_action_from_poses(
                prev_pose_xyzw,
                target_pose_xyzw,
                float(self.last_gripper_width),
                float(gripper_action),
            )
            curr_p_action = teleop_delta_xyzw[:3].astype(np.float32)
            curr_r_action = self._quat_xyzw_to_wxyz(teleop_delta_xyzw[3:7])
            self.deploy_action(target_pose, gripper_action)
            last_p = target_p
            last_r = target_r
        elif control_signal_set:
            time.sleep(max(1 / self.fps - (time.time() - start_time), 0))
            return None, last_p, last_r

        state_data = self.get_robot_state()
        tcp_pose = state_data["tcp_pose"]
        joint_pos = state_data["joint_pos"]

        # Use measured simulator pose as teleop feedback state to avoid accumulating
        # unreachable command targets, which often manifests as jitter.
        if self.teleop_feedback_from_measured_pose:
            last_p = np.asarray(tcp_pose[:3], dtype=np.float32)
            last_r = R.from_quat(np.asarray(tcp_pose[3:], dtype=np.float32), scalar_first=True)

        processed_data = {
            "policy_wrist_img": state_data["policy_wrist_img"],
            "policy_side_img": state_data["policy_side_img"],
            "demo_wrist_img": state_data["demo_wrist_img"],
            "demo_side_img": state_data["demo_side_img"],
            "wrist_img_raw": state_data["wrist_img_raw"],
            "side_img_raw": state_data["side_img_raw"],
            "tcp_pose": tcp_pose,
            "joint_pos": joint_pos,
            "action": np.concatenate((curr_p_action, curr_r_action, [gripper_action])),
            "action_mode": INTV,
        }

        time.sleep(max(1 / self.fps - (time.time() - start_time), 0))
        return processed_data, np.asarray(last_p, dtype=np.float32), last_r

    def rewind_robot(self, curr_pos, curr_rot, inverse_action):
        _ = inverse_action
        if len(self._snapshots) <= 1:
            return curr_pos, curr_rot

        self._snapshots.pop()
        restored = self._restore_snapshot(len(self._snapshots) - 1)
        if not restored:
            return curr_pos, curr_rot

        pose = self._extract_tcp_pose(self.last_obs, self.active_env_idx)
        new_pos = pose[:3]
        new_rot = R.from_quat(pose[3:], scalar_first=True)
        return new_pos, new_rot
