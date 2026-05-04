"""
Heuristic Pick-and-Place Policy using Inverse Kinematics
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)


class HeuristicPickPolicy:
    """
    IK-based policy for pick-and-place task using ManiSkill environment.

    Generates delta actions computed from IK solutions.
    """

    def __init__(self, env, config: Optional[Dict] = None):
        """
        Args:
            env: ManiSkill environment instance (for FK/IK)
            config: Optional configuration dict
        """
        self.env = env
        self.config = config or {}

        # Task parameters
        self.arm_dof = 7
        self.max_tcp_delta = float(self.config.get('max_tcp_delta', 0.2))
        self.max_rot_delta = float(self.config.get('max_rot_delta', 0.6))
        self.approach_tolerance = float(self.config.get('approach_tolerance', 0.015))
        self.grasp_height_offset = float(self.config.get('grasp_height_offset', 0.00))
        self.lock_lift_rotation = bool(self.config.get('lock_lift_rotation', False))
        self.use_cube_pose_for_lift_rotation = bool(self.config.get('use_cube_pose_for_lift_rotation', False))
        self.fixed_goal_pos = self.config.get('fixed_goal_pos', None)
        if self.fixed_goal_pos is not None:
            self.fixed_goal_pos = np.asarray(self.fixed_goal_pos, dtype=np.float32).reshape(-1)
            if self.fixed_goal_pos.shape[0] != 3:
                raise ValueError(f"fixed_goal_pos must have shape (3,), got {self.fixed_goal_pos.shape}")
        self.fixed_goal_quat_wxyz = self.config.get('fixed_goal_quat_wxyz', None)
        if self.fixed_goal_quat_wxyz is not None:
            self.fixed_goal_quat_wxyz = np.asarray(self.fixed_goal_quat_wxyz, dtype=np.float32).reshape(-1)
            if self.fixed_goal_quat_wxyz.shape[0] != 4:
                raise ValueError(
                    f"fixed_goal_quat_wxyz must have shape (4,), got {self.fixed_goal_quat_wxyz.shape}"
                )

        # State tracking
        self.prev_tcp_pose = None
        self.phase = 'approach'
        self.grasp_issued = False
        self.debug = bool(self.config.get('debug', False))
        self._grasp_ref_rp = None
        self._lift_rot_ref = None
        self._cube_to_tcp_rot_at_grasp = None

    def get_action(self, obs: Dict, obj_pose: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute action for current observation.

        Args:
            obs: ManiSkill observation dict
            
        Returns:
            action: (8,) array [dx, dy, dz, dqx, dqy, dqz, dqw, gripper]
        """
        # Get target pose
        target_tcp_pos, target_tcp_rot, target_gripper = self.get_target_pose(obs, obj_pose=obj_pose)

        # Compute action
        action_delta = self._compute_delta_action(obs, target_tcp_pos, target_tcp_rot)

        # Append gripper command
        action = np.concatenate([action_delta, [target_gripper]])
        if self.debug:
            print(
                f"Target TCP Pos: {target_tcp_pos}, "
                f"Target Gripper: {target_gripper}, action: {action}"
            )
        return action.astype(np.float32)

    def get_target_pose(self, obs: Dict, obj_pose: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return the current heuristic TCP target pose and gripper command."""
        return self._get_target(obs, obj_pose=obj_pose)

    def to_maniskill_action(self, action: np.ndarray) -> np.ndarray:
        """Convert ARMADA 8D action to ManiSkill 7D pd_ee_delta_pose action.

        ARMADA format: [dx, dy, dz, dqw, dqx, dqy, dqz, gripper_target]
        ManiSkill format: [dx, dy, dz, drotvec_x, drotvec_y, drotvec_z, gripper_cmd]
        where gripper_cmd uses +1=open, -1=close.
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        assert action.shape[0] == 8, f"Expected 8D ARMADA action, got {action.shape}"

        dp = action[:3]
        dquat = action[3:7]
        dquat_norm = np.linalg.norm(dquat)
        if dquat_norm < 1e-8:
            drotvec = np.zeros(3, dtype=np.float32)
        else:
            dquat = dquat / dquat_norm
            drotvec = R.from_quat(dquat, scalar_first=True).as_rotvec().astype(np.float32)
            rot_norm = np.linalg.norm(drotvec)
            if rot_norm > self.max_rot_delta > 1e-8:
                drotvec = drotvec / rot_norm * self.max_rot_delta

        # ARMADA target convention: 0=open, 1=close.
        gripper_target = float(np.clip(action[7], 0.0, 1.0))
        gripper_cmd = 1.0 - 2.0 * gripper_target

        ms_action = np.concatenate([dp, drotvec, [gripper_cmd]]).astype(np.float32)
        return np.clip(ms_action, -1.0, 1.0)

    def _extract_obj_pose(self, obs: Dict, obj_pose: Optional[np.ndarray] = None) -> np.ndarray:
        """Resolve object pose from explicit input, observation, or environment fallback."""
        if obj_pose is not None:
            return np.asarray(obj_pose, dtype=np.float32).reshape(-1)

        extra_obs = obs.get('extra', {})
        if 'obj_pose' in extra_obs:
            return np.asarray(extra_obs['obj_pose'], dtype=np.float32).reshape(-1)

        # RGBD mode often omits obj_pose from `extra`; read from env state directly.
        if hasattr(self.env.unwrapped, 'cube') and hasattr(self.env.unwrapped.cube, 'pose'):
            cube_pose = self.env.unwrapped.cube.pose
            p = np.asarray(cube_pose.p).reshape(-1)
            q = np.asarray(cube_pose.q).reshape(-1)
            return np.concatenate([p, q], axis=0).astype(np.float32)

        return np.array([0.55, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def _get_target(self, obs: Dict, obj_pose: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute target TCP pose and gripper command.

        Returns:
            (target_pos, target_rot, gripper_cmd)
        """
        extra_obs = obs.get('extra', {})
        tcp_pose = np.asarray(extra_obs.get('tcp_pose', np.zeros(7, dtype=np.float32)), dtype=np.float32).reshape(-1)
        tcp_pos = tcp_pose[:3]
        cube_pose = self._extract_obj_pose(obs, obj_pose=obj_pose)
        cube_pos = cube_pose[:3]
        cube_quat = cube_pose[3:7]
        if self.fixed_goal_pos is not None:
            goal_pos = self.fixed_goal_pos
        else:
            goal_pos = np.asarray(
                extra_obs.get('goal_pos', cube_pos + np.array([0.0, 0.0, 0.1], dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)


        grasp_pose = cube_pos.copy()
        grasp_pose[2] = cube_pos[2] + self.grasp_height_offset
        lift_target = goal_pos.copy()

        tcp_quat = tcp_pose[3:7]
        grasp_rot = self._get_grasp_aligned_rotation(cube_quat, tcp_quat=tcp_quat)

        if self.phase == 'approach':
            target_pos = grasp_pose
            target_rot = grasp_rot
            target_gripper = 0.0
            if np.linalg.norm(tcp_pos - grasp_pose) <= self.approach_tolerance:
                if np.linalg.norm(cube_quat) > 1e-8 and np.linalg.norm(tcp_quat) > 1e-8:
                    cube_rot_now = R.from_quat(cube_quat / np.linalg.norm(cube_quat), scalar_first=True)
                    tcp_rot_now = R.from_quat(tcp_quat / np.linalg.norm(tcp_quat), scalar_first=True)
                    self._cube_to_tcp_rot_at_grasp = cube_rot_now.inv() * tcp_rot_now
                self.phase = 'grasp'
        elif self.phase == 'grasp':
            # One-step grasp command.
            target_pos = grasp_pose
            target_rot = grasp_rot
            target_gripper = 1.0
            self.grasp_issued = True
            if self._cube_to_tcp_rot_at_grasp is None and np.linalg.norm(cube_quat) > 1e-8 and np.linalg.norm(tcp_quat) > 1e-8:
                cube_rot_now = R.from_quat(cube_quat / np.linalg.norm(cube_quat), scalar_first=True)
                tcp_rot_now = R.from_quat(tcp_quat / np.linalg.norm(tcp_quat), scalar_first=True)
                self._cube_to_tcp_rot_at_grasp = cube_rot_now.inv() * tcp_rot_now
            self.phase = 'lift'
        else:  # self.phase == 'lift'
            target_pos = lift_target
            if (
                self.use_cube_pose_for_lift_rotation
                and self.fixed_goal_quat_wxyz is not None
                and self._cube_to_tcp_rot_at_grasp is not None
            ):
                goal_cube_rot = R.from_quat(
                    self.fixed_goal_quat_wxyz / np.linalg.norm(self.fixed_goal_quat_wxyz),
                    scalar_first=True,
                )
                target_rot = (goal_cube_rot * self._cube_to_tcp_rot_at_grasp).as_matrix().astype(np.float32)
            elif self.lock_lift_rotation:
                if self._lift_rot_ref is None:
                    self._lift_rot_ref = grasp_rot.copy()
                target_rot = self._lift_rot_ref
            else:
                target_rot = grasp_rot
            target_gripper = 1.0

        if self.debug:
            print(
                f"phase={self.phase} tcp={np.round(tcp_pos,4)} "
                f"grasp={np.round(grasp_pose,4)} goal={np.round(lift_target,4)}"
            )

        return target_pos, target_rot, target_gripper

    def _get_grasp_aligned_rotation(self, cube_quat: np.ndarray, tcp_quat: Optional[np.ndarray] = None) -> np.ndarray:
        """Build grasp orientation by keeping initial TCP roll/pitch and aligning yaw to cube."""
        if tcp_quat is not None and self._grasp_ref_rp is None:
            tcp_q = np.asarray(tcp_quat, dtype=np.float32).reshape(-1)
            if np.linalg.norm(tcp_q) > 1e-8:
                tcp_rot_ref = R.from_quat(tcp_q / np.linalg.norm(tcp_q), scalar_first=True)
                rp = tcp_rot_ref.as_euler('xyz', degrees=False)[:2]
                self._grasp_ref_rp = rp.astype(np.float32)

        # Extract cube yaw from world-frame orientation and align gripper yaw with it.
        cube_rot = R.from_quat(np.asarray(cube_quat, dtype=np.float32), scalar_first=True)
        _, _, cube_yaw = cube_rot.as_euler('xyz', degrees=False)

        # For a cube top grasp, yaw has pi/2 symmetry. Choose shortest turn among these modes.
        if tcp_quat is not None:
            tcp_rot = R.from_quat(np.asarray(tcp_quat, dtype=np.float32), scalar_first=True)
            _, _, tcp_yaw = tcp_rot.as_euler('xyz', degrees=False)

            def _wrap(a: float) -> float:
                return (a + np.pi) % (2 * np.pi) - np.pi

            candidates = [cube_yaw + k * (np.pi / 2.0) for k in range(4)]
            errors = [abs(_wrap(c - tcp_yaw)) for c in candidates]
            chosen_yaw = candidates[int(np.argmin(errors))]
        else:
            chosen_yaw = cube_yaw

        if self._grasp_ref_rp is not None:
            roll_ref, pitch_ref = float(self._grasp_ref_rp[0]), float(self._grasp_ref_rp[1])
            return R.from_euler('xyz', [roll_ref, pitch_ref, chosen_yaw], degrees=False).as_matrix().astype(np.float32)

        # Fallback when no valid TCP reference is available.
        yaw_rot = R.from_euler('z', chosen_yaw).as_matrix()
        neutral = self._get_neutral_rotation()

        return yaw_rot @ neutral

    def _compute_delta_action(self, obs: Dict, target_pos: np.ndarray,
                              target_rot: np.ndarray) -> np.ndarray:
        """
        Compute delta action using IK.

        Returns:
            (7,) delta action [dx, dy, dz, dqx, dqy, dqz, dqw]
        """
        tcp_pose = np.asarray(obs.get('extra', {}).get('tcp_pose', np.zeros(7, dtype=np.float32)), dtype=np.float32).reshape(-1)
        tcp_pos = tcp_pose[:3]
        tcp_rot_quat = tcp_pose[3:7]

        # Compute position delta (clip to max)
        pos_delta = target_pos - tcp_pos
        pos_delta_norm = np.linalg.norm(pos_delta)
        if pos_delta_norm > self.max_tcp_delta:
            pos_delta = pos_delta / pos_delta_norm * self.max_tcp_delta

        current_rot = R.from_quat(tcp_rot_quat, scalar_first=True)
        target_rot_quat = R.from_matrix(target_rot).as_quat(scalar_first=True)
        rel_rot = current_rot.inv() * R.from_quat(target_rot_quat, scalar_first=True)
        rot_delta_quat = rel_rot.as_quat(scalar_first=True)

        # Combine into action
        action = np.concatenate([pos_delta, rot_delta_quat])  # [7]

        return action

    def _get_neutral_rotation(self) -> np.ndarray:
        """Get neutral (downward-facing) rotation matrix."""
        # Gripper pointing down (standard picking orientation)
        return np.array([
            [0,  0,  1],
            [1,  0,  0],
            [0,  1,  0]
        ], dtype=np.float32)

    def _sample_random_valid_target(self, obs: Dict) -> np.ndarray:
        """Sample a random valid TCP target position."""
        # Safe workspace bounds
        x_range = [0.25, 0.75]
        y_range = [-0.2, 0.2]
        z_range = [0.12, 0.35]

        target = np.array([
            np.random.uniform(*x_range),
            np.random.uniform(*y_range),
            np.random.uniform(*z_range)
        ])

        return target

    def _sample_place_location(self, obs: Dict) -> np.ndarray:
        """Sample a valid place location (random in workspace)."""
        x_range = [0.2, 0.8]
        y_range = [-0.3, 0.3]
        z = 0.05  # Place on table

        target = np.array([
            np.random.uniform(*x_range),
            np.random.uniform(*y_range),
            z
        ])

        return target

    def reset(self):
        """Reset policy state between episodes."""
        self.phase = 'approach'
        self.grasp_issued = False
        self.prev_tcp_pose = None
        self._grasp_ref_rp = None
        self._lift_rot_ref = None
        self._cube_to_tcp_rot_at_grasp = None
