"""
Heuristic Pick-and-Place Policy using Inverse Kinematics

Implements a curriculum learning approach:
1. Stage 1: Random reaching (5-10 episodes)
2. Stage 2: Guided picking (15-25 episodes)
3. Stage 3: Full pick & place (10-15 episodes)
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
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
        self.max_tcp_delta = 0.05  # Max delta per step (meters)
        self.max_rot_delta = 0.2   # Max rotation delta (radians)
        self.gripper_threshold = 0.3  # Height for grasping
        self.lift_height = 0.1     # Height to lift object

        # State tracking
        self.prev_tcp_pose = None
        self.target_reached_count = 0
        self.grasp_attempted = False

    def get_action(self, obs: Dict, stage: str) -> np.ndarray:
        """
        Compute action for current observation.

        Args:
            obs: ManiSkill observation dict
            stage: 'stage1' (reaching), 'stage2' (guided pick), 'stage3' (full place)

        Returns:
            action: (8,) array [dx, dy, dz, dqx, dqy, dqz, dqw, gripper]
        """
        # Get target pose based on stage
        target_tcp_pos, target_tcp_rot, target_gripper = self._get_target(obs, stage)

        # Compute action
        action_delta = self._compute_delta_action(obs, target_tcp_pos, target_tcp_rot)

        # Append gripper command
        action = np.concatenate([action_delta, [target_gripper]])

        return action.astype(np.float32)

    def _get_target(self, obs: Dict, stage: str) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute target TCP pose and gripper command based on stage.

        Returns:
            (target_pos, target_rot, gripper_cmd)
        """
        # Extract current state
        tcp_pos = obs['agent']['tcp_position']  # [3]
        tcp_rot = obs['agent']['tcp_rotation']   # [3, 3]
        cube_pos = obs.get('object', {}).get('position', np.array([0.5, 0.5, 0.1]))

        tcp_rot_quat = R.from_matrix(tcp_rot).as_quat(scalar_first=True)  # [w, x, y, z]

        if stage == 'stage1':
            # Random reaching
            target_pos = self._sample_random_valid_target(obs)
            target_rot = self._get_neutral_rotation()
            target_gripper = 0.0  # Open gripper

        elif stage == 'stage2':
            # Guided pick: approach cube, grasp, lift
            if tcp_pos[2] > cube_pos[2] + self.gripper_threshold:
                # Above cube - approach
                target_pos = cube_pos + np.array([0, 0, 0.05])
                target_rot = self._get_neutral_rotation()
                target_gripper = 0.0
            elif self.target_reached_count < 5:
                # Reached above - move down slowly
                target_pos = cube_pos.copy()
                target_pos[2] = cube_pos[2]
                target_rot = self._get_neutral_rotation()
                target_gripper = 0.0
                self.target_reached_count += 1
            else:
                # Close gripper and lift
                target_pos = cube_pos + np.array([0, 0, self.lift_height])
                target_rot = self._get_neutral_rotation()
                target_gripper = 1.0
                self.grasp_attempted = True

        else:  # stage3
            # Full place: pick, move to target location, place
            place_target = self._sample_place_location(obs)

            if not self.grasp_attempted:
                # First, grasp
                target_pos = cube_pos + np.array([0, 0, self.lift_height])
                target_rot = self._get_neutral_rotation()
                target_gripper = 1.0
            else:
                # Move to place location and release
                target_pos = place_target + np.array([0, 0, 0])
                target_rot = self._get_neutral_rotation()
                target_gripper = 0.0  # Release

        return target_pos, target_rot, target_gripper

    def _compute_delta_action(self, obs: Dict, target_pos: np.ndarray,
                              target_rot: np.ndarray) -> np.ndarray:
        """
        Compute delta action using IK.

        Returns:
            (7,) delta action [dx, dy, dz, dqx, dqy, dqz, dqw]
        """
        tcp_pos = obs['agent']['tcp_position']
        tcp_rot = obs['agent']['tcp_rotation']
        qpos = obs['agent']['qpos'][:self.arm_dof]

        # Compute position delta (clip to max)
        pos_delta = target_pos - tcp_pos
        pos_delta_norm = np.linalg.norm(pos_delta)
        if pos_delta_norm > self.max_tcp_delta:
            pos_delta = pos_delta / pos_delta_norm * self.max_tcp_delta

        # Compute rotation delta
        current_rot_quat = R.from_matrix(tcp_rot).as_quat(scalar_first=True)
        target_rot_quat = R.from_matrix(target_rot).as_quat(scalar_first=True)

        # Compute relative rotation
        rel_rot = R.from_matrix(
            R.from_quat(target_rot_quat[1:] + [target_rot_quat[0]]).as_matrix() @
            R.from_matrix(tcp_rot).as_matrix().T
        )
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
        x_range = [0.2, 0.8]
        y_range = [-0.3, 0.3]
        z_range = [0.15, 0.6]

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
        self.target_reached_count = 0
        self.grasp_attempted = False
        self.prev_tcp_pose = None
