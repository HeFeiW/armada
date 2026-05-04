"""Heuristic teleoperation helper for ManiSkill rollout.

This module provides a small in-process "human" controller that converts the
existing heuristic pick policy into direct TCP pose targets. It is designed to
plug into the ManiSkill teleop step without changing the rollout/HIL decision
flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from maniskill_armada.heuristic_policy import HeuristicPickPolicy


@dataclass
class HeuristicTeleopCommand:
    target_tcp_pose: np.ndarray
    target_gripper_width: float
    should_exit: bool = False
    exit_reason: str = ""


class HeuristicTeleopNode:
    """Generate teleop TCP targets from the ManiSkill heuristic pick policy."""

    def __init__(self, env, config: Optional[Dict] = None):
        self.env = env
        self.config = config or {}
        self.success_target_pose_wxyz = np.asarray(
            self.config.get(
                "success_target_pose_wxyz",
                [0.02510096, -0.00136661, 0.2814997, 0.7723232, 0.61468965, 0.15892147, 0.02043187],
            ),
            dtype=np.float32,
        ).reshape(7)
        policy_cfg = dict(self.config)
        policy_cfg.setdefault("fixed_goal_pos", self.success_target_pose_wxyz[:3].tolist())
        policy_cfg.setdefault("fixed_goal_quat_wxyz", self.success_target_pose_wxyz[3:].tolist())
        policy_cfg.setdefault("use_cube_pose_for_lift_rotation", False)
        policy_cfg.setdefault("lock_lift_rotation", True)
        self.policy = HeuristicPickPolicy(env, config=policy_cfg)
        self.max_steps = int(self.config.get("max_steps", 64))
        self.exit_on_success = bool(self.config.get("exit_on_success", True))
        self.gripper_max_width = float(self.config.get("gripper_max_width", 0.09))
        self.debug = bool(self.config.get("debug", False))
        self.step_count = 0
        self._done = False

    def reset(self):
        self.policy.reset()
        self.step_count = 0
        self._done = False

    def next_command(self, last_p: np.ndarray, last_r: R, obs: Optional[Dict] = None) -> HeuristicTeleopCommand:
        """Return the next heuristic TCP target pose for teleoperation."""
        raw_extra = {}
        last_obs = getattr(self.env, "last_obs", None)
        if isinstance(last_obs, dict):
            raw_extra = dict(last_obs.get("extra", {}))

        if self._done:
            last_pose = np.concatenate(
                [np.asarray(last_p, dtype=np.float32).reshape(3), last_r.as_quat(scalar_first=True).astype(np.float32)],
                axis=0,
            )
            return HeuristicTeleopCommand(
                target_tcp_pose=last_pose,
                target_gripper_width=self.gripper_max_width,
                should_exit=True,
                exit_reason="done",
            )

        self.step_count += 1
        obs = obs or {}
        extra_obs = dict(obs.get("extra", {}))
        extra_obs["tcp_pose"] = np.concatenate(
            [np.asarray(last_p, dtype=np.float32).reshape(3), last_r.as_quat(scalar_first=True).astype(np.float32)],
            axis=0,
        )
        heuristic_obs = {"extra": extra_obs}

        target_pos, target_rot, target_gripper = self.policy.get_target_pose(heuristic_obs)
        target_pose = np.concatenate(
            [target_pos.astype(np.float32), R.from_matrix(target_rot).as_quat(scalar_first=True).astype(np.float32)],
            axis=0,
        )
        target_gripper_width = float((1.0 - float(target_gripper)) * self.gripper_max_width)

        should_exit = False
        exit_reason = ""
        if self.exit_on_success and hasattr(self.env, "is_task_success") and self.env.is_task_success():
            should_exit = True
            exit_reason = "success"
        elif self.max_steps > 0 and self.step_count >= self.max_steps:
            should_exit = True
            exit_reason = "max_steps"

        if self.debug:
            goal_pose_raw = raw_extra.get("goal_pose", None)
            goal_pos_raw = raw_extra.get("goal_pos", None)
            target_pos_err = float(np.linalg.norm(target_pose[:3] - self.success_target_pose_wxyz[:3]))
            target_rot_err = float(
                np.linalg.norm(
                    (
                        R.from_quat(self.success_target_pose_wxyz[3:], scalar_first=True).inv()
                        * R.from_quat(target_pose[3:], scalar_first=True)
                    ).as_rotvec()
                )
            )
            print(
                "[HEURISTIC TELEOP] "
                f"phase={self.policy.phase} step={self.step_count} "
                f"extra.goal_pose={goal_pose_raw} extra.goal_pos={goal_pos_raw} "
                f"target_tcp_pose={np.round(target_pose, 6).tolist()} "
                f"success_target_pose_wxyz={np.round(self.success_target_pose_wxyz, 6).tolist()} "
                f"pos_err_to_success={target_pos_err:.6f} rot_err_to_success_rad={target_rot_err:.6f}"
            )

        if should_exit:
            self._done = True

        return HeuristicTeleopCommand(
            target_tcp_pose=target_pose,
            target_gripper_width=target_gripper_width,
            should_exit=should_exit,
            exit_reason=exit_reason,
        )
