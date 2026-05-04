"""
Data format conversion utilities for ManiSkill to ARMADA compatibility.

Handles:
- Observation conversion (ManiSkill obs dict -> ARMADA format)
- Camera rendering (multiple viewpoints)
- Rotation representation conversion
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict, Tuple, Optional, Any
import cv2


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _list_camera_names_from_obs(obs: Optional[Dict]) -> list:
    if not isinstance(obs, dict):
        return []
    sensor_data = obs.get('sensor_data', {})
    if not isinstance(sensor_data, dict):
        return []
    names = []
    for name, payload in sensor_data.items():
        if isinstance(payload, dict) and ('rgb' in payload):
            names.append(str(name))
    return names


def _pick_camera_name(names: list, keywords: list, exclude: Optional[str] = None) -> Optional[str]:
    if not names:
        return None
    exclude = str(exclude) if exclude is not None else None
    lowered = [(n, n.lower()) for n in names]

    # keyword match first
    for kw in keywords:
        kw = kw.lower()
        for n, ln in lowered:
            if exclude is not None and n == exclude:
                continue
            if kw in ln:
                return n

    # fallback: any other
    for n, _ln in lowered:
        if exclude is not None and n == exclude:
            continue
        return n
    return None


def auto_select_camera_names(obs: Optional[Dict], default_side: str = 'base_camera', default_wrist: str = 'base_camera') -> Tuple[str, str, list]:
    """Choose distinct side/wrist camera names based on sensor_data keys."""
    names = _list_camera_names_from_obs(obs)
    if not names:
        return default_side, default_wrist, []

    side = default_side if default_side in names else _pick_camera_name(
        names,
        keywords=['base', 'side', 'front', 'external', 'main', 'viewer'],
        exclude=None,
    )
    if side is None:
        side = names[0]

    wrist = default_wrist if default_wrist in names else None
    if wrist is None or wrist == side:
        wrist = _pick_camera_name(
            names,
            keywords=['wrist', 'hand', 'ee', 'tcp', 'gripper', 'end', 'effector'],
            exclude=side,
        )
    if wrist is None:
        wrist = side
    return side, wrist, names


def set_env_goal_pos(env: Any, goal_pos_xyz: np.ndarray) -> bool:
    """Best-effort: force ManiSkill env goal position while leaving object spawn random.

    Tries to update `env.unwrapped.goal_site` pose and common attributes like `goal_pos` / `_goal_pos`.
    Returns True if any update succeeded.
    """
    if goal_pos_xyz is None:
        return False
    goal_pos_xyz = np.asarray(goal_pos_xyz, dtype=np.float32).reshape(3)

    unwrapped = getattr(env, 'unwrapped', env)
    updated = False

    # 1) Update goal_site pose if present.
    goal_site = getattr(unwrapped, 'goal_site', None)
    if goal_site is not None:
        # Try sapien.Pose (common in ManiSkill)
        pose_obj = None
        try:
            from sapien import Pose  # type: ignore

            pose_obj = Pose(p=goal_pos_xyz.astype(float).tolist(), q=[1.0, 0.0, 0.0, 0.0])
        except Exception:
            pose_obj = None

        if pose_obj is not None:
            try:
                if hasattr(goal_site, 'set_pose'):
                    goal_site.set_pose(pose_obj)
                    updated = True
                elif hasattr(goal_site, 'pose'):
                    goal_site.pose = pose_obj
                    updated = True
            except Exception:
                pass

    # 2) Update common stored goal position attributes if present.
    for attr in ['goal_pos', '_goal_pos']:
        if hasattr(unwrapped, attr):
            try:
                current = getattr(unwrapped, attr)
            except Exception:
                current = None

            try:
                # Preserve torch tensor type/device if used internally.
                if current is not None and current.__class__.__name__ == 'Tensor':
                    try:
                        import torch  # type: ignore

                        new_val = torch.as_tensor(goal_pos_xyz, dtype=current.dtype, device=current.device)
                        if hasattr(current, 'ndim') and getattr(current, 'ndim', 1) == 2:
                            new_val = new_val.reshape(1, 3)
                        setattr(unwrapped, attr, new_val)
                        updated = True
                        continue
                    except Exception:
                        pass

                # Default numpy path.
                setattr(unwrapped, attr, goal_pos_xyz.copy())
                updated = True
            except Exception:
                pass

    return updated


def maniskill_obs_to_armada_format(obs: Dict, env: Any) -> Dict:
    """
    Convert ManiSkill observation dict to ARMADA format.

    Args:
        obs: ManiSkill observation dict
        env: ManiSkill environment (for ground truth extraction)

    Returns:
        dict with keys: tcp_pose, joint_pos, gripper_state
    """
    extra_obs = obs.get('extra', {})
    agent_obs = obs.get('agent', {})

    tcp_pose = extra_obs.get('tcp_pose', None)
    if tcp_pose is None:
        tcp_pose = getattr(getattr(env.unwrapped.agent, 'tcp', None), 'pose', None)
        if tcp_pose is not None:
            tcp_pose = np.concatenate([
                np.asarray(tcp_pose.p)[0],
                np.asarray(tcp_pose.q)[0],
            ])
    if tcp_pose is None:
        raise ValueError('Cannot extract tcp_pose from ManiSkill observation')
    tcp_pose = np.asarray(tcp_pose, dtype=np.float32).reshape(-1)

    qpos = agent_obs.get('qpos', None)
    if qpos is not None:
        joint_pos = np.asarray(qpos, dtype=np.float32).reshape(-1)[:7]
    else:
        joint_pos = np.asarray(env.unwrapped.agent.robot.get_qpos(), dtype=np.float32).reshape(-1)[:7]

    gripper_state = extra_obs.get('is_grasped', np.array([0.0]))
    gripper_value = np.asarray(gripper_state, dtype=np.float32).reshape(-1)[0]

    return {
        'tcp_pose': tcp_pose,           # (7,) float32
        'joint_pos': joint_pos,         # (7,) float32
        'gripper_state': gripper_value  # float32
    }


def render_cameras(obs_or_env: Any, env: Any = None, camera_names: Optional[list] = None,
                   resolution: Tuple[int, int] = (640, 480)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render RGB images from multiple camera viewpoints.

    Args:
        env: ManiSkill environment
        camera_names: List of camera names (default: ['viewer', 'wrist_camera'])
        resolution: (width, height) resolution, will be adjusted to (height, width) for rendering

    Returns:
        (side_img, wrist_img): Both (height, width, 3) uint8 RGB
    """
    # If camera_names is not provided, prefer choosing distinct cameras from sensor_data.
    if camera_names is None:
        obs_tmp = obs_or_env if isinstance(obs_or_env, dict) else None
        side, wrist, _names = auto_select_camera_names(obs_tmp, default_side='base_camera', default_wrist='base_camera')
        camera_names = [side, wrist]
        # One-time debug print to help verify camera wiring.
        if not hasattr(render_cameras, '_printed_camera_selection'):
            setattr(render_cameras, '_printed_camera_selection', True)
            if _names:
                print(f"[collect_data] Available cameras: {_names}; selected side='{side}', wrist='{wrist}'")
            else:
                print(f"[collect_data] No sensor_data cameras found; fallback side='{side}', wrist='{wrist}'")

    obs = obs_or_env if isinstance(obs_or_env, dict) else None
    if env is None and obs is None:
        env = obs_or_env

    # Try extracting sensor data directly from ManiSkill observations first.
    try:
        side_img = None
        wrist_img = None

        if obs is not None:
            sensor_data = obs.get('sensor_data', {})
            if camera_names[1] in sensor_data and 'rgb' in sensor_data[camera_names[1]]:
                wrist_tensor = sensor_data[camera_names[1]]['rgb']
                if hasattr(wrist_tensor, 'detach'):
                    wrist_img = wrist_tensor.detach().cpu().numpy()[0]
                else:
                    wrist_img = np.asarray(wrist_tensor)[0]
            if camera_names[0] in sensor_data and 'rgb' in sensor_data[camera_names[0]]:
                side_tensor = sensor_data[camera_names[0]]['rgb']
                if hasattr(side_tensor, 'detach'):
                    side_img = side_tensor.detach().cpu().numpy()[0]
                else:
                    side_img = np.asarray(side_tensor)[0]

        if env is not None:
            try:
                rendered = env.render()
                if hasattr(rendered, 'detach'):
                    rendered = rendered.detach().cpu().numpy()
                else:
                    rendered = np.asarray(rendered)
                if rendered.ndim == 4:
                    rendered = rendered[0]
                if side_img is None:
                    side_img = rendered
            except Exception:
                pass

        if side_img is None:
            side_img = np.random.randint(0, 256, (*resolution[::-1], 3), dtype=np.uint8)
        if wrist_img is None:
            wrist_img = side_img.copy()

    except Exception as e:
        print(f"Warning: Camera rendering failed ({e}), using random images")
        side_img = np.random.randint(0, 256, (*resolution[::-1], 3), dtype=np.uint8)
        wrist_img = side_img.copy()

    # Ensure correct shape and type
    side_img = np.asarray(side_img, dtype=np.uint8)
    wrist_img = np.asarray(wrist_img, dtype=np.uint8)

    # Ensure (H, W, 3) format
    if side_img.ndim == 2:
        side_img = np.stack([side_img] * 3, axis=-1)
    if wrist_img.ndim == 2:
        wrist_img = np.stack([wrist_img] * 3, axis=-1)

    return side_img, wrist_img


def resize_images(img: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Resize image to target size.

    Args:
        img: Input image (H, W, 3) uint8
        target_size: (height, width)

    Returns:
        Resized image (target_size[0], target_size[1], 3) uint8
    """
    if img.shape[:2] == target_size:
        return img

    resized = cv2.resize(img, (target_size[1], target_size[0]),
                        interpolation=cv2.INTER_LINEAR)
    return resized


def normalize_image(img: np.ndarray, to_range: str = 'float') -> np.ndarray:
    """
    Normalize image to specified range.

    Args:
        img: Input image (H, W, 3) uint8 [0, 255]
        to_range: 'float' -> [0.0, 1.0], 'norm' -> [-1.0, 1.0]

    Returns:
        Normalized image (float32)
    """
    img = img.astype(np.float32)

    if to_range == 'float':
        return img / 255.0
    elif to_range == 'norm':
        return (img / 255.0) * 2.0 - 1.0
    else:
        return img


def extract_action_from_poses(current_pose: np.ndarray, target_pose: np.ndarray,
                              current_gripper: float, target_gripper: float) -> np.ndarray:
    """
    Extract 8D action from current and target poses.

    Args:
        current_pose: (7,) [x, y, z, qx, qy, qz, qw]
        target_pose: (7,) [x, y, z, qx, qy, qz, qw]
        current_gripper: float [0, 1]
        target_gripper: float [0, 1]

    Returns:
        action: (8,) [dx, dy, dz, dqx, dqy, dqz, dqw, dg]
    """
    # Position delta
    pos_delta = target_pose[:3] - current_pose[:3]

    # Rotation delta (quaternion difference)
    curr_rot = R.from_quat(current_pose[3:7])
    tgt_rot = R.from_quat(target_pose[3:7])

    # Relative rotation
    rel_rot = curr_rot.inv() * tgt_rot
    rot_delta_quat = rel_rot.as_quat()  # [qx, qy, qz, qw]

    # Gripper delta
    gripper_delta = target_gripper - current_gripper

    action = np.concatenate([
        pos_delta,        # [3] dx, dy, dz
        rot_delta_quat,   # [4] dqx, dqy, dqz, dqw
        [gripper_delta]   # [1] gripper command
    ]).astype(np.float32)

    return action


def quat_to_6d_rotation(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to 6D rotation representation.

    6D rotation = first two columns of rotation matrix (flattened).

    Args:
        quat: (4,) quaternion [qx, qy, qz, qw] (scalar_last)

    Returns:
        (6,) rotation_6d representation
    """
    rot = R.from_quat(quat)
    rot_matrix = rot.as_matrix()  # (3, 3)

    # First two columns flattened
    rot_6d = rot_matrix[:, :2].flatten()  # (6,)

    return rot_6d.astype(np.float32)


def rot_6d_to_quat(rot_6d: np.ndarray) -> np.ndarray:
    """
    Convert 6D rotation representation back to quaternion.

    Args:
        rot_6d: (6,) rotation_6d representation

    Returns:
        (4,) quaternion [qx, qy, qz, qw] (scalar_last)
    """
    # Reconstruct 3x3 rotation matrix from 6D
    col1 = rot_6d[:3]
    col2 = rot_6d[3:6]

    # Orthonormalize using Gram-Schmidt
    col1 = col1 / np.linalg.norm(col1)
    col2 = col2 - np.dot(col1, col2) * col1
    col2 = col2 / np.linalg.norm(col2)

    # Third column (cross product)
    col3 = np.cross(col1, col2)

    rot_matrix = np.stack([col1, col2, col3], axis=1)

    rot = R.from_matrix(rot_matrix)
    quat = rot.as_quat()  # [qx, qy, qz, qw]

    return quat.astype(np.float32)
