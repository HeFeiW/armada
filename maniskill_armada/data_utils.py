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


def maniskill_obs_to_armada_format(obs: Dict, env: Any) -> Dict:
    """
    Convert ManiSkill observation dict to ARMADA format.

    Args:
        obs: ManiSkill observation dict
        env: ManiSkill environment (for ground truth extraction)

    Returns:
        dict with keys: tcp_pose, joint_pos, gripper_state
    """
    # Extract agent state (task-dependent)
    agent_obs = obs.get('agent', {})

    # TCP pose: position + quaternion (scalar_last format)
    tcp_pos = agent_obs.get('tcp_position', None)  # [3]

    if tcp_pos is None:
        raise ValueError("Cannot extract TCP position from obs")

    # Get TCP rotation
    tcp_rot_matrix = agent_obs.get('tcp_rotation', None)

    if tcp_rot_matrix is None:
        # Try from base frame + arm angles
        tcp_rot_matrix = np.eye(3)

    # Convert rotation matrix to quaternion (scalar_last: qx, qy, qz, qw)
    if tcp_rot_matrix.shape == (3, 3):
        rot = R.from_matrix(tcp_rot_matrix)
        tcp_quat = rot.as_quat()  # [qx, qy, qz, qw] scalar_last
    else:
        tcp_quat = tcp_rot_matrix  # Assume already quaternion

    # Full TCP pose: [x, y, z, qx, qy, qz, qw]
    tcp_pose = np.concatenate([tcp_pos, tcp_quat]).astype(np.float32)

    # Joint positions (first 7 DOF for arm)
    qpos = agent_obs.get('qpos', None)
    if qpos is not None:
        joint_pos = qpos[:7].astype(np.float32)
    else:
        joint_pos = np.zeros(7, dtype=np.float32)

    # Gripper state (0=open, 1=closed)
    gripper_state = agent_obs.get('gripper_qpos', np.array([0.0, 0.0]))
    gripper_value = np.mean(gripper_state).astype(np.float32)

    return {
        'tcp_pose': tcp_pose,           # (7,) float32
        'joint_pos': joint_pos,         # (7,) float32
        'gripper_state': gripper_value  # float32
    }


def render_cameras(env: Any, camera_names: Optional[list] = None,
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
    if camera_names is None:
        camera_names = ['viewer', 'wrist_camera']

    # Try rendering with env's render method
    try:
        # ManiSkill typically uses render(mode='rgb_array', camera_name=...)
        side_img = env.render(mode='rgb_array', camera_name=camera_names[0],
                              resolution=resolution)

        # For wrist, try wrist camera; fallback to second viewer
        try:
            wrist_img = env.render(mode='rgb_array', camera_name=camera_names[1],
                                   resolution=resolution)
        except:
            # Fallback to multiple viewers
            wrist_img = env.render(mode='rgb_array', camera_name='third_person_camera',
                                   resolution=resolution)

    except Exception as e:
        print(f"Warning: Camera rendering failed ({e}), using random images")
        # Fallback: dummy images
        side_img = np.random.randint(0, 256, (*resolution[::-1], 3), dtype=np.uint8)
        wrist_img = np.random.randint(0, 256, (*resolution[::-1], 3), dtype=np.uint8)

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
    rel_rot = tgt_rot * curr_rot.inv()
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
