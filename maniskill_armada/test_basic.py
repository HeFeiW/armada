#!/usr/bin/env python3
"""
Quick test of ManiSkill environment and basic data collection
"""

import sys
sys.path.insert(0, '/home/stu4/armada')
sys.path.insert(0, '/home/stu4')

import gymnasium as gym
import numpy as np
from maniskill_armada.heuristic_policy import HeuristicPickPolicy
from maniskill_armada.data_utils import maniskill_obs_to_armada_format, render_cameras

def test_environment():
    """Test if ManiSkill environment can be created and stepped"""
    print("Testing ManiSkill environment...")

    try:
        # Try creating environment
        env = gym.make('PickCube-v1')
        print("✓ Environment created successfully")

        # Test reset
        obs, info = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  Observation keys: {obs.keys() if isinstance(obs, dict) else 'array'}")

        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("✓ Environment step successful")

        return env

    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return None


def test_observation_extraction(env):
    """Test observation conversion"""
    print("\nTesting observation extraction...")

    try:
        obs, _ = env.reset()

        armada_obs = maniskill_obs_to_armada_format(obs, env)
        print("✓ Observation extraction successful")
        print(f"  TCP pose shape: {armada_obs['tcp_pose'].shape}")
        print(f"  Joint pos shape: {armada_obs['joint_pos'].shape}")
        print(f"  Gripper state: {armada_obs['gripper_state']:.3f}")

    except Exception as e:
        print(f"✗ Observation extraction failed: {e}")
        return False

    return True


def test_camera_rendering(env):
    """Test camera rendering"""
    print("\nTesting camera rendering...")

    try:
        side_img, wrist_img = render_cameras(env)
        print("✓ Camera rendering successful")
        print(f"  Side image shape: {side_img.shape}, dtype: {side_img.dtype}")
        print(f"  Wrist image shape: {wrist_img.shape}, dtype: {wrist_img.dtype}")

    except Exception as e:
        print(f"✗ Camera rendering failed: {e}")
        return False

    return True


def test_policy(env):
    """Test heuristic policy"""
    print("\nTesting heuristic policy...")

    try:
        policy = HeuristicPickPolicy(env)
        obs, _ = env.reset()

        action = policy.get_action(obs, 'stage1')
        print("✓ Policy inference successful")
        print(f"  Action shape: {action.shape}, dtype: {action.dtype}")
        print(f"  Action values: {action}")

    except Exception as e:
        print(f"✗ Policy test failed: {e}")
        return False

    return True


def test_mini_collection(env):
    """Test collecting a few steps"""
    print("\nTesting mini data collection (5 steps)...")

    try:
        policy = HeuristicPickPolicy(env)
        obs, _ = env.reset()

        data = {
            'wrist_cam': [],
            'side_cam': [],
            'tcp_pose': [],
            'joint_pos': [],
            'action': []
        }

        for step in range(5):
            side_img, wrist_img = render_cameras(env)
            armada_obs = maniskill_obs_to_armada_format(obs, env)
            action = policy.get_action(obs, 'stage1')

            data['wrist_cam'].append(wrist_img)
            data['side_cam'].append(side_img)
            data['tcp_pose'].append(armada_obs['tcp_pose'])
            data['joint_pos'].append(armada_obs['joint_pos'])
            data['action'].append(action)

            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        # Stack data
        for key in data:
            if len(data[key]) > 0:
                data[key] = np.stack(data[key])

        print("✓ Mini collection successful")
        print(f"  Steps collected: {len(data['action'])}")
        print(f"  Wrist cam shape: {data['wrist_cam'].shape}")
        print(f"  TCP pose shape: {data['tcp_pose'].shape}")
        print(f"  Action shape: {data['action'].shape}")

    except Exception as e:
        print(f"✗ Mini collection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == '__main__':
    print("="*60)
    print("ManiSkill + ARMADA Integration Tests")
    print("="*60)

    # Run tests
    env = test_environment()
    if env is None:
        print("\nTest failed: Cannot create environment")
        sys.exit(1)

    test_observation_extraction(env)
    test_camera_rendering(env)
    test_policy(env)
    test_mini_collection(env)

    env.close()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
