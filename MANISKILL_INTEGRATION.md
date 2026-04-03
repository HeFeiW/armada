# Maniskill 复现指南

本指南帮助在 ManiSkill 环境中复现 ARMADA 系统的关键功能。

---

## 1. 核心模块需要复现

### 优先级 1: FLOAT 故障检测
- **目标**: 在 ManiSkill 环境中实现基于最优运输的故障检测
- **关键文件**: `float/float.py`, `float/util.py`
- **接口**:
  ```python
  float_detector = FLOAT()
  float_detector.runtime_initialize(policy, demo_data, config)
  is_failure = float_detector.detect_failure(obs)
  ```
- **核心算法**:
  1. 策略编码器: 提取观测的潜在表示
  2. OT 匹配: 当前观测 vs 专家演示库
  3. 自适应阈值: 基于历史成本的百分位

### 优先级 2: RobotEnv 环境抽象
- **目标**: 在 ManiSkill 中处理观测、动作、硬件接口
- **关键文件**: `hardware/robot_env.py`
- **需要适配**:
  - `get_robot_state()`: 返回 {images, state, timestamp}
  - `deploy_action(action)`: 在环境中执行动作
  - `reset_robot()`: ManiSkill 的 reset() 方法
  - `rewind_robot(history, steps)`: 实现轨迹回放

### 优先级 3: EpisodeManager 观测缓冲
- **目标**: 维护历史观测用于策略输入
- **关键文件**: `armada/utils/episode_manager.py`
- **简化实现**:
  ```python
  class EpisodeManager:
      def __init__(self, buffer_size=3):
          self.buffer = collections.deque(maxlen=buffer_size)

      def update_observation(self, obs):
          self.buffer.append(obs)

      def get_policy_observation(self):
          # 返回 [images], [states] 的历史
          return self._stack_observations()
  ```

### 优先级 4: 策略加载与推理
- **目标**: 加载预训练策略并进行推理
- **关键文件**: `armada/diffusion_policy/policy/*.py`
- **接口**:
  ```python
  policy = load_policy('checkpoint.pt')
  obs = {'images': [...], 'low_dim_state': [...]}
  action = policy.predict_action(obs)  # [Ta, action_dim]
  ```

### 优先级 5: Episode 运行循环
- **目标**: 串联上述模块, 实现完整Episode执行
- **参考**: `armada/env_runner/real_env_runner.py`
- **流程**:
  ```
  reset() → for step in range(max_steps):
      obs = env.get_state()
      ep_manager.update_observation(obs)
      action = policy.predict_action(ep_manager.get_obs())
      is_failure = float_detector.detect_failure(obs)
      if is_failure:
          env.rewind(history, steps=5)
          float_detector.update_threshold()
      else:
          env.step(action)
  ```

---

## 2. ManiSkill 环境适配

### 创建 ManiSkillRobotEnv 包装器

```python
# maniskill_armada/robot_env_maniskill.py

from hardware.robot_env import RobotEnv

class ManiSkillRobotEnv(RobotEnv):
    """ManiSkill 环境的 RobotEnv 适配器"""

    def __init__(self, env_id, config):
        """
        Args:
            env_id: ManiSkill 环境ID
            config: 配置字典
        """
        import gymnasium as gym
        self.env = gym.make(env_id)
        self.config = config
        # 不初始化硬件 (ManiSkill 模拟)

    def get_robot_state(self):
        """从 ManiSkill 获取状态"""
        obs, info = self.env.observation, self.env.info

        # 提取相关信息
        return {
            'tcp_pose': extract_tcp_pose(obs),      # 6DOF 或 7DOF
            'joint_angles': extract_joint_angles(obs),
            'gripper_state': extract_gripper(obs),
            'images': {
                'side': render_side_view(),         # 从 ManiSkill 渲染
                'wrist': render_wrist_view()
            },
            'timestamp': time.time()
        }

    def deploy_action(self, action):
        """在 ManiSkill 中执行动作"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return terminated or truncated

    def reset_robot(self):
        """复位 ManiSkill 环境"""
        self.env.reset()

    def rewind_robot(self, action_history, num_steps):
        """
        回放轨迹 (ManiSkill 中的实现)

        方案:
        1. 保存环境快照 (每步)
        2. 回放时恢复到 current - num_steps 的快照
        """
        if hasattr(self.env, 'save_state'):
            # 方案A: 环境内部状态保存
            self.env.restore_state(self.env_snapshots[-num_steps])
        else:
            # 方案B: 反向重放 (应用反向动作)
            # 简化: 直接 reset + 重新执行到 current-num_steps
            self.reset_and_replay_to_step(len(action_history) - num_steps)
```

### 观测格式转换

```python
# maniskill_armada/utils.py

def extract_tcp_pose(maniskill_obs):
    """从 ManiSkill obs 提取 TCP 位姿"""
    # ManiSkill 通常提供不同的观测格式
    # 需要根据具体环境调整
    if 'agent' in maniskill_obs:
        agent_obs = maniskill_obs['agent']
        # 提取 TCP 位置和旋转
        tcp_pose = agent_obs['tcp_pose']  # [7] or [6]
        return tcp_pose
    return None

def render_side_view():
    """渲染侧视图"""
    return env.render(mode='rgb_array', camera_name='side_camera')

def render_wrist_view():
    """渲染腕部视图"""
    return env.render(mode='rgb_array', camera_name='wrist_camera')
```

---

## 3. 文件结构建议

```
maniskill_armada/
├── __init__.py
├── robot_env_maniskill.py          # ManiSkill 环境适配器
├── utils.py                         # 观测提取, 渲染等
├── run_episode.py                   # Episode 执行脚本
├── evaluate.py                      # 评估脚本
└── configs/
    ├── maniskill_env.yaml           # 环境配置
    ├── policy_checkpoint.pt         # 预训练策略
    └── expert_demos.zarr            # 专家演示 (用于FLOAT初始化)

# 在项目root, 保留原始ARMADA代码
armada/
float/
hardware/
```

---

## 4. 关键实现步骤

### Step 1: 最小可行循环

```python
# maniskill_armada/run_episode.py

import gymnasium as gym
from maniskill_armada.robot_env_maniskill import ManiSkillRobotEnv
from armada.utils.episode_manager import EpisodeManager

def run_episode_minimal(env_id, policy, max_steps=500):
    """最小可行 Episode (不含故障检测)"""

    env = ManiSkillRobotEnv(env_id, config={})
    ep_manager = EpisodeManager(buffer_size=3)

    obs, info = env.env.reset()
    done = False
    step = 0

    while not done and step < max_steps:
        # 1. 获取观测
        robot_state = env.get_robot_state()
        ep_manager.update_observation(robot_state)

        # 2. 策略推理
        batch_obs = ep_manager.get_policy_observation()
        action = policy.predict_action(batch_obs)[0, 0]

        # 3. 执行动作
        terminated = env.deploy_action(action)
        done = terminated or done

        step += 1

    return {'success': not done, 'steps': step}
```

### Step 2: 添加 FLOAT 故障检测

```python
# maniskill_armada/run_episode_with_failure_detection.py

from float.float import FLOAT

def run_episode_with_failure_detection(env_id, policy, expert_demos, max_steps=500):
    """Episode 执行 + 故障检测"""

    env = ManiSkillRobotEnv(env_id, config={})
    ep_manager = EpisodeManager(buffer_size=3)
    float_detector = FLOAT()

    # 初始化 FLOAT
    float_detector.runtime_initialize(policy, expert_demos, config={
        'use_ot_matching': True,
        'async_processing': True
    })

    obs, info = env.env.reset()
    done = False
    step = 0
    failures = 0

    while not done and step < max_steps:
        robot_state = env.get_robot_state()
        ep_manager.update_observation(robot_state)

        # 策略推理
        batch_obs = ep_manager.get_policy_observation()
        action = policy.predict_action(batch_obs)[0, 0]

        # 故障检测
        is_failure = float_detector.detect_failure(batch_obs)

        if is_failure:
            print(f"Step {step}: 故障检测! 执行回放...")
            env.rewind_robot(action_history=[], num_steps=5)
            failures += 1
            float_detector.update_threshold()
            # 可选: 等待或重新初始化
            continue

        # 执行动作
        terminated = env.deploy_action(action)
        done = terminated or done
        float_detector.process_step(robot_state, action)

        step += 1

    float_detector.finalize_episode()

    return {
        'success': not done,
        'steps': step,
        'failures_detected': failures
    }
```

### Step 3: 评估与统计

```python
# maniskill_armada/evaluate.py

def evaluate(env_id, policy, expert_demos, num_episodes=10):
    """评估多个 Episode"""

    results = {
        'success_rate': 0,
        'avg_steps': 0,
        'avg_failures': 0
    }

    successes = 0
    total_steps = 0
    total_failures = 0

    for ep in range(num_episodes):
        result = run_episode_with_failure_detection(
            env_id, policy, expert_demos
        )

        if result['success']:
            successes += 1

        total_steps += result['steps']
        total_failures += result.get('failures_detected', 0)

    results['success_rate'] = successes / num_episodes
    results['avg_steps'] = total_steps / num_episodes
    results['avg_failures'] = total_failures / num_episodes

    return results
```

---

## 5. 集成检查清单

- [ ] ManiSkill 环境装好 (`gymnasium`, `maniskill2`)
- [ ] ARMADA 核心模块可导入 (float, hardware, armada)
- [ ] 已加载预训练策略 (.pt 文件)
- [ ] 已准备专家演示数据 (.zarr 文件, 用于 FLOAT)
- [ ] 实现了 `ManiSkillRobotEnv` 的关键方法
- [ ] 最小循环可运行 (Step 1: run_episode_minimal.py)
- [ ] FLOAT 集成正常 (Step 2: run_episode_with_failure_detection.py)
- [ ] 评估脚本可生成指标 (Step 3: evaluate.py)
- [ ] 记录 Episode 数据 (observations, actions, failures)

---

## 6. 潜在问题与解决

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| OT 匹配速度慢 | 专家演示太多 | 子采样演示或使用快速OT算法 |
| ManiSkill 观测维度不匹配 | 不同环境 | 编写适配器 extract_tcp_pose, render_* |
| 回放失败 | 环境不可复现 | 使用环保快照或接受不完美回放 |
| 策略推理失败 | 观测格式错误 | 调试 EpisodeManager 的 update_observation |
| 显存溢出 | 全GPU训练 | 减小 batch_size 或使用混合精度 |

---

## 7. 性能期望

根据原论文:
- **成功率**: ~85-95% (依任务而定)
- **故障检测准确度**: ~95%
- **平均 Episode 长度**: 100-500 步 (取决于任务)
- **故障恢复时间**: < 100 ms (使用异步FLOAT)

在 ManiSkill 中可能：
- 成功率相近或更高 (环境更可控)
- 故障检测可靠性取决于演示多样性

---

## 8. 参考资源

- 原 ARMADA 代码: `/home/stu4/armada/`
- ARCHITECTURE.md: 详细架构说明
- API_REFERENCE.md: API 接口文档
- ManiSkill 文档: https://maniskill.org/
