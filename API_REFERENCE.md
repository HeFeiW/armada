# ARMADA 核心 API 接口参考

## 快速导览

| 模块 | 核心类 | 用途 |
|------|--------|------|
| `float/float.py` | `FLOAT` | 故障检测 |
| `hardware/robot_env.py` | `RobotEnv` | 硬件控制 |
| `armada/diffusion_policy/policy/*.py` | `*Policy` | 策略推理 |
| `armada/env_runner/real_env_runner.py` | `RealEnvRunner` | Episode执行 |
| `armada/nodes/communication_hub.py` | `CommunicationHub` | 消息路由 |
| `armada/utils/episode_manager.py` | `EpisodeManager` | 观测缓冲 |

---

## 1. FLOAT 故障检测 API

### 初始化与配置

```python
from float.float import FLOAT

float_detector = FLOAT()

# 运行时初始化 (在训练或加载模型后)
float_detector.runtime_initialize(
    policy=trained_policy,
    demo_data={
        'images': expert_images,        # [num_demos, T, C, H, W]
        'low_dim_state': expert_states, # [num_demos, T, state_dim]
        'actions': expert_actions       # [num_demos, T, action_dim]
    },
    config={
        'use_ot_matching': True,
        'ot_threshold_percentile': 80,  # 初始阈值
        'async_processing': True,
        ...
    }
)
```

### 推理接口

```python
# 每个策略步调用
obs = {
    'images': current_images,          # [B, T₀, C, H, W]
    'low_dim_state': current_state,    # [B, T₀, state_dim]
    'timestamps': timestamps           # [B, T₀]
}

# 故障检测 (异步, 非阻塞)
is_failure = float_detector.detect_failure(obs)

if is_failure:
    print("故障检测! 请求人工干预")
```

### Episode 管理

```python
# Step 处理
float_detector.process_step(
    state={'pose': [...], 'gripper': [...], ...},
    action=predicted_action
)

# 轨迹回放
float_detector.rewind_step()

# Episode 结束
float_detector.finalize_episode()
# → 更新OT统计, 自适应阈值调整
```

---

## 2. RobotEnv 硬件控制 API

### 初始化

```python
from hardware.robot_env import RobotEnv

robot_env = RobotEnv(
    robot_ip='192.168.1.100',
    camera_serials=['123456', '789012'],  # [侧摄像头, 腕摄像头]
    gripper_port='/dev/ttyUSB0',
    sigma_device='/dev/hidraw0',
    config={
        'image_resolution': (224, 224),
        'frequency': 10,  # Hz
        ...
    }
)
```

### 状态获取

```python
# 获取当前完整状态
robot_state = robot_env.get_robot_state()
# 返回:
# {
#     'tcp_pose': [x, y, z, qx, qy, qz, qw],  # TCP 位姿 (位置 + 四元数)
#     'joint_angles': [θ1, θ2, ..., θ7],      # 7个关节角
#     'gripper_state': float,                  # 0-1 (开->闭)
#     'images': {
#         'side': array(H, W, 3),              # 侧视摄像头
#         'wrist': array(H, W, 3)              # 腕部摄像头
#     },
#     'timestamp': float
# }
```

### 动作执行

```python
# 发送动作到机械臂
action = {
    'tcp_pose': [x, y, z, qx, qy, qz, qw],
    'gripper': gripper_command,
    # 或
    'joint_angles': [θ1, ..., θ7]
}

success = robot_env.deploy_action(action)
```

### 人工干预

```python
# 获取人类命令 (Sigma7 设备 / 键盘)
human_command = robot_env.human_teleop_step()
# 返回:
# {
#     'action': [...],
#     'device': 'sigma7' | 'keyboard' | 'gamepad',
#     'valid': bool
# }
```

### 复位与恢复

```python
# 复位到Home位, 打开夹爪
robot_env.reset_robot()

# 场景对齐 (与参考演示一致)
robot_env.align_with_reference(
    ref_observations={
        'images': reference_images,
        'states': reference_states
    }
)

# 回放回到安全状态 (应用反向动作)
robot_env.rewind_robot(
    action_history=[a0, a1, a2, ...],
    num_steps=5  # 回放多少步
)
```

---

## 3. 策略推理 API

### 加载策略

```python
# 任选一种策略
from armada.diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from armada.diffusion_policy.policy.diffusion_transformer_hybrid_dinov2_policy import DiffusionTransformerHybridDinoV2Policy

# 从配置或检查点加载
policy = DiffusionTransformerHybridDinoV2Policy(
    shape_meta={
        'observation': {
            'images': [3, 224, 224],
            'low_dim_state': [17]  # 位置, 速度, 夹爪等
        },
        'action': [8]  # 动作维度
    },
    noise_scheduler_config={...}
)

# 加载预训练权重 + EMA
checkpoint = torch.load('model.pt')
policy.load_state_dict(checkpoint['model'])
policy.eval()
```

### 推理

```python
# 准备观测
obs = {
    'images': images,              # [B, T₀, C, H, W]
    'low_dim_state': states,       # [B, T₀, state_dim]
    'timestamps': timestamps       # [B, T₀]
}

with torch.no_grad():
    # 预测 Ta 帧的动作序列
    actions = policy.predict_action(obs)  # [B, Ta, action_dim]

    # 通常取 t=0 的动作执行
    action_to_execute = actions[0, 0]  # [action_dim]
```

---

## 4. Episode 执行 API

### RealEnvRunner

```python
from armada.env_runner.real_env_runner import RealEnvRunner

runner = RealEnvRunner(
    config={
        'policy_checkpoint': 'path/to/model.pt',
        'robot_ip': '192.168.1.100',
        'demo_data_path': 'path/to/expert_demos.zarr',
        'max_steps_per_episode': 500,
        ...
    }
)

# 执行一个 Episode
episode_data = runner.run_episode(
    max_steps=500,
    max_retries=3,
    enable_failure_detection=True
)

# 返回:
# {
#     'success': bool,
#     'total_steps': int,
#     'failures_detected': int,
#     'human_interventions': int,
#     'observations': [...],
#     'actions': [...],
#     'rewards': [...]
# }
```

---

## 5. 通信 API

### Socket 服务器 (Hub)

```python
from armada.communication.socket_server import SocketServer

server = SocketServer(
    host='0.0.0.0',
    port=5000,
    max_connections=10
)

server.start_connection()

# 发送消息
msg = {
    'type': 'NEED_HUMAN_CHECK',
    'data': {
        'robot_id': 0,
        'step': 42,
        'reason': 'OT cost exceeded'
    }
}

server.send(client_addr, msg)
```

### Socket 客户端 (节点)

```python
from armada.communication.socket_client import SocketClient

client = SocketClient(
    host='127.0.0.1',
    port=5000
)

client.start_connection()

# 发送消息
msg = {
    'type': 'INFORM_ROBOT_STATE',
    'data': {...}
}

success = client.send(msg)  # 自动重试
```

### 消息类型汇总

| 消息类型 | 发送者 | 接收者 | 内容 |
|---------|--------|--------|------|
| `INFORM_ROBOT_STATE` | Robot | Hub | obs, state, step |
| `NEED_HUMAN_CHECK` | Robot | Hub | failure reason |
| `COMMAND` | Teleop | Hub | action |
| `REWIND_ROBOT` | Robot | Hub | num_steps |
| `EXECUTE_HUMAN_CHECK` | Hub | Robot | decision |
| `SCENE_ALIGNMENT_REQUEST` | Hub | Teleop | reference_obs |
| `SIGMA.*` | Teleop | Hub | device command |

---

## 6. Episode 缓冲 API

### EpisodeManager

```python
from armada.utils.episode_manager import EpisodeManager

em = EpisodeManager(
    shape_meta={
        'obs': {
            'images': [3, 224, 224],
            'low_dim_state': [17]
        }
    },
    buffer_size=3  # T₀ = 3 (history length)
)

# 每步更新
em.update_observation({
    'images': current_image,
    'low_dim_state': current_state,
    'timestamp': time.time()
})

# 获取批量化观测供策略推理
batch_obs = em.get_policy_observation()
# 返回:
# {
#     'images': [B, T₀, C, H, W],
#     'low_dim_state': [B, T₀, state_dim],
#     'timestamps': [B, T₀]
# }
```

---

## 7. 节点 API

### CommunicationHub (中心枢纽)

```python
from armada.nodes.communication_hub import CommunicationHub

hub = CommunicationHub(
    config={
        'host': '0.0.0.0',
        'port': 5000,
        'max_robots': 4,
        'max_operators': 4,
        ...
    }
)

# 启动
hub.start()

# 消息路由自动处理, 通过注册 handlers
hub.register_handler('NEED_HUMAN_CHECK', handler_func)
```

### RobotNode (机器人节点)

```python
from armada.nodes.robot_node import RobotNode

robot = RobotNode(
    robot_id=0,
    robot_ip='192.168.1.100',
    hub_host='127.0.0.1',
    hub_port=5000,
    policy_checkpoint='model.pt',
    config={...}
)

# 启动 Episode 循环
robot.run_episodes(num_episodes=10)
```

### TeleopNode (操作节点)

```python
from armada.nodes.teleop_node import TeleopNode

teleop = TeleopNode(
    operator_id=0,
    hub_host='127.0.0.1',
    hub_port=5000,
    config={
        'sigma_device': '/dev/hidraw0',
        'keyboard_enabled': True,
        ...
    }
)

# 启动
teleop.run()
```

---

## 8. 数据录制 API

```python
from record import record

# 录制一个 Episode
record(
    output_path='data/demos',
    resolution=(224, 224),
    fps=10
)
# → 保存 Zarr 格式: data/demos/episode_0001.zarr
```

---

## 完整最小示例

```python
# 1. 初始化硬件
from hardware.robot_env import RobotEnv
robot_env = RobotEnv(robot_ip='192.168.1.100', ...)

# 2. 加载策略
from armada.diffusion_policy.policy.diffusion_transformer_hybrid_dinov2_policy import Policy
policy = Policy.from_checkpoint('model.pt')

# 3. 初始化故障检测
from float.float import FLOAT
float_detector = FLOAT()
float_detector.runtime_initialize(policy, demo_data, config)

# 4. 初始化 Episode 缓冲
from armada.utils.episode_manager import EpisodeManager
ep_manager = EpisodeManager(...)

# 5. 执行 Episode 循环
for step in range(500):
    # 获取观测
    obs = robot_env.get_robot_state()
    ep_manager.update_observation(obs)

    # 策略推理
    batch_obs = ep_manager.get_policy_observation()
    actions = policy.predict_action(batch_obs)
    action = actions[0, 0]  # 第一个批次, 第一个时间步

    # 故障检测
    is_failure = float_detector.detect_failure(batch_obs)

    if is_failure:
        print("检测到故障, 回放...")
        robot_env.rewind_robot(action_history, num_steps=5)
        float_detector.update_threshold()
        continue

    # 执行动作
    robot_env.deploy_action(action)
    float_detector.process_step(obs, action)
```

---

## 常见配置参数

### RobotEnv

```python
config = {
    'image_resolution': (224, 224),
    'frequency': 10,                    # Hz
    'enable_gripper': True,
    'enable_sigma': True,
    'sigma_sensitivity': 0.5,
    'max_action_magnitude': 0.1
}
```

### FLOAT

```python
config = {
    'use_ot_matching': True,
    'ot_threshold_percentile': 80,      # 初始阈值百分位
    'ot_threshold_min': 0.1,            # 最小阈值
    'ot_threshold_max': 1.0,            # 最大阈值
    'async_processing': True,
    'batch_size': 16,
    'latent_dim': 256,
    'cost_fn': 'l2'                     # OT成本函数
}
```

### RealEnvRunner

```python
config = {
    'policy_checkpoint': 'path/to/model.pt',
    'robot_ip': '192.168.1.100',
    'demo_data_path': 'path/to/expert_demos.zarr',
    'max_steps_per_episode': 500,
    'enable_failure_detection': True,
    'enable_human_intervention': True,
    'log_dir': 'logs/',
    'save_video': True,
    'device': 'cuda'
}
```
