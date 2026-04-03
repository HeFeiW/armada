# ARMADA 架构与代码结构文档

## 目录
1. [系统架构](#系统架构)
2. [模块详解](#模块详解)
3. [数据流与通信](#数据流与通信)
4. [核心接口](#核心接口)
5. [项目配置](#项目配置)

---

## 系统架构

### 高层组件图

```
┌─────────────────────────────────────────────────┐
│        Communication Hub (中心消息代理)          │
│   - socket_server.py + CommunicationHub         │
│   - 路由机器人/操作员消息                       │
│   - 管理状态队列 & 请求队列                     │
└────────┬──────────────────────────────┬─────────┘
         │                              │
    ┌────▼────────┐            ┌───────▼─────┐
    │ Robot Node  │            │ Teleop Node │
    │             │            │             │
    │┌──────────┐ │            │┌─────────┐  │
    ││Policy    │ │          Σ │人类输入  │  │
    ││Inference ├┼─────────────┤(Sigma7) │  │
    ││          │ │            │(键盘)  │  │
    ││FLOAT     │ │            ││游戏柄  │  │
    ││Detector  │ │            │└─────────┘  │
    │└──────────┘ │            └─────────────┘
    │             │
    │ Hardware:   │
    │ - Flexiv机械臂
    │ - 双摄像头  │
    │ - 夹爪      │
    ├─────────────┤
    │RobotEnv    │ (硬件抽象层)
    └─────────────┘
```

### 执行流水线

```
Step t=n:
  Robot State (Pose, Gripper, Images)
         │
         ▼
  EpisodeManager (维护 T₀=3 帧历史观测)
         │
         ▼
  Policy.predict_action() → Action (动作序列)
         │
         ▼
  ┌──────────────────────────────┐
  │ FLOAT: 异步故障检测           │
  │ - 当前观测 → 专家演示的OT匹配 │
  │ - OT成本 vs 自适应阈值        │
  └──────────────────────────────┘
         │
    ┌────┴────────┐
    │ 故障?       │
    ├─────┬──────┐
   否     │     是
    │     │      │
    ▼     ▼      ▼
  继续  请求人工  回放机器人
       干预      → 恢复状态
                 → 更新OT阈值
```

---

## 模块详解

### 1. float/ - 故障检测与最优运输

| 文件 | 主要类/函数 | 作用 |
|------|-----------|------|
| `float.py` | `FLOAT(AsyncFailureDetectionModule)` | 基于OT的实时故障检测器 |
| `util.py` | `compute_ot_cost()` | 两点云间最优运输成本计算 |
| `async_failure_detector.py` | `AsyncFailureDetectionModule` | 异步处理基类（线程安全） |
| `base_failure_detector.py` | `BaseFailureDetector` | 基类接口 |

#### FLOAT 核心接口

```python
class FLOAT(AsyncFailureDetectionModule):
    def runtime_initialize(policy, demo_data, config):
        """初始化: 提取专家演示的潜在表示"""

    def detect_failure(obs) -> bool:
        """实时故障检测: 当前OT成本 > 自适应阈值?"""

    def process_step(state, action):
        """处理每个策略步"""

    def rewind_step():
        """处理轨迹回放"""

    def finalize_episode():
        """更新OT统计 & 自适应阈值"""
```

**工作原理**:
1. 策略编码器提取当前观测的潜在表示
2. 与专家演示库计算最优运输（贪心OT计划）
3. OT成本 > `threshold` → 触发失败信号
4. 失败后更新阈值（自适应百分位数）

---

### 2. hardware/ - 硬件抽象层

| 文件 | 主要类 | 功能 |
|------|--------|------|
| `robot_env.py` | `RobotEnv` | 统一硬件接口 |
| `my_device/robot.py` | `FlexivRobot` | Flexiv机械臂驱动 |
| `my_device/camera.py` | `CameraD400` | RealSense D400双目摄像头 |
| `my_device/sigma.py` | `Sigma7Device` | Sigma.7 力反馈设备 |
| `my_device/keyboard.py` | `KeyboardListener` | 键盘事件监听 |
| `robot_env.py` | `RobotEnv` | 硬件集成入口 |

#### RobotEnv 核心接口

```python
class RobotEnv:
    def __init__(robot_ip, camera_sn, config):
        """初始化: Flexiv机械臂, 摄像头, 夹爪, Sigma7, 键盘"""

    def get_robot_state() -> dict:
        """获取: TCP位姿, 关节角, 双摄像头图像, 夹爪状态"""

    def human_teleop_step() -> command:
        """捕获: 人类操作员命令 (Sigma7设备/键盘/游戏柄)"""

    def deploy_action(action) -> success:
        """执行: 发送动作给机械臂"""

    def reset_robot():
        """复位: 回到Home位, 打开夹爪"""

    def align_with_reference(ref_state):
        """对齐: 视觉场景与参考演示对齐"""

    def rewind_robot(history, steps):
        """回放: 应用反向动作恢复到安全状态"""
```

**关键属性**:
- `robot`: FlexivRobot 驱动
- `cameras`: [侧视摄像头, 腕部摄像头]
- `gripper`: 夹爪控制
- `sigma`: Sigma.7 力反馈设备
- `keyboard_listener`: 键盘事件

---

### 3. armada/ - 训练与部署框架

#### 3.1 armada/diffusion_policy/ - 策略模型

| 路径 | 主要类 | 作用 |
|------|--------|------|
| `policy/base_lowdim_policy.py` | `BaseLowdimPolicy` | 策略基类 |
| `policy/diffusion_unet_lowdim_policy.py` | `DiffusionUnetLowdimPolicy` | Diffusion UNet (本体状态) |
| `policy/diffusion_unet_image_policy.py` | `DiffusionUnetImagePolicy` | Diffusion UNet (视觉) |
| `policy/diffusion_transformer_hybrid_dinov2_policy.py` | `DiffusionTransformerHybridDinoV2Policy` | Vision Transformer (DinoV2编码器) |
| `dataset/mydataset.py` | `MyDataset` | Zarr回放缓冲区数据集 |
| `dataset/base_dataset.py` | `BaseDataset` | 数据集基类 |

#### 策略接口 (BaseLowdimPolicy)

```python
class BaseLowdimPolicy:
    def predict_action(obs: dict) -> action:
        """obs: {image: [...], state: [...], timestamp: ...}
           action: [Ta, action_dim] (动作序列)"""

    def reset():
        """重置策略状态"""

    def train_mode() / eval_mode():
        """切换训练/推理模式"""
```

**输入格式**:
- `obs['images']`: [B, T₀, C, H, W] (批量, 历史帧, 通道, 高, 宽)
- `obs['low_dim_state']`: [B, T₀, state_dim] (本体状态: 位置, 速度等)
- `obs['timestamps']`: [B, T₀]

**输出格式**:
- Action: [B, Ta, action_dim] (预测未来 Ta 帧的动作)

#### 3.2 armada/communication/ - 网络通信

| 文件 | 主要类 | 功能 |
|------|--------|------|
| `socket_server.py` | `SocketServer` | TCP服务器(中心枢纽) |
| `socket_client.py` | `SocketClient` | TCP客户端(节点) |

#### 通信协议

```python
class SocketServer:
    def send(addr, message: dict):
        """广播消息到指定客户端
           格式: <<MSG_START>>{serialized_message}<<MSG_END>>"""

class SocketClient:
    def send(data: dict) -> bool:
        """发送到服务器 (最多重试3次)"""
```

**消息格式**: JSON + 自定义分隔符
- 开始: `<<MSG_START>>`
- 结束: `<<MSG_END>>`
- 内容: `{type, data}`

---

#### 3.3 armada/nodes/ - 分布式节点

| 文件 | 主要类 | 职责 |
|------|--------|------|
| `communication_hub.py` | `CommunicationHub` | 中央消息代理 & 状态管理器 |
| `robot_node.py` | `RobotNode(RealEnvRunner)` | 机器人控制节点 |
| `teleop_node.py` | `TeleopNode` | 人工操作节点 |

#### CommunicationHub - 消息路由

```python
class CommunicationHub:
    # 消息队列
    robot_dict[robot_id] = {observations, state, step_index}
    teleop_dict[operator_id] = {commands, active, state}
    request_queue = [需要人工干预的请求]
    scene_alignment_queue = [场景对齐请求]

    # 消息类型 (locked=同步, unlocked=异步)
    LOCKED:
        - NEED_HUMAN_CHECK: 机器人请求人工检查
        - INFORM_TELEOP_STATE: 通知操作员状态改变
        - REPORT_FAILURE: 上报故障

    UNLOCKED:
        - TELEOP_CTRL_START: 开始手动控制
        - TELEOP_CTRL_STOP: 停止手动控制
        - COMMAND: 发送操作命令
        - REWIND_ROBOT: 回放机器人
        - RESET: 复位

    DEVICE:
        - SIGMA.*: Sigma.7 设备命令 (DETACH, RESUME, RESET, ...)
```

#### RobotNode - 机器人执行器

```python
class RobotNode(RealEnvRunner):
    # 继承自 RealEnvRunner (策略执行)

    def _initialize_robot_state():
        """初始化状态机: idle / teleop_controlled / agent_controlled"""

    def _setup_message_routes():
        """注册来自Hub的消息处理器"""

    # 执行循环 (伪代码)
    for step in episode:
        obs = robot_env.get_state()

        if state == 'teleop_controlled':
            action = human_command
        else:
            # 策略推理
            action = policy.predict(obs)

            # 故障检测
            is_failure = float.detect_failure(obs)
            if is_failure:
                # 请求人工干预
                hub.send('NEED_HUMAN_CHECK')
                # 等待人工决定...

                # 回放恢复
                robot_env.rewind_robot(history, steps=5)

                # 更新OT阈值
                float.update_threshold()

        # 执行动作
        robot_env.deploy_action(action)
```

#### TeleopNode - 人工操作器

```python
class TeleopNode:
    def _initialize_devices():
        """初始化: Sigma.7力反馈设备, 游戏柄, 键盘"""

    def _setup_message_routes():
        """注册来自Hub的消息处理器"""

    def handle_scene_alignment_request():
        """场景对齐: 指导人类将场景与参考设置对齐"""

    def human_decide_process():
        """人工决策: 继续 / 从不同起点重试 / 手动控制"""
```

---

#### 3.4 armada/env_runner/ - 策略执行引擎

| 文件 | 主要类 | 作用 |
|------|--------|------|
| `base_env_runner.py` | `BaseEnvRunner` | 基类 |
| `real_env_runner.py` | `RealEnvRunner` | 真实机器人执行器 |

#### RealEnvRunner - 核心执行器

```python
class RealEnvRunner(BaseEnvRunner):
    def _load_policy(checkpoint_path):
        """加载训练模型 + EMA"""

    def _setup_transformers():
        """旋转表示转换 (四元数 ↔ 欧拉角 ↔ 轴角)"""

    def _initialize_robot_env():
        """初始化RobotEnv (支持多机器人)"""

    def _initialize_episode_manager():
        """创建EpisodeManager (观测缓存)"""

    def _initialize_failure_detection_module():
        """初始化FLOAT故障检测器"""

    # 执行 Episode
    def run_episode(max_steps):
        """执行一个完整Episode:
           1. 初始化环境
           2. 循环直到成功或最大步数
             - 获取观测
             - 策略推理
             - FLOAT故障检测
             - 执行动作或回放恢复
           3. 记录结果到回放缓冲区"""
```

---

#### 3.5 armada/utils/ - 工具模块

| 文件 | 主要类/函数 | 功能 |
|------|-----------|------|
| `episode_manager.py` | `EpisodeManager` | 有状态观测缓冲 |
| `message_distillation.py` | 消息解析/路由 | 消息格式转换 |
| `keyboard_listener.py` | `KeyboardListener` | 全局键盘事件监听 |

#### EpisodeManager

```python
class EpisodeManager:
    def update_observation(obs_dict):
        """维护 T₀=3 帧的历史观测"""

    def get_policy_observation() -> dict:
        """返回批量化观测供策略推理:
           - images: [B, T₀, C, H, W]
           - low_dim_state: [B, T₀, state_dim]
           - timestamps: [B, T₀]"""
```

---

### 4. record.py - 数据录制

```python
def record(output_path, resolution, fps):
    """录制一个人类演示Episode:
       1. 初始化RobotEnv
       2. 循环获取机器人状态 + 人类命令
       3. 将观测和动作保存到Zarr回放缓冲区"""

def main():
    """多Episode录制循环
       参数: --output, --resolution, --fps"""
```

**输出格式**: Zarr格式 (高效压缩)
- `images`: 摄像头图像
- `states`: 机器人状态 (位置, 速度, 夹爪)
- `actions`: 人类命令

---

## 数据流与通信

### Episode 执行流程

```
时间轴:
├─ Episode 开始
│  └─ robot_env.reset_robot()
│
├─ Step t=0:  obs[0] → policy → action[0..Ta-1] → FLOAT → execute
├─ Step t=1:  obs[1] → policy → action[0..Ta-1] → FLOAT → execute
├─ ...
│
├─ 故障检测触发 (t=k):
│  ├─ FLOAT.detect_failure() = True
│  ├─ hub.send('NEED_HUMAN_CHECK')
│  ├─ [等待人工决策...]
│  ├─ robot_env.rewind_robot(history, steps=5)  # 回放5步
│  ├─ FLOAT.update_threshold()  # 自适应阈值
│  └─ 恢复或继续
│
└─ Episode 结束 (成功或失败)
   └─ 保存到回放缓冲区
```

### Socket 通信 (Hub ↔ 节点)

```
Robot Node                 Hub Server              Teleop Node
│                          │                       │
├─ INFORM_ROBOT_STATE ────>│                       │
│                          ├─ 路由 ──────────────>│
│                          │                       │
│                      [处理]                      │
│                          │                       │
│                          │<─ 人工决策 ──────────┤
│                          │                       │
│<─────── EXECUTE_HUMAN ───┤                       │
│                          │                       │
└─ REWIND_ROBOT ──────────>│                       │
                           │
```

---

## 核心接口

### 配置系统 (Hydra)

```yaml
# armada/config/training/
├─ default.yaml: 默认训练配置
├─ policy/
│  ├─ diffusion_unet_lowdim.yaml
│  ├─ diffusion_unet_image.yaml
│  └─ diffusion_transformer_hybrid_dinov2.yaml
├─ dataset/
│  └─ default.yaml (路径, batch_size等)
└─ ... (其他子配置)
```

### 环境变量与启动

```bash
# 数据录制
python record.py --output /path --resolution 224 224 --fps 10

# 训练 (见 armada/train.sh)
python armada/train.py \
  --config-path=config/training \
  --config-name=default \
  policy=diffusion_transformer_hybrid_dinov2 \
  ...

# 推理/Rollout
python armada/run_rollout.py \
  --checkpoint=/path/to/model.pt \
  --robot-ip=<ip> \
  ...
```

---

## 关键设计决策

| 决策 | 原因 |
|------|------|
| **异步FLOAT** | 避免故障检测阻塞策略执行 |
| **EpisodeManager** | 缓冲历史观测, 减少I/O |
| **Socket TCP** | 跨机器网络通信相对简单 |
| **Zarr格式** | 高效压缩 & 快速随机访问 |
| **自适应阈值** | 适应不同任务的故障特性 |
| **回放恢复** | 无需重新初始化, 快速恢复 |
| **对齐步骤** | 确保视觉场景与参考演示一致 |

---

## 扩展与定制

### 添加新设备
1. `hardware/my_device/new_device.py`: 实现驱动
2. `hardware/robot_env.py`: 在 `__init__` 中集成
3. `armada/nodes/teleop_node.py`: 添加消息处理 (如需)

### 添加新策略
1. 继承 `BaseLowdimPolicy` 或 `BaseImagePolicy`
2. 实现 `predict_action(obs)`
3. 在 `armada/config/training/policy/` 中创建配置

### 多机器人支持
- `RobotNode` 与 `TeleopNode` 通过 `robot_id` 与 `operator_id` 区分
- `CommunicationHub` 维护字典: `robot_dict[robot_id]`, `teleop_dict[operator_id]`
- 启动多个 `RobotNode` 实例 (不同 IPs/端口)

---

## 参考文献
- 主论文: https://arxiv.org/abs/2510.02298
- 基础框架: Diffusion Policy (github.com/real-stanford/diffusion_policy)
