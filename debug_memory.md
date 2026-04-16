# Debug Memory

## Goal
定位 ManiSkill PickCube policy rollout 效果差的根因：数据采集、训练、还是部署链路。

## Current Facts (2026-04-10)
- 数据结构检查通过：`replay_buffer.zarr` 包含 `action/joint_pos/tcp_pose/wrist_cam/side_cam/action_mode/failure_indices`，总步数 30180，总 episode 200。
- 用户观察：渲染 200 条视频中机器人动态看起来正常。
- 用户疑点 A：训练环境里是棕色桌子+红色方块，但渲染视频里发蓝。
- 用户疑点 B：视频每条长度常见 150~180，但 rollout 中“自动获取人类演示 latent”阶段看到长度 56。

## Findings

### F1. 渲染颜色偏蓝是通道重复转换导致
- 根因：`render_rollout_videos.py` 中 `_make_episode_frame()` 已输出 BGR 帧，但写视频时又执行一次 `cv2.COLOR_RGB2BGR`，造成 R/B 互换。
- 已修复：移除写入前的二次转换，直接写帧。
- 文件：`maniskill_armada/render_rollout_videos.py`

### F2. “56” 大概率不是原始帧数，而是 FLOAT 内部的 chunk/索引长度
- FLOAT 使用 `Ta` 分块处理，核心索引是 `idx = timestep // Ta - 1`。
- 多处以 `demo_len // Ta` 和 `max_episode_length // Ta` 建图与计算 OT，而非直接按原始帧步。
- 因此看到 `56` 时，常见含义之一是“56 个 chunk”，对应原始步可近似为 `56 * Ta`。
- 相关文件：
  - `float/float.py`
  - `armada/env_runner/real_env_runner.py`

## Open Questions
- Q1: 你看到的“56”具体来自哪一行日志（原文）？
- Q2: 该次 rollout 的 `Ta`、`max_episode_length`、以及是否启用了 cache（`human_demo_latent_cache.npz`）分别是多少？
- Q3: 该次 FLOAT 匹配使用的数据集路径是 `train_dataset_path` 还是 `save_buffer_path`？

## Next Verification Steps
1. 重渲染 3 条 episode（修复后）确认颜色恢复正常。
2. 在 rollout 日志中定位“56”原始打印行，判断其语义是：
   - chunk index (`idx`)；
   - `demo_len // Ta`；
   - 人类 episode 数；
   - 或其他统计。
3. 临时加打印（建议）：
   - `Ta`
   - `demo_len`
   - `demo_len // Ta`
   - `max_episode_length`
   - `max_episode_length // Ta`
4. 若 `human_demo_latent_cache.npz` 已存在，记录其路径并确认签名是否命中/重建。

## Rollout Debug Switch Added (2026-04-10)
- 在 rollout 增加 `debug_policy_freeze_rotation` 开关（默认 `false`）。
- 开启后，policy 预测动作中的旋转通道将被替换为单位旋转，仅保留平移 + gripper。
- 作用：快速验证“旋转分支是否为 rollout 失败主因”。
- 变更文件：
   - `armada/env_runner/real_env_runner.py`
   - `armada/config/maniskill_rollout.yaml`
   - `armada/config/base_rollout.yaml`

## New Tooling (2026-04-10)
- 新增最小离线对齐测试脚本：`check_policy_dataset_alignment.py`
- 目标：复用 rollout 同款 checkpoint 加载路径（workspace + payload），在训练集观测窗口上调用 `policy.predict_action`，并与训练集动作做误差统计（MAE/MSE + pos/rot/gripper 分项）。
- 用途：快速判断“模型是否至少在训练分布观测上行为对齐”，优先隔离训练/表示问题，再进入在线 rollout 排障。
- 增强项：新增旋转 geodesic 角度误差（deg）与 gripper 二值一致率，避免仅用 rot6d L1 导致误判。

## Issue Log
- 问题：运行 `check_policy_dataset_alignment.py` 报错 `No module named 'diffusion_policy'`。
- 原因：脚本未像 `armada/train.py` 一样将 `armada/diffusion_policy` 加入模块搜索路径，Hydra 无法解析 `_target_=diffusion_policy.workspace.*`。
- 处理：在脚本启动时添加 `sys.path` 注入（`armada/` 与 `armada/diffusion_policy/`）。
- 状态：已修复，待用户重跑确认。

## Latest Run Result (2026-04-10)
- 命令：`python check_policy_dataset_alignment.py --training-config train_maniskill_poc --checkpoint ... --dataset-zarr ... --num-samples 8 --ta 8 --device-id 0`
- 状态：运行成功。
- 结果摘要：
   - MAE(all dims): 0.299824 +- 0.042980
   - MSE(all dims): 0.285687 +- 0.096714
   - MAE(pos xyz): 0.014055 +- 0.011618
   - MAE(rot6d): 0.470530 +- 0.076104
   - MAE(gripper): 0.132897 +- 0.313006
- 初步解读：
   - 平移对齐较好（pos 误差低），说明视觉到位移的主链路基本可用。
   - 旋转分量误差显著偏高（rot6d），是当前最主要偏差来源。
   - gripper 波动较大，可能存在二值开合边界或时序对齐问题。

## Latest Run Result (2026-04-10, 256 samples)
- MAE(all dims): 0.280528 +- 0.050688
- MSE(all dims): 0.237079 +- 0.071905
- MAE(pos xyz): 0.027432 +- 0.022914
- MAE(rot6d): 0.385078 +- 0.081393
- Rotation geodesic error (deg): 126.014334 +- 26.407679
- MAE(gripper): 0.412511 +- 0.463574
- Gripper binary accuracy: 0.592285 +- 0.485348
- 结论：平移分支可用；旋转与 gripper 分支明显不对齐。

## Additional Evidence
- 训练集 gripper 标签统计：仅有 0/1 两值，比例约 41.9% / 58.1%。
- 推断：当前 gripper 二值准确率 0.592 接近多数类基线，说明模型在 gripper 上几乎接近“猜多数类”。

## Convention Check
- `RotationTransformer` 使用 PyTorch3D quaternion 接口（默认实部在前，wxyz），与本项目动作处理链约定一致。
- 因此“纯 quaternion 顺序错误”不是首要嫌疑，优先排查旋转目标可学习性与 gripper 监督方式。

## New Hypothesis (2026-04-10)
- PickCube 任务中每条轨迹的 lift 目标位姿可能不同，但训练观测里未显式提供该目标位姿，造成部分时刻“同观测多动作”的不可观测多模态。
- 证据：
   - 数据采集保存字段仅含 `wrist_cam/side_cam/tcp_pose/joint_pos/action`，未包含目标位姿字段，见 `maniskill_armada/collect_data.py`。
   - 训练任务观测定义仅含图像与 `ee_pose`，未含 `goal_pos`，见 `armada/config/training/task/maniskill_pick.yaml`。
   - 启发式策略在生成 lift 目标时使用 `extra.goal_pos`，见 `maniskill_armada/heuristic_policy.py`。

## Implication
- 抓取后阶段出现动作随机、旋转误差大、gripper 接近多数类基线，和“缺失任务条件变量”一致。

## Priority Fix Path
1. 快速验证：收集/训练一个固定目标位姿版本（去掉多目标随机性），观察离线对齐与 rollout 是否显著提升。
2. 正式修复：把 `goal_pos` 写入数据并作为训练/rollout观测输入（conditioned policy）。
3. 次级优化：对 gripper 使用分类式评估与训练权重调节，避免回归均值化。

## Minimal Validation Implemented (2026-04-10)
- `maniskill_armada/collect_data.py` 新增：
   - `--fixed-reset-seed <int>`: 每个 episode 使用同一个 reset seed（固定初始/目标分布）
   - `--fixed-goal-pos x y z`: 启发式策略固定 lift 目标（策略层 override）
- `maniskill_armada/heuristic_policy.py` 新增 `fixed_goal_pos` 配置读取并在 `_get_target` 中优先生效。
- `ARMADA_MANISKILL_README.md` 已补充最小验证采集命令示例。

## Rollout Save Diagnostics Added (2026-04-10)
- 在 `real_env_runner.py` 增加日志：
   - 初始化时打印训练基底 episodes 数量。
   - `episode_data is None` 时显式提示“未追加轨迹”。
   - cleanup 时打印 `base_episodes / added_rollout_episodes / total_episodes`。
- 目的：快速区分“save_buffer 里只有训练基底”与“确实有新增 rollout 轨迹”。

## Timeout + Save Fixes (2026-04-10)
- `hardware/maniskill_robot_env.py`
   - 新增 `maniskill.max_episode_steps` 配置读取（默认 300），并传入 `gym.make(..., max_episode_steps=...)`。
   - 目的：避免环境在约 56 步附近过早 timeout（Ta=8 chunk 检查下会显示为 56）。
- `armada/env_runner/real_env_runner.py`
   - 回放写盘时，`wrist_cam/side_cam` 改为保存 `*_img_raw`（原始相机分辨率）而非 `demo_*` 预处理图。
   - 目的：与训练基底 replay buffer schema 对齐，避免 `_validate_episode_shapes` 因图像 shape 不一致而拒绝追加。

## Teleop Jitter Mitigation (2026-04-10)
- 观察：teleop 控制中机械臂抖动，怀疑累计不可达目标导致振荡。
- 修复：在 `hardware/maniskill_robot_env.py::human_teleop_step` 中加入测量反馈模式：每步后将 `last_p/last_r` 更新为仿真返回的真实 `tcp_pose`，而非上一步命令目标。
- 新配置：`maniskill.teleop_feedback_from_measured_pose`（默认 `true`）。
- 预期：显著降低不可达目标累积引起的姿态抖动，提高 teleop 可控性。

## Notes
- 颜色问题属于可视化层，不会影响训练数据本身的数值内容。
- “56 vs 150~180”若语义是 chunk 与 frame 的单位差异，则两者不冲突。
