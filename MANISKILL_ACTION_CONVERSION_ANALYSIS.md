# ManiSkill 环境下 action 转换链条梳理（_run_policy_inference_loop）

日期：2026-04-29

本文件目标：把 `RealEnvRunner._run_policy_inference_loop` 在 `env_backend=maniskill` 时，policy 输出如何一步步变成 ManiSkill `pd_ee_delta_pose` 执行动作，**用统一符号/公式**写清楚：
- 每一步的参考系（world/base/TCP）
- 旋转的表示法（wxyz vs xyzw，quat vs rotvec）
- 位置/旋转是“绝对目标”还是“相对增量”

并标出最可能导致“执行结果与预期不一致”的转换错误点。

---

## 1. 统一符号与约定

### 1.1 参考系
- $W$：世界坐标系（ManiSkill 观测 `extra.tcp_pose` 默认就是 world pose）。
- $E$：TCP/末端执行器坐标系。
- $B$：机器人 base/root（在 PickCube 这类任务里 base 通常固定在 world，常见情况下 $B\approx W$；但分析中仍保留符号区分）。

### 1.2 位姿表示
- 位置：$p\in\mathbb{R}^3$。
- 旋转：用旋转矩阵 $R\in SO(3)$ 或四元数 $q$。

四元数有两种常见排列：
- **wxyz（scalar-first）**：$q^{wxyz}=[w,x,y,z]$。
- **xyzw（scalar-last）**：$q^{xyzw}=[x,y,z,w]$。

在本 repo 里：
- `scipy.spatial.transform.Rotation.from_quat(..., scalar_first=True)` 期望 **wxyz**。
- `scipy.spatial.transform.Rotation.from_quat(q_xyzw)`（不带 `scalar_first`）期望 **xyzw**。
- `RotationTransformer`（PyTorch3D）默认使用 **wxyz**。

定义重排算子：
- $\pi_{wxyz\to xyzw}([w,x,y,z])=[x,y,z,w]$
- $\pi_{xyzw\to wxyz}([x,y,z,w])=[w,x,y,z]$

### 1.3 “空间固定（world/space）”与“体固定（TCP/body）”旋转增量
给定当前旋转 $R$ 与目标旋转 $R^*$，存在两种常用的相对旋转定义：

1) **体固定（body-fixed，右乘）**
$$
\Delta R_{body} = R^{-1}R^* \quad\Rightarrow\quad R^*=R\,\Delta R_{body}
$$

2) **空间固定（space-fixed，左乘）**
$$
\Delta R_{space} = R^*R^{-1} \quad\Rightarrow\quad R^*=\Delta R_{space}\,R
$$

注意两者一般不相等，关系为共轭：
$$
\Delta R_{space} = R\,\Delta R_{body}\,R^{-1}
$$

这点是本次链条里最容易出错的地方。

---

## 2. 已知：policy.predict_action 与 heuristic_policy.get_action 的一致性

`maniskill_armada/heuristic_policy.py` 中，旋转增量的计算为：

- 取当前 TCP 旋转 $R$（来自 `obs.extra.tcp_pose`）
- 取目标 TCP 旋转 $R^*$
- 计算
$$
\Delta R_{policy} = R^{-1}R^*
$$
并输出对应四元数（wxyz）或其 rotvec。

也就是说，**heuristic policy 输出的是 body-fixed（右乘）旋转增量**。

用户说明：`policy.predict_action` 的输出与 `heuristic_policy.get_action` 一致。于是我们把“policy 输出旋转增量”的语义也视为：
$$
\Delta R_{policy} \equiv \Delta R_{body}
$$

---

## 3. 调用链总览（ManiSkill 单环境）

关键调用链（单 env）：

1) `RealEnvRunner._run_policy_inference_loop`
2) `EpisodeManager.get_absolute_action_for_step`（把 policy action 序列“积分”为绝对 TCP pose）
3) `ManiSkillRobotEnv.deploy_action`（把绝对 TCP pose 再转回 ManiSkill 的 `pd_ee_delta_pose` action 并 `env.step`）

从“类型”上看，链条是：

- policy 输出：**相对增量（delta）动作序列**
- EpisodeManager：把 delta **积分**成 **绝对目标 pose**
- ManiSkillRobotEnv：把绝对目标 pose 再 **差分**回 **delta action**（`pd_ee_delta_pose`）

因此只要其中任一处对“旋转增量是左乘还是右乘”的假设不一致，就会产生可观偏差。

---

## 4. 逐步公式化：_run_policy_inference_loop 的 action 转换

下面以单步 $k\to k+1$（对应 chunk 内的某个 `step`）为例；并把 `num_samples` 的 batch 维度先忽略，只写单样本。

### 4.1 policy 输出（action_seq 的语义）

`policy.predict_action(policy_obs)` 返回字典，其中 `action` 被取出为：
$$
A\in\mathbb{R}^{N\times T_a\times D}
$$
随后 `action_seq = A`。

对某一步 $s$，记 policy 原始输出为：
$$
a_s = [\Delta p_s,\; r_s,\; g_s]
$$
其中：
- $\Delta p_s\in\mathbb{R}^3$：位置增量（代码里直接做 `p + dp`，因此隐含假设是 **在 world/base 坐标系下的平移增量**）。
- $r_s\in\mathbb{R}^{d_r}$：旋转增量的某种表示（可能是 quaternion/6D/axis-angle 等），由 `RotationTransformer` 反解回 quaternion。
- $g_s\in\mathbb{R}$：夹爪通道（语义需额外确认，见第 6 节）。

### 4.2 EpisodeManager：delta -> 绝对目标 pose

对应 `armada/utils/episode_manager.py:get_absolute_action_for_step`。

内部状态：维护“上一次的目标”
- $p_k^{cmd}$：上一次目标位置（world/base）
- $R_k^{cmd}$：上一次目标旋转

对 step $s$：

**(1) 平移积分（world/base 增量）**
$$
p_{k+1}^{cmd} = p_k^{cmd} + \Delta p_s
$$

**(2) 旋转积分（代码实现为右乘）**

先把旋转通道反解成 **wxyz 四元数** 并转旋转矩阵：
$$
\Delta q_s^{wxyz}=\mathrm{RotTf}^{-1}(r_s),\qquad \Delta R_s = R(\Delta q_s^{wxyz})
$$
然后代码做：
$$
R_{k+1}^{cmd} = R_k^{cmd}\,\Delta R_s
$$
这是 **body-fixed（右乘）** 更新。

**(3) 输出绝对目标 pose（用于 deploy）**

`deployed_action`（传给 `deploy_action`）为：
$$
\text{tcp\_target}^{wxyz} = [p_{k+1}^{cmd},\; q(R_{k+1}^{cmd})^{wxyz}]
$$

结论：EpisodeManager 假设 policy 输出的旋转增量是
$$
\Delta R_s = (R_k^{cmd})^{-1}R_{k+1}^{cmd}
$$
即 **body-fixed / 右乘**。

这与第 2 节（heuristic policy）一致。

### 4.3 ManiSkillRobotEnv.deploy_action：绝对目标 pose -> pd_ee_delta_pose

对应 `hardware/maniskill_robot_env.py:deploy_action`，其核心是 `_abs_target_to_delta_action`。

它首先从观测中取当前测得 TCP pose（wxyz）：
$$
\text{tcp\_meas}^{wxyz}=[p_k^{meas},\;q(R_k^{meas})^{wxyz}]
$$

然后将 wxyz 改为 xyzw，调用：

`maniskill_armada.data_utils.extract_action_from_poses(current_pose_xyzw, target_pose_xyzw, ...)`

注意：该函数内部对旋转差分定义为：
$$
\Delta R_{data\_utils} = R^*\,(R)^{-1}
$$
即
$$
\Delta R_{data\_utils} = R_{k+1}^{cmd}\,(R_k^{meas})^{-1}
$$
这是 **space-fixed（左乘）** 的相对旋转。

随后 `deploy_action` 将其转换为 rotvec（轴角向量）：
$$
\Delta \omega = \mathrm{Log}(\Delta R_{data\_utils})\in\mathbb{R}^3
$$
最终给 ManiSkill env 的动作（单 env）为：
$$
a^{ms} = [\Delta p,\; \Delta\omega,\; g]\in\mathbb{R}^7
$$
其中
- 平移差分：$\Delta p = p_{k+1}^{cmd}-p_k^{meas}$（world/base）
- 旋转差分：$\Delta\omega = \mathrm{Log}(R_{k+1}^{cmd}(R_k^{meas})^{-1})$
- 夹爪 $g$：由 `gripper_action` 经 `gripper_max_width` 线性归一化得到（见第 6 节）。

---

## 5. 核心一致性检查：为什么会“不一致”？

本链条中出现了一个关键事实：

- EpisodeManager 用的是 **body-fixed** 更新：$R_{k+1}^{cmd}=R_k^{cmd}\,\Delta R_s$。
- `extract_action_from_poses` 计算的是 **space-fixed** 差分：$\Delta R_{space}=R_{k+1}^{cmd}(R_k^{meas})^{-1}$。

接下来是否会错，取决于 ManiSkill `pd_ee_delta_pose` 控制器**到底把 rotvec 当成左乘还是右乘增量**。

### 5.1 如果 ManiSkill 控制器是 body-fixed（右乘）语义
即执行近似满足：
$$
R_{k+1}^{meas}\approx R_k^{meas}\,\mathrm{Exp}(\Delta\omega)
$$

那么为了达到目标 $R_{k+1}^{cmd}$，应该给：
$$
\mathrm{Exp}(\Delta\omega) = (R_k^{meas})^{-1}R_{k+1}^{cmd}
$$
也就是 **body-fixed 相对旋转**。

但当前实现给的是：
$$
\mathrm{Exp}(\Delta\omega) = R_{k+1}^{cmd}(R_k^{meas})^{-1}
$$
这等价于把 body-fixed 增量共轭了一次：
$$
R_{k+1}^{cmd}(R_k^{meas})^{-1} = R_k^{meas}\,((R_k^{meas})^{-1}R_{k+1}^{cmd})\,(R_k^{meas})^{-1}
$$

于是执行结果会变成：
$$
R_{k+1}^{meas} \approx R_k^{meas}\,\big(R_{k+1}^{cmd}(R_k^{meas})^{-1}\big)
\neq R_{k+1}^{cmd}
$$
除非 $R_k^{meas}$ 与增量可交换（一般不成立）。

**这会直接导致“旋转执行方向/幅度与预期不一致”。**

### 5.2 如果 ManiSkill 控制器是 space-fixed（左乘）语义
即执行近似满足：
$$
R_{k+1}^{meas}\approx \mathrm{Exp}(\Delta\omega)\,R_k^{meas}
$$

那么给当前实现的
$$
\mathrm{Exp}(\Delta\omega)=R_{k+1}^{cmd}(R_k^{meas})^{-1}
$$
会刚好得到：
$$
R_{k+1}^{meas}\approx R_{k+1}^{cmd}
$$

因此：
- **若 ManiSkill 是左乘语义，则当前 deploy_action 的旋转差分是对的**；此时若还不一致，问题更可能在 EpisodeManager（把 policy 输出解释错了）或 gripper 通道。
- **若 ManiSkill 是右乘语义，则当前 deploy_action 的旋转差分是错的**；这与“heuristic policy 使用右乘增量”的事实也更一致。

### 5.3 结合“policy 与 heuristic 一致”的推断
由于 heuristic policy 明确使用
$$
\Delta R_{policy}=R^{-1}R^*
$$
并且通常我们期望把 rotvec 直接当作 body-fixed 增量送入 `pd_ee_delta_pose`，所以更自然的结论是：

> ManiSkill `pd_ee_delta_pose` 在本任务配置下更可能是 **body-fixed（右乘）** 语义。

若这个推断成立，则当前链条最可疑的错误点就是：

**`extract_action_from_poses` 的旋转差分公式用了左乘（space-fixed），应该改成右乘（body-fixed）。**

---

## 6. 第二个高风险点：夹爪通道语义/归一化

`ManiSkillRobotEnv._abs_target_to_delta_action` 对夹爪的处理是：

- 假设输入 `gripper_action` 是“夹爪开口宽度”（单位米，范围 $[0,\text{gripper\_max\_width}]$）。
- 归一化到 ManiSkill 动作空间（近似 $[-1,1]$）
$$
 g = \mathrm{clip}\Big(2\frac{\text{gripper\_action}}{\text{gripper\_max\_width}}-1,\,-1,\,1\Big)
$$

这意味着：
- `gripper_action = 0` 会得到 $g=-1$（更像“关闭”）
- `gripper_action = gripper_max_width` 会得到 $g=+1$（更像“打开”）

而 `heuristic_policy.to_maniskill_action` 的注释与实现是：
- policy/hybrid 8D action 最后一维 `gripper_target` 采用 **0=open, 1=close**
- ManiSkill gripper 命令采用 **+1=open, -1=close**
- 映射为：
$$
 g = 1 - 2\,\text{gripper\_target}
$$

两者若直接混用，会出现**符号反转**：
- 若 policy 输出 `gripper_target∈{0,1}`（0=open, 1=close），那么直接塞给 `deploy_action`：
  - `0 -> g=-1`（被解释为 close）
  - `1 -> g≈+1`（被解释为 open，且会因为除以 0.09 被强烈饱和）

这会导致非常典型的现象：**policy 想关夹爪却开、想开却关，且动作幅度被不合理放大/饱和。**

因此在排查“不一致”时，建议首先确认：
- `policy.predict_action` 的 gripper 通道到底是“宽度（米）”还是“0/1 目标”还是“[-1,1] 命令”？
- 训练数据里 gripper 通道的定义是否与 deploy 时一致？

---

## 7. 最小化的“自洽”修复方向（按假设分支）

下面给出两条互斥的修复方向；选哪条取决于你确认 ManiSkill 控制器的语义，以及 policy 训练数据的定义。

### 7.1 方向 A（更符合 heuristic）：把旋转差分改成 body-fixed（右乘）

把 `extract_action_from_poses` 的相对旋转从
$$
\Delta R = R^*R^{-1}
$$
改为
$$
\Delta R = R^{-1}R^*
$$

对应代码层面就是把：
- `rel_rot = tgt_rot * curr_rot.inv()`
改为：
- `rel_rot = curr_rot.inv() * tgt_rot`

这会同时影响：
- `EpisodeManager.get_absolute_action_for_step` 里用于记录的 `converted_dq`
- `ManiSkillRobotEnv.deploy_action` 里用于执行的 rotvec

若 heuristic 的旋转/夹爪在 ManiSkill 中表现正确，这个方向通常是最一致的。

### 7.2 方向 B：坚持 space-fixed（左乘），则 EpisodeManager 必须用左乘积分

如果你确认 ManiSkill 控制器就是左乘语义、且训练数据也是 space-fixed，那么 EpisodeManager 里旋转积分应改为：
$$
R_{k+1}^{cmd} = \Delta R_s\,R_k^{cmd}
$$
而不是 $R_k^{cmd}\,\Delta R_s$。

### 7.3 方向 C：避免“delta→absolute→delta”的往返

在 ManiSkill 仿真里，其实可以直接把 policy 的 delta action 转成 `pd_ee_delta_pose` 动作：
- $dp$ 直接用
- $dq^{wxyz}$ 直接转 rotvec：$d\omega=\mathrm{Log}(R(dq))$
- gripper 通道按约定映射到 $[-1,1]$

这样能显著减少“语义不一致”的机会。

---

## 8. 建议的快速一致性诊断（不用跑很长）

对任意一步，拿到当前测得旋转 $R_k^{meas}$ 与 EpisodeManager 产生的目标 $R_{k+1}^{cmd}$，计算两种 delta：

- body-fixed：$\Delta R_{body}=(R_k^{meas})^{-1}R_{k+1}^{cmd}$
- space-fixed：$\Delta R_{space}=R_{k+1}^{cmd}(R_k^{meas})^{-1}$

把它们各自按“右乘/左乘”应用回去，看哪一个能复原目标：

- 检查 1（右乘复原）：$R_k^{meas}\,\Delta R_{?} \stackrel{?}{=} R_{k+1}^{cmd}$
- 检查 2（左乘复原）：$\Delta R_{?}\,R_k^{meas} \stackrel{?}{=} R_{k+1}^{cmd}$

哪个成立，就说明你的控制器语义更接近哪个。

同理，夹爪通道可以打印三元组：
- `policy_gripper`（原始输出）
- `deploy_action` 里最终送入 env 的 `g`
- 环境反馈的实际夹爪宽度/开合状态（如果 obs/info 可取）

---

## 9. 本次梳理结论（最可疑点）

基于“policy 与 heuristic 一致（heuristic 使用 body-fixed 右乘增量）”这一前提，本链条里最可疑的两个问题是：

1) **旋转 delta 的左/右乘语义不一致**
- EpisodeManager/heuristic 使用：$R_{new}=R\,\Delta R$（body-fixed）
- `extract_action_from_poses`/`deploy_action` 使用：$\Delta R=R_{tgt}R^{-1}$（space-fixed）
- 若 ManiSkill 控制器期望 body-fixed，则会导致执行旋转与目标不一致（共轭误差）。

2) **夹爪通道的单位/符号约定可能不一致**
- `deploy_action` 假设输入是“宽度（米）”并做 $2\frac{w}{w_{max}}-1$
- heuristic/policy 可能输出的是“0=open,1=close”目标
- 直接传递会导致反转或饱和。

---

### 相关代码位置（便于对照）
- `armada/env_runner/real_env_runner.py:_run_policy_inference_loop`
- `armada/utils/episode_manager.py:get_absolute_action_for_step`
- `hardware/maniskill_robot_env.py:deploy_action` 与 `_abs_target_to_delta_action`
- `maniskill_armada/data_utils.py:extract_action_from_poses`
- `maniskill_armada/heuristic_policy.py:_compute_delta_action` 与 `to_maniskill_action`
