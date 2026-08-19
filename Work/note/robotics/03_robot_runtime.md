# 03｜Robot Runtime：传感器同步与 ROS 抽象

## 本模块解决的问题

机器人端 Runtime 的核心难点不是"跑模型"，而是**异构传感器的数据同步**。本章回答：

```text
机器人 Runtime 的数据流是什么？（sensor → sync → model → action → controller）
不同频率的传感器数据怎么同步？
ROS 的 Topic / Service / Action / QoS 各是什么？
non-ROS runtime 什么时候用？
```

配套代码：`src/robotics/runtime/`（数据流 + ROS-like 原语）。

---

## 1. 机器人 Runtime 的数据流

```text
Sensor（相机 30Hz、IMU 200Hz、关节编码器 100Hz）
   ↓
Data Synchronization（把不同频率的数据对齐成"一个时刻"的观测）
   ↓
Model Runtime（VLA / Policy 推理）
   ↓
Action（动作向量）
   ↓
Controller（执行器控制，闭环）
```

和 Web Infra 的最大区别：**这是一条物理实时链路**——数据必须新鲜、一致、准时，否则机器人动作过时、危险（呼应 Stage 14）。

---

## 2. 核心难题：传感器频率不一致

相机 30Hz、IMU 200Hz、关节 100Hz——模型推理需要"某一时刻的完整观测"，但传感器永远不会在同一时刻同时出数。两种同步策略：

```text
latest  用每个传感器最新的读数（不等，但观测是"拼凑"的）
exact   等到所有传感器都有同一时刻的读数（数据一致，但要等）
```

### 本机实测（10s，20Hz 控制环，相机 30Hz + IMU 200Hz + 关节 100Hz）

```text
strategy  cycles  skips  staleness mean  max
latest    200     0      0.021s         0.033s
exact      19     181    0.000s         0.000s
```

### 读法（本模块的灵魂）

1. **latest：200 个控制周期全跑，但观测陈旧 21-33ms**。因为相机（30Hz）每 33ms 才更新一次，模型拿到的"当前观测"里，相机数据可能已经 33ms 旧了。

2. **exact：只有 19 个周期有完全对齐的观测，181 个跳过**。因为 30Hz/200Hz/100Hz 三个频率几乎永远不同步，等"完全对齐"会让控制频率从 20Hz 暴跌到 ~2Hz。

**这是机器人 Infra 的经典权衡**：控制频率 vs 数据一致性。工业上的解法：

```text
ApproximateTimeSynchronizer：在容差（如 5ms）内对齐，近似一致 + 高频率
latest + 插值/预测：用最新数据 + 对低频传感器做插值
按传感器重要性分级：IMU（高频）用 latest，相机（低频）用缓存的最新帧
```

没有银弹，只有按任务的实时性要求选策略。

---

## 3. ROS 概念（本机无 ROS，用 ROS-like 原语讲解）

| 概念 | 语义 | 用途 |
|---|---|---|
| Topic | pub/sub，多对多 | 传感器数据流（相机帧、IMU） |
| Service | 请求-响应（一次） | 短操作（复位、查询状态） |
| Action | 长时间任务 + 反馈 + 可取消 | 导航、抓取（有目标/进度/结果） |
| QoS | 可靠 vs 尽力、历史深度 | 控制实时性 vs 数据完整性 |

本模块 `ros_like.py` 用 Python 实现了这四个抽象（无需装 ROS），演示语义：

```text
topic   ：发布 5 帧，订阅者收到 5 帧
service ：reset arm -> "reset arm done"
action  ：navigate 目标，5 步反馈，最终 done（支持取消）
```

### QoS 的含义

```text
reliable   消息一定送达（可能重发）—— 用于关键命令
best_effort 消息可能丢（不重发）    —— 用于高频传感器（丢了就丢了，下一帧更新）
history depth 保留多少历史消息      —— 消费者可回溯
```

机器人用 best_effort + 小 depth 传高频传感器（省资源），用 reliable 传命令（不丢）。

---

## 4. non-ROS runtime

ROS 适合研究、原型、多模块协作。但**量产机器人**常用 non-ROS runtime，原因：

```text
1. ROS 的调度开销不适合极致实时（本模块的 jitter 问题）
2. ROS 的依赖复杂，量产部署重
3. 自己的 runtime 可以定制实时调度、安全边界
```

non-ROS runtime 的设计要点（本模块的 `RobotRuntime` 就是雏形）：

```text
1. 明确的数据流（sensor → sync → model → action → controller）
2. 显式的同步策略（latest / exact / 近似）
3. 安全边界（动作超时、越界保护、紧急停止）
4. 实时调度（优先级、deadline）
```

**结论：ROS 是工具，不是必须。理解数据流和同步的本质，比会用 ROS 更重要**——这是"机器人 Infra 和传统 Web Infra 最大区别（物理实时性和安全边界）"的落脚点。

---

## 5. 本模块闭环小结

```text
问题：机器人端 Runtime 怎么组织异构传感器 + 模型 + 控制
      ↓
数据流：sensor → sync → model → action → controller
      ↓
难题：传感器频率不一致，latest（快但旧）vs exact（新但慢）
      ↓
ROS：Topic/Service/Action/QoS 是抽象；non-ROS 是量产选择
      ↓
下一步：Stage 26 实时性（Latency/Jitter/Deadline/优先级，已有 Stage 14 铺垫）
```

要继续就说「继续」。
