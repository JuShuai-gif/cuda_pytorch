# 02｜分布式系统基础：投递语义与幂等性

## 本模块解决的问题

机器人远程控制和任务系统是**分布式系统**：云端、边缘、机器人之间通过不可靠的网络通信。本章回答：

```text
RPC / MQ / Cache / 锁等分布式组件各自解决什么？
at-most-once / at-least-once / exactly-once 差在哪？
为什么幂等性对机器人远程控制至关重要？
```

配套代码：`src/robotics/distributed/`（投递语义模拟 + 幂等 executor）。

---

## 1. 分布式组件清单（理论）

| 组件 | 解决什么 | 机器人场景举例 |
|---|---|---|
| RPC | 服务间调用 | 云端 → 机器人下发任务 |
| Message Queue | 削峰、解耦、异步 | 任务队列、数据上传 |
| Cache | 读多写少的热数据 | 模型权重缓存、状态缓存 |
| Database | 持久化 | 任务记录、机器人状态 |
| Object Storage | 大文件 | 模型文件、日志、视频 |
| Service Discovery | 服务寻址 | 机器人发现模型服务 |
| Distributed Lock | 并发互斥 | 避免重复下发任务 |

这些是"零件"，本章聚焦其中最关键的**通信语义**——因为它是所有零件的底层，也是机器人场景最容易出错的地方。

---

## 2. 三种投递语义

网络不可靠（丢包、重发），消息投递有三种保证：

```text
at-most-once   发送一次，可能丢，但绝不重复
at-least-once  重试直到确认，可能重复，但绝不丢
exactly-once   恰好一次（理想，单靠不可靠链路无法实现）
```

**exactly-once 的真相**：在真实分布式系统里，exactly-once 无法仅靠网络协议实现（发送方无法知道"接收方是否处理了"）。工业上的做法是：

```text
exactly-once ≈ at-least-once（不丢）+ 幂等（重复不重复执行）
```

即**用 at-least-once 保证不丢，用幂等消除重复**。

---

## 3. 实测：三种策略下机器人的最终位置

场景：云端发 100 个"前进 1 米"命令，链路丢包 20%。

```text
strategy                  deliveries   robot position   expected
at_most_once              90            90m             100m
at_least_once             114          114m             100m
at_least_once+idempotent  114          100m             100m
```

### 读法（本模块的灵魂）

1. **at-most-once：机器人少走 10m**（90m vs 100m）。丢了 10 个命令，任务不完整。

2. **at-least-once：机器人多走 14m（114m）**。重试导致 14 个命令重复执行。**对机器人这是灾难**——"前进 1 米"重复执行，机器人多走了 14 米，可能撞墙、撞人、冲出工作区。

3. **at-least-once + 幂等：恰好 100m（正确）**。重试仍然发生（deliveries=114），但幂等 executor 记录了已执行的 command_id，重复的命令被忽略。

**这就是幂等性对机器人远程控制至关重要的原因**：网络重试不可避免（at-least-once），但机器人的**动作不能重复执行**。幂等是"不丢 + 不重"的唯一解。

---

## 4. 幂等性的实现

```python
class RobotExecutor:
    def __init__(self, idempotent=True):
        self.executed_ids = set()  # 已执行的命令 ID
        self.position = 0

    def apply(self, command):
        if self.idempotent and command.id in self.executed_ids:
            return False  # 重复 -> 忽略
        self.executed_ids.add(command.id)
        self.position += 1.0
        return True
```

关键：**每个命令带唯一 ID**（command_id / request_id），执行方记录已处理的 ID。这是"幂等"的通用实现——无论 RPC、MQ、任务系统，都靠"唯一 ID + 去重"。

### 幂等的边界

- 幂等 ID 的**持久化**：机器人重启后，`executed_ids` 要能恢复（否则重启后重复命令又会被执行）。所以 executed_ids 要存数据库/磁盘，不能只在内存。
- 幂等的**粒度**：一次任务 vs 一个动作。通常命令 ID 对应"一次动作"，任务级幂等要更粗的 ID。
- 幂等的**时效**：ID 不能无限累积（清理过期 ID），否则内存/存储爆炸。

---

## 5. 与机器人任务的结合

```text
云端任务系统：
  下发"任务 123：去 A 点抓取" → 任务带 task_id
  ↓ 网络重试
机器人：
  收到 task_id=123 → 检查是否已执行 → 是则忽略，否则执行
  ↓ 执行结果回传（带 task_id）
云端：
  收到 task_id=123 的结果 → 幂等地更新任务状态
```

整个任务系统的可靠性，都建立在"唯一 ID + 幂等"之上。这是 Stage 23（云边架构）和 Stage 24（OTA）的地基。

---

## 6. 本模块闭环小结

```text
问题：不可靠网络上，机器人命令如何"不丢又不重"
      ↓
语义：at-most-once（丢）/ at-least-once（重）/ exactly-once（难）
      ↓
实测：at-most 90m，at-least 114m（危险），at-least+幂等 100m（正确）
      ↓
结论：exactly-once ≈ at-least-once + 幂等；幂等靠唯一 ID + 去重
      ↓
下一步：Stage 23 云边架构（Cloud → Edge → Robot 分层 + 任务/模型/数据）
```

要继续就说「继续」。
