# 01｜可观测性：Metrics / Logs / Traces 三大体系

## 本模块解决的问题

规模化后，"用户看到 200ms 延迟"必须能回答"这 200ms 发生在哪"。本章回答：

```text
Metrics / Logs / Traces 各自解决什么问题？
Request ID / Task ID / Robot ID 怎么贯穿 Cloud → Edge → Robot？
怎么用 trace 定位慢请求的瓶颈层？
```

配套代码：`src/observability/`（三体系原语 + 跨层 trace 演示）。

---

## 1. 三大体系的分工

| 体系 | 回答什么 | 例子 |
|---|---|---|
| Metrics | "整体怎么样"（聚合数值） | QPS、latency p99、error rate、GPU util、温度 |
| Logs | "发生了什么"（离散事件） | 错误、警告、状态变化 |
| Traces | "一次请求走了哪些层"（链路） | 请求跨 Cloud→Edge→Robot 的耗时 |

三者互补，缺一不可：

```text
Metrics 告诉你"有问题"（p99 突然升高）
Traces 告诉你"问题在哪一层"（某个请求的 model_infer 慢）
Logs 告诉你"为什么会这样"（那台 robot 的模型加载失败）
```

---

## 2. Request ID / Task ID / Robot ID 的贯穿

这是分布式追踪的核心：**每个请求、任务、机器人都有唯一 ID，且逐层传递**。

```text
request_id（一次请求）
   ├─ task_id（关联的任务）
   └─ robot_id（执行的机器人）
         ↓ 每个 span 都携带这些 ID
```

有了贯穿 ID，就能从任意一层反查整个链路：

```text
"robot_0 的某个动作失败" → 查 robot_0 的日志 → 拿到 request_id
   → 用 request_id 查 trace → 看到完整的 Cloud→Edge→Robot 链路
```

**没有贯穿 ID，分布式系统的调试就是大海捞针**——你只知道"某台机器某个时间报了个错"，无法关联到"是哪次请求、哪个任务、哪个用户触发的"。

---

## 3. 实测：用 trace 定位慢请求

模拟 200 个请求（每 50 个有一个慢请求），记录三体系：

```text
Metrics：
  latency mean=5.51ms  p50=5.16ms  p99=22.17ms
  error_rate = 2.0%

慢请求 req_50 的 trace：
  cloud.schedule       1.05ms  (task_id=task_10)
  edge.forward         1.05ms  (robot_id=robot_0)
  robot.model_infer   20.06ms  (model_version=v2)   ← 瓶颈
```

### 读法

1. **Metrics 先报警**：p99（22ms）远大于 p50（5ms），说明有慢请求拖尾。

2. **Trace 定位瓶颈**：取一个慢请求（req_50）的 trace，看到 22ms 里 20ms 在 `robot.model_infer`，cloud/edge 各只 1ms。**瓶颈在模型推理，不在调度/转发**。

3. **Logs 提供上下文**：慢请求的 WARN 日志带了 `robot_id` 和 `model_version`，能进一步查"是哪台机器人、哪个模型版本慢"。

这就是"用户看到 200ms 延迟"的完整定位链路：**Metrics 发现 → Trace 定位 → Logs 归因**。

---

## 4. 机器人系统的监控清单（master prompt 要求）

| 指标 | 体系 | 用途 |
|---|---|---|
| request QPS / latency / error rate | Metrics | 服务健康 |
| GPU utilization / memory / temperature | Metrics | 资源 + 散热（Stage 15） |
| model version / robot version | 维度标签 | 灰度/回滚追踪 |
| task success rate | Metrics | 业务指标（Stage 21/27） |
| 每层耗时（span） | Traces | 定位瓶颈 |
| 错误事件（带 ID） | Logs | 归因 |

**关键：所有指标都要带 robot_id / model_version 维度标签**，否则无法区分"是某个机器人坏了，还是整体变慢"。

---

## 5. 本模块闭环小结

```text
问题：200ms 延迟发生在哪一层
      ↓
三体系：Metrics（发现）+ Traces（定位）+ Logs（归因）
      ↓
贯穿：Request/Task/Robot ID 逐层传递，任意层可反查全链路
      ↓
实测：p99 22ms，trace 定位到 model_infer 20ms
      ↓
下一步：Stage 29 可靠性（故障注入 + Watchdog + 恢复）
```

要继续就说「继续」。
