# 02｜Inference Server：从串行到 Dynamic Batching

## 本模块解决的问题

自建一个推理服务，理解"请求 → 队列 → 模型 → GPU"这条链的调度核心。本章回答：

```text
为什么串行处理请求的吞吐这么低？
static batching 和 dynamic batching 差在哪？
为什么 static batching 会在尾部卡住？
queue 和 timeout 各自解决什么问题？
```

配套代码：`src/serving/inference_server/`（`server.py` 队列 + worker + batching + `benchmark.py`）。

---

## 1. 一个推理服务的最小结构

```text
HTTP/RPC 请求
   ↓
请求队列（有界，满则拒绝）
   ↓
Worker 线程（攒 batch → 推理 → 回传结果）
   ↓
模型（GPU）
```

本模块用**线程 + 队列**实现这个核心（HTTP 层在生产用 FastAPI/gRPC 包一层，核心调度逻辑不变）。三个组件：

| 组件 | 作用 | 本模块实现 |
|---|---|---|
| request queue | 缓冲请求，削峰 | `queue.Queue(maxsize=...)` |
| batching 策略 | 攒多少请求一起推理 | no_batch / static / dynamic |
| worker | 从队列取 batch，跑模型 | 单线程循环 |

---

## 2. 三种 batching 策略

```text
no_batch ：每个请求单独推理（batch=1），GPU 利用率最低
static   ：攒满 max_batch 个请求才推理
dynamic  ：攒 batch，但超时（max_wait）也 flush（不空等）
```

### 本机实测（500 请求，并发 16，hidden=512 MLP）

```text
strategy   throughput     mean_ms   p50_ms   p99_ms
no_batch   439/s          35.96     33.17    128.58
static      49/s           4.90      4.84     17.11
dynamic    3130/s          4.80      4.83      7.27
```

### 读法（三个关键洞察）

1. **no_batch 吞吐最低（439/s）**：每个请求 batch=1 单独推理，GPU 的 GEMM 小、利用率低，大部分时间浪费在 launch 和低效的小 GEMM 上（呼应 Stage 14 的 batch=1 问题）。

2. **dynamic 吞吐提升 7 倍（3130/s），且 p99 只有 7.3ms**：batch=8 一起推理，GPU 利用率高；`max_wait` 超时 flush 保证请求不会空等。**这是 throughput 和 latency 的最佳平衡点**。

3. **static 吞吐反而暴跌到 49/s——这不是"慢"，是"卡死"**：static 要等满 batch=8 才推理。500 个请求 = 62 个满 batch（496 个）+ 尾部 4 个请求**永远凑不满第 8 个**，worker 一直阻塞等待，那 4 个请求直到客户端超时（10s）才放弃。所以 static 的 elapsed 包含了 10 秒的空等。

---

## 3. 为什么 static 会卡死，dynamic 不会

这是本模块最重要的洞察：

```text
static = 等满 batch（无超时）→ 尾部请求凑不满 → 卡死
dynamic = static + max_wait flush → 凑不满也超时处理 → 不卡
```

**dynamic batching 就是"加了超时 flush 的 static batching"**。任何真实系统里，纯 static（无超时）都会在请求速率低、或流量尾部（最后一波请求不满）时卡住。所以：

- **timeout 不是可选项，是 batching 的必需部分**。
- Stage 12 的 continuous batching 是 dynamic batching 在 LLM 场景的进一步演化（decode 阶段请求动态增删）。

---

## 4. queue 和 timeout 的角色

| 机制 | 解决什么 | 本模块 |
|---|---|---|
| queue | 削峰：请求突发时缓冲，而不是直接打 GPU | `max_queue=128` |
| timeout | 等待上限：请求不能无限等 batch | `infer(timeout=...)` |
| 有界队列 + 拒绝 | 过载保护：队列满直接拒绝（load shedding 雏形） | `queue.Full → RuntimeError` |

这三个机制是 Stage 17（Production Inference Service）的基础：backpressure、rate limit、load shedding 都在这里生根。

---

## 5. 为什么 GPU 100% 时不能无限接请求（Stage 17 预告）

本模块已经埋下答案：

```text
GPU 100% 时，请求进队列，队列会满。
队列满了继续接请求 → 要么无限堆积（内存爆、延迟爆炸），要么拒绝。
所以"GPU 100% 就不能无限接"——必须有队列上限 + 拒绝策略。
```

完整展开（backpressure、circuit breaker、autoscaling）在 Stage 17。

---

## 6. 本模块闭环小结

```text
问题：推理服务怎么调度请求才能吞吐和延迟兼顾
      ↓
结构：queue + worker + batching
      ↓
实测：no_batch 439/s → dynamic 3130/s（7x），static 卡死在尾部
      ↓
结论：dynamic batching = static + timeout flush，是生产默认选择
      ↓
下一步：Stage 17 Production Inference Service（backpressure/rate limit/
      circuit breaker/load shedding/autoscaling）
```

要继续就说「继续」。
