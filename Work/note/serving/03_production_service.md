# 03｜Production Inference Service：过载保护与可靠性机制

## 本模块解决的问题

Stage 16 已经建了"队列 + worker + batching"。本章把它升级为生产级，回答 master prompt 的核心问题：

```text
GPU 已经 100% 时，为什么不能继续无限接请求？
rate limit / circuit breaker / load shedding 各保护什么？
三层防线怎么从外到内协同？
```

配套代码：`src/serving/production_service/`（三个可靠性原语 + 过载模拟）。

---

## 1. 多层架构

```text
客户端
   ↓
API Gateway ── rate limit（令牌桶，按速率拒绝）
   ↓
Scheduler ── circuit breaker（下游失败熔断）+ load shedding（队列满丢弃）
   ↓
Inference Worker ── 模型（GPU）
```

三个机制对应三层防线：

| 机制 | 层 | 保护谁 | 拒绝什么 |
|---|---|---|---|
| rate limit | 网关 | 下游（GPU） | 超速率请求 |
| load shedding | 队列 | 自己（服务） | 超容量请求 |
| circuit breaker | 调用 | 调用方 | 对失败下游的调用 |

---

## 2. 为什么 GPU 100% 不能无限接请求

**答案：因为延迟会爆炸到不可接受。**

本模块实测（10000 请求 2 秒突发 vs 1000 req/s 的 GPU）：

```text
无保护          admitted=10000  dropped=0     p50=5.0s  p99=9.9s
load shedding   admitted=102    dropped=9898  p50=0.05s  p99=0.10s
```

**无保护时，p99 = 9.9 秒**：GPU 每秒处理 1000 个，10000 个请求堆积，最后一个要等 10 秒。所有请求的延迟都爆炸了——不是"慢"，是"雪崩"。而且队列无限增长还会耗尽内存（OOM）。

**load shedding 时，p99 = 0.1 秒**：队列容量 100，满了就丢弃。丢弃 98.98% 的请求，但保留下来的请求延迟稳定。这就是权衡：

```text
无限接 → 全部请求都等死（延迟爆炸 + 内存爆）
丢弃   → 大部分请求快速失败，小部分请求得到正常服务
```

生产系统的原则：**宁可快速拒绝，不要慢慢堆积**。一个快速返回的 429/503 比一个 10 秒后才超时的请求好得多。

---

## 3. 三个机制的细节

### rate limit（令牌桶）

```text
令牌桶：容量 N，每秒补充 rate 个令牌，请求消耗 1 个令牌
```

作用：**从源头平滑限制到达速率**，让下游（GPU）不被突发打穿。和 load shedding 的区别：rate limit 按**速率**（时间维度）限制，load shedding 按**容量**（空间维度）限制。

### load shedding（有界队列 + 丢弃）

```text
队列容量 N，满了就丢弃新请求（或丢弃最旧的）
```

作用：**保护服务自身**——队列无限增长 = 延迟爆炸 + 内存耗尽。这是"为什么不能无限接"的直接答案。

### circuit breaker（三态熔断器）

```text
closed  正常，请求通过
open    熔断，快速失败（不调用下游）
half_open 试探性放行，成功则回 closed，失败则回 open
```

本模块实测（下游连续失败）：

```text
fail->closed  fail->closed  fail->open  reject(open)  reject(open) ...
```

第 3 次失败后熔断（open），后续请求快速失败，**不再重试失败的下游**。作用：下游（worker/GPU）故障时，**快速失败比无限重试好**——无限重试会耗尽线程、连接、超时，把一个故障放大成整个服务的雪崩。

---

## 4. 其他生产机制（理论，Stage 后续展开）

| 机制 | 作用 | 本工程状态 |
|---|---|---|
| timeout | 请求/调用超时上限 | Stage 16 已实现 |
| retry | 失败重试（需幂等） | 理论 |
| backpressure | 队列深度反馈到上游 | 理论 |
| autoscaling | 根据负载扩缩 worker | Stage 19 |
| 灰度发布 / A/B | 版本逐步放量 | Stage 20/21 |

---

## 5. 本模块闭环小结

```text
问题：GPU 100% 时为什么不能无限接请求
      ↓
答案：延迟爆炸（p99 9.9s）+ 内存耗尽
      ↓
防线：rate limit（速率）+ load shedding（容量）+ circuit breaker（失败）
      ↓
原则：宁可快速拒绝，不要慢慢堆积
      ↓
下一步：Stage 18 Kubernetes（Pod/Deployment/Service + GPU device plugin/scheduling）
```

要继续就说「继续」。
