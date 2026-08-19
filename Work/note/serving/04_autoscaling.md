# 04｜Autoscaling：为什么 CPU 利用率不是 GPU 推理服务的好指标

## 本模块解决的问题

（Stage 18 Kubernetes 已跳过，本模块用模拟器讲 autoscaling 的核心概念，不依赖 K8s 集群。）

推理服务负载突增时需要扩容，但**用哪个信号触发扩容**是错的源头。本章回答：

```text
为什么 CPU utilization 通常不是 GPU inference service 最好的扩容指标？
queue length / latency based scaling 各自怎么工作？
扩缩容的响应延迟会带来什么代价？
```

配套代码：`src/serving/autoscaling/`（三种指标的离散事件模拟）。

---

## 1. 问题的根源：GPU 是瓶颈，CPU 是旁观者

GPU 推理服务里，CPU 和 GPU 的分工（Stage 2 讲过）：

```text
CPU：launch kernel、copy 数据、调度（很闲）
GPU：真正计算（瓶颈）
```

**当 GPU 已经 100% 饱和时，CPU 可能只有 20% 利用率**——因为 CPU 只是"指挥"，真正干活的是 GPU。所以：

```text
CPU utilization = 20%（看起来很有余量）
GPU utilization = 100%（实际已经打满）
```

如果用 CPU utilization 触发扩容，autoscaler 会认为"还早着呢"，**永远不扩容**，即使 GPU 已经过载、请求堆积、延迟爆炸。

---

## 2. 实测：三种指标的对比

模拟：服务 1 个 GPU worker（服务率 100 req/s），负载从 100 突增到 500 req/s（需要 5 个 worker）。

```text
metric   final_workers  mean_workers  dropped   mean_latency
cpu      1              1.00          67000     1.000s
queue    5              4.33          2400      0.992s
latency  8              7.47          0         0.576s
```

### 读法

1. **CPU 指标完全失效**：final_workers=1（从不扩容），丢弃 67000 个请求，延迟 1 秒。因为模拟里 CPU 利用率是固定的 20%（GPU 饱和时 CPU 仍闲），autoscaler 永远看不到"需要扩容"。

2. **queue 指标正确扩容**：扩容到 5（= 500/100），丢弃 2400（扩容响应延迟期间溢出的）。

3. **latency 指标最激进**：扩容到 8（max 上限），丢弃 0（完全消化 spike），但可能**过度扩容**（8 个 worker 服务 800 req/s，超出需要的 500，浪费资源）。

**结论：GPU 推理服务应该用 queue length 或 latency 触发扩容，而不是 CPU utilization。**

---

## 3. 三种扩容指标的权衡

| 指标 | 优点 | 缺点 |
|---|---|---|
| CPU utilization | 通用、稳定 | **和 GPU 负载脱节**（GPU 服务里失效） |
| queue length | 直接反映积压 | 滞后（积压已经发生） |
| latency | 直接反映用户体验 | 噪声大、可能过度扩容 |
| GPU utilization | 直接反映 GPU | 但 GPU util 高 ≠ 需要扩容（可能只是 batch 大） |

**工业实践**：
- GPU 推理服务用 **queue length** 或 **custom metric（GPU 队列深度、pending 请求数）**。
- 一个纯 CPU util 的 HPA 用在 GPU 服务上，是经典的生产事故（GPU 打满不扩容，用户全部超时）。

---

## 4. 扩缩容的响应延迟

本模块的 queue 指标丢弃了 2400 个请求——这不是 bug，是**扩容响应延迟**：

```text
负载突增 → 队列开始积压 → autoscaler 下一个周期才检测到 → 启动新 worker（还要预热）→ 才能处理
```

这个延迟（检测 + 启动 + 预热）期间，积压的请求要么排队（延迟高），要么丢弃（load shedding）。所以 autoscaling 不能替代前面 Stage 17 的**过载保护**——它们是互补的：

```text
过载保护（rate limit / load shedding）：立即止损
autoscaling：事后扩容，恢复容量
```

**先止损，再扩容**，这是生产服务的标准组合。

---

## 5. 本模块闭环小结

```text
问题：GPU 推理服务用什么信号触发扩容
      ↓
根源：CPU 和 GPU 负载脱节（GPU 100% 时 CPU 可能 20%）
      ↓
实测：CPU 指标从不扩容（丢 67000）；queue/latency 正确扩容
      ↓
结论：用 queue length / latency / GPU 相关指标，不用 CPU util
      ↓
下一步：Stage 20 灰度发布（模型 V1/V2 按比例放量 + 监控 + 回滚）
```

要继续就说「继续」。
