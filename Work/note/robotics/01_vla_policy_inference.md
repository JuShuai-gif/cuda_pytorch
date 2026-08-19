# 01｜VLA / Robot Policy 推理：实时性、Jitter 与 batch=1

## 本模块解决的问题

机器人推理和服务器 LLM 推理的目标完全不同：服务器追求 **max tokens/s（吞吐）**，机器人追求 **low latency + low jitter + predictable latency（可预测的实时响应）**。本章回答：

```text
机器人在线推理和服务器推理差在哪？
sensor-to-action latency 由哪些段组成？
为什么 mean latency 会掩盖 jitter，jitter 为什么致命？
batch=1 推理的瓶颈是什么，怎么优化？
```

配套代码：`src/robotics/policy_inference/`（`pipeline.py` + `realtime.py` + `batch1.py`）。

---

## 1. 机器人推理 vs 服务器推理

| 维度 | 服务器 LLM | 机器人 VLA/VLA policy |
|---|---|---|
| 目标 | max tokens/s | low & predictable latency |
| batch | 越大越好（KV cache 摊薄） | 几乎总是 batch=1 |
| 关键指标 | TTFT / TPOT / 吞吐 | sensor-to-action latency / jitter / deadline miss |
| 失败后果 | 用户多等一会 | 动作迟到 → 碰撞 / 超调 |
| 约束 | 显存、算力 | **物理实时性 + 安全边界** |

核心区别：**机器人有物理实时性**。一个 20ms 控制环（50Hz），如果某次推理 200ms 才返回，机器人就"卡"了 10 个周期，动作已经过时——这不是"慢"，是"危险"。

---

## 2. sensor-to-action latency 分解

```text
Camera（帧采集）
   ↓ image capture（传感器 → 内存）
   ↓ preprocess（resize/normalize）
   ↓ H2D（CPU → GPU）
   ↓ Vision Encoder（ViT）
   ↓ Policy / Language（LLM）
   ↓ Action Decoder（LLM → 关节角/动作向量）
   ↓ Postprocess（clamp/归一化）
   ↓ Robot Control（写执行器）
```

Stage 13 已测过 vision/LLM 段的分解（vision 37% + LLM 30% + CPU preprocess 26%）。机器人额外关注两端：

- **image capture 的抖动**：相机帧率不稳、曝光时间变化 → 输入本身就抖。
- **control 段的截止时间**：action 必须在下一个控制周期前到达执行器，否则丢帧。

---

## 3. 实测：mean 掩盖 jitter，jitter 致命

用 VLA policy 跑控制循环（deadline=10ms，200 个周期），对比"干净"和"注入 CPU jitter（每 50 周期一次 5ms stall，模拟 GC/调度抢占）"：

```text
                    mean      p50      p99      jitter(p99-p50)   miss rate
clean               7.17ms   7.12ms   8.74ms   ~1.6ms            0.0%
with CPU jitter     7.39ms   7.08ms   12.25ms  5.17ms            3.0%
```

### 读法（本模块的灵魂）

1. **mean 几乎不变（7.17 → 7.39ms，+3%）**，但 **p99 从 8.74 涨到 12.25ms（+40%）**。如果只看 mean，你会以为"没问题"；看 p99，才看到 jitter 已经破坏了实时性。

2. **deadline miss rate 从 0% 涨到 3%**：p99 超过 10ms deadline 后，每 100 个控制周期有 3 次动作迟到。对 50Hz 控制环，这意味着每秒 1.5 次"卡顿"。

3. **jitter 的来源是 CPU 的偶发 stall**（GC、调度器抢占、内存分配），不是 GPU。GPU 的 kernel 时间很稳定（CUDA 的确定性），是 CPU 侧的不可预测性制造了 tail latency。

这就是 master prompt 反复强调的：**"平均 latency = 15ms 但 p99 = 200ms" 可能导致机器人不可用**。

---

## 4. batch=1 推理的优化

机器人在线控制几乎总是 batch=1，此时推理是 launch-bound：

```text
batch=1 的小 policy → 每个 op 的 GPU 计算极小 → CPU 的 launch 开销占比大
```

本机实测（VLA policy 的 naive forward vs CUDA Graph）：

```text
naive：wall 4716us / event 4693us
graph：wall 4500us / event 4482us   （1.05x）
```

**为什么收益只有 5%（不像 Stage 2 的 5x）？** 因为 CUDA Graph 的收益取决于 launch overhead 的占比：

- Stage 2 的 64 个 tiny elementwise op：launch 开销占主导 → graph 5x。
- 本模块的 VLA policy（ViT + LLM 的 transformer layer）：每个 op 是较大的 GEMM，launch 开销占比小 → graph 只有 5%。

所以 batch=1 优化的完整手段（按优先级）：

| 手段 | 收益 | 适用场景 |
|---|---|---|
| CUDA Graph | 中 | 固定 shape、launch 开销占比大时 |
| operator fusion | 高 | 减少中间张量和 kernel 数 |
| 去同步（异步 H2D + 多 stream） | 高 | 让 preprocess 和 GPU 重叠 |
| 预分配 buffer（避免 allocator） | 中 | 消除 cudaMalloc 抖动 |
| 低精度（fp16/int8） | 高 | 减小 GEMM 时间 |

**关键：batch=1 的优化目标不是"快一点"，而是"消除 jitter"**——CUDA Graph 的价值在实时场景里主要是**让 latency 更稳定**（每次 replay 都是完全相同的 kernel 序列），而不只是更快。

---

## 5. 实时性指标总结

| 指标 | 含义 | 机器人为什么关心 |
|---|---|---|
| p50 | 典型延迟 | 常规表现 |
| p99 | 尾部延迟 | **决定 deadline miss** |
| jitter | p99 - p50（或 std） | 控制环的稳定性 |
| deadline miss rate | 超过控制周期比例 | 掉帧率、安全 |

监控机器人推理时，**必须采集 p99 和 deadline miss rate**，而不是只看平均延迟或 GPU 利用率。

---

## 6. 本模块闭环小结

```text
问题：机器人推理和服务器推理差在哪
      ↓
差异：机器人要 low latency + low jitter + predictable，不是 max throughput
      ↓
实测：mean 掩盖 jitter（+3%），p99 涨 40%，deadline miss 0%→3%
      ↓
优化：batch=1 的 CUDA Graph / fusion / 去同步 / 预分配
      ↓
下一步：Stage 15 Edge AI（Jetson/ARM/NPU + thermal/power + 长稳）
```

要继续就说「继续」。
