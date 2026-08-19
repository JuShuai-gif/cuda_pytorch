# 03｜Latency 与 Throughput：Little's Law 与 Batch 的权衡

## 本模块解决的问题

"延迟低"和"吞吐高"经常被混为一谈，但它们是被不同变量驱动的两个量。本章建立二者的定量关系，回答：

```text
为什么 batch 变大，吞吐上升但单请求延迟也上升？
为什么 1/latency 不等于 throughput？
什么时候该优化 latency，什么时候该优化 throughput？
```

配套代码：`src/inference/benchmark_throughput.py`（batch sweep），`src/inference/benchmark_latency.py`（单请求延迟）。

---

## 1. 两个量的定义再次对齐

```text
Latency    = 一个请求从进到出的时间（单位 ms）
Throughput = 单位时间完成的请求数 / 样本数 / token 数
```

它们不是互为倒数，因为它们测量的是**不同边界的系统**：

- latency 测的是"一个请求的端到端"；
- throughput 测的是"很多请求叠加后，系统稳定输出的速率"。

---

## 2. Little's Law：把两者串起来

稳定系统中，以下关系恒成立（这是排队论的基本恒等式，不是近似）：

```text
L = λ × W

L = 系统中平均在途请求数（concurrency，或"在飞请求"）
λ = 平均到达/完成速率（throughput）
W = 平均每请求驻留时间（latency）
```

推论，对应到 GPU 推理服务：

```text
throughput = concurrency / latency
latency    = concurrency / throughput
```

这解释了很多反直觉现象：

1. **吞吐上不去**：如果并发 L 固定，而 latency W 变大，吞吐 λ 必然下降。所以"单请求变慢"会直接拖累吞吐。
2. **延迟不降反升**：如果为了提吞吐而提高并发 L，在 latency 没有同步下降时，W 反而可能上升（请求排队）。
3. **1/latency 只是并发=1 的特例**：只有当系统里同时只有 1 个请求（batch=1、串行）时，throughput 才等于 1/latency。

---

## 3. Batch 如何同时影响两边

GPU 有大量并行单元（SM、Tensor Core）。单个样本的算子往往吃不饱整个 GPU，所以把多个样本合成一个 batch 能提升硬件利用率：

```text
batch ↑
   → 每个 GEMM 的 M/N/K 更大 → Tensor Core 利用率 ↑
   → 单样本摊销的 kernel launch 开销 ↓
   → 结果：吞吐 ↑
```

但代价：

```text
batch ↑
   → 每个请求要等同 batch 的其它请求到齐（batching delay）
   → 每个 kernel 绝对耗时更长，排在后面的请求等更久
   → 结果：单请求 latency ↑
```

所以 **batch 是 latency 与 throughput 之间的旋钮**，不存在"既把 latency 压到最低又把 throughput 拉到最高"的 batch。选择 batch 是在回答"我要优化谁"。

---

## 4. 实测：batch sweep 的双向影响

本机（Thor, sm_110）4 层 hidden=1024 残差 MLP，`benchmark_throughput.py`：

```text
batch    samples/s       avg batch latency（每个 batch 的处理时间）
1        4,499           0.22 ms
8        26,845          0.30 ms
64       102,514         0.62 ms
256      222,284         1.15 ms
```

三个观察：

1. **吞吐单调上升**：batch 1→256，吞吐升约 49 倍。因为更大的 GEMM 吃满了 Tensor Core。
2. **单 batch 延迟也上升**：0.22→1.15ms。这就是 latency 的代价。
3. **边际收益递减**：从 64→256 吞吐只翻倍多一点，说明逐渐逼近硬件饱和（内存带宽 / SM 数）。继续加大 batch 只会让延迟涨而吞吐不再涨。

结论：**存在一个吞吐饱和点**，过了它 batch 只是增加延迟。找到这个点要靠 sweep，不能拍脑袋。

---

## 5. 什么时候优化 latency，什么时候优化 throughput

| 场景 | 优先目标 | 典型手段 |
|---|---|---|
| 机器人在线控制、VLA 实时推理 | latency + jitter | batch=1、CUDA Graph、fusion、去同步 |
| 离线批量处理、数据标注 | throughput | 大 batch、batching、低精度 |
| 在线 LLM 服务 | 两者权衡 | continuous batching、KV cache、调度 |
| 云边协同 | 视链路 | 边缘保 latency，云端保 throughput |

判断方法：先问"用户/机器人对这个请求的延迟敏感，还是对系统单位时间能处理多少量敏感"。机器人（尤其控制闭环）几乎永远是 latency-first；离线推理几乎永远是 throughput-first；在线服务是带约束的优化问题。

---

## 6. Latency 的分布为什么比均值重要

同一套系统，均值一样的两个实现可能有完全不同的尾部：

```text
实现 A：p50=15ms  p99=18ms  max=20ms
实现 B：p50=10ms  p99=180ms max=300ms
```

实现 B 的均值甚至可能更低，但对机器人控制环，p99=180ms 意味着每 100 次就有 1 次动作迟到，可能导致碰撞或超时。**实时系统的可用性由尾部决定**（详见 Stage 26 实时性）。所以任何 latency 实验都必须报 p90/p95/p99，而不是只报 mean 或只报"运行时间 = 10ms"。

尾部来源常见于：allocator 重分配、CPU 调度抢占、GC、时钟/thermal 抖动、跨 stream 的偶发同步。

---

## 7. Throughput 的另类度量：GPU 时间利用率

吞吐高不一定 GPU 效率高。更硬的吞吐指标是：

```text
tokens/s per GPU     单位算力的产出
MFU（推理版）        有效 FLOPs / (峰值 FLOPs × time)
DRAM bandwidth util  实际 bytes/s / 峰值 bytes/s
```

一个 kernel 很长的慢实现，wall 吞吐可能高（因为它在跑），但 MFU 可能很低（因为没吃满 Tensor Core）。所以"吞吐提升了 30%"也要问：是吃得更满了，还是只是多跑了无效工作。

---

## 8. 本模块闭环小结

```text
问题：latency 和 throughput 是什么关系，batch 怎么选
      ↓
原理：L = λ × W；batch 提升硬件利用率但增加排队
      ↓
测量：latency（p50/p90/p95/p99）+ throughput（samples/s、tokens/s）
      ↓
实验：batch sweep 找吞吐饱和点
      ↓
决策：latency-first（机器人）vs throughput-first（离线）vs 权衡（在线）
```

下一模块：`note/kernel/01_cuda_execution_model.md`，下钻到 Thread/Warp/Block/Grid/SM 和内存层次，回答"GPU 为什么快、什么会拖慢它"。
