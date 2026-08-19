# 01｜推理性能指标：从"GPU 很忙"到可证伪的推理结论

## 本模块解决的问题

面对"模型推理慢 / GPU 利用率低 / 吞吐上不去"时，第一件事不是打开 TensorRT 或 fp16，而是建立不会混淆的测量契约：每个指标到底在测什么、分子分母是什么、边界在哪里、能回答什么问题、不能回答什么问题。

本章的目标是形成下面这个最小证据闭环，并让"GPU utilization 很高"这类说法变成可以被数据证伪的结论：

```text
固定模型 + 正确性
        ↓
同步边界下测 baseline（原始样本 + p50/p90/p95/p99）
        ↓
用 timeline 判断 CPU-bound / launch-bound / memory-bound / compute-bound
        ↓
只对已识别瓶颈做一个改动
        ↓
同口径重测、重 profile、解释差异
```

配套代码见 `src/inference/benchmark_latency.py`、`src/inference/benchmark_throughput.py` 和 `src/common/measure.py`。本仓库不保存任何未实测的性能数字。

---

## 1. 先定义"一次推理"的边界

推理系统的延迟至少可以切成四层，每层是不同团队负责、不同工具测量：

```text
client latency
   ↓
service latency（队列 + 调度 + 网络 + 序列化）
   ↓
runtime latency（Python + dispatcher + allocator + kernel launch）
   ↓
device latency（CUDA kernel 真正执行）
```

如果两次实验用了不同的边界，任何 speedup 数字都不可比。本模块 benchmark 的边界是**单请求、隔离的模型 forward**：

```text
host synchronize
  → 一次 forward（固定 batch、shape、dtype）
  → host synchronize
```

它隔离出"模型本身在 GPU 上跑一次要多久"，但**刻意切断了跨请求的流水**，所以不等于生产 serving 的自然吞吐。生产吞吐要另做"不逐请求同步、只在测量窗口末尾同步"的长窗口实验（见 `benchmark_throughput.py`）。

### 为什么 CUDA async 会让朴素计时出错

CPU 调用 `model(x)` 后通常**立即返回**，因为 kernel 只是被提交到 CUDA stream，GPU 可能还在执行。下面的代码大多只测到了 launch 时间：

```python
import time
t0 = time.time()
model(x)          # 可能只是入队
elapsed = time.time() - t0   # 错：GPU 可能还没跑完
```

正确方法二选一：

1. **CUDA Event**：在 stream 上记录 start/end event，`elapsed_time` 给出 device 时间，排除了 host 开销。
2. **host wall + 同步**：起点和终点都 `torch.cuda.synchronize()`，得到客户端真正感知的 wall latency（含 host 开销）。

本模块同时报两个数，而且**两者的差本身就是结论**（见下）。

---

## 2. 指标字典

### Latency（延迟）

单次请求从发起到完成的耗时。必须报分布，而不是一个均值：

| 指标 | 含义 | 为什么重要 |
|---|---|---|
| mean | 算术平均 | 被长尾拉高，容易骗人 |
| p50 | 中位数 | "典型"体验 |
| p90 / p95 | 第 90/95 百分位 | 一般用户 / 尾部用户 |
| p99 | 第 99 百分位 | 长尾；机器人实时性主要看这里 |
| max | 最坏情况 | 调度抖动、GC、thermal 的痕迹 |

机器人场景（Stage 14 会展开）里，**平均 15ms 但 p99 = 200ms** 往往意味着控制环抖动、掉帧、动作超时，比平均慢更致命。

### Throughput（吞吐）

单位时间处理的工作量。单位随场景不同：

```text
samples/s   图片 / 音频 / 单个请求数
tokens/s    LLM 生成
QPS / RPS   serving 请求数
```

**吞吐是延迟的反问题**：一个请求延迟越低，不等于系统吞吐越高（后面 03 展开 Little's Law）。

### Latency 与 Throughput 的经典陷阱

把 `1 / latency` 当成 throughput，只在**严格串行、无重叠**时才成立。真实 serving 里请求并发进入、kernel 之间可以重叠，吞吐远高于 `1/单请求延迟`。反过来，把吞吐的倒数当成"每个请求的延迟"也错，因为排队和 batch 分摊都会改变单个请求的体感延迟。

### Batch Size（批大小）

一次 forward 里同时处理的样本数。它是延迟与吞吐之间最直接的旋钮：

```text
batch ↑  →  tensor core 利用率 ↑  →  吞吐 ↑
batch ↑  →  单样本摊销的 launch 开销 ↓
batch ↑  →  单请求排队时间 ↑  →  单请求延迟 ↑
```

对 latency-sensitive 的机器人在线控制，常见 batch=1；对离线/云端 LLM，batch 越大吞吐越高（直到显存或 kernel 效率饱和）。

---

## 3. GPU 利用率为什么 ≠ 执行效率

这是本模块最重要的一条判断。`nvidia-smi` 里的 `GPU-Util` 是**采样窗口内"GPU 是否至少有一个 kernel 在执行"的比例**，它是一个"忙不忙"的信号，不是"忙得是否有效"的信号。

```text
GPU utilization = 采样到 GPU 有 active kernel 的时间占比
```

推论：

- 一个只吃满 3% 峰值 FLOPs 的**低效 kernel**，只要一直在跑，utilization 就是 100%。
- 一个高效但很短的 Tensor Core burst 与 CPU gap 交替出现，utilization 可能只有 30%，但真正的关键路径是 CPU。

所以要区分"设备忙"和"设备忙得有价值"，至少还要看：

```text
SM utilization      SM 上是否有 warp 可调度的比例（比 GPU-Util 更细）
occupancy           每个 SM 实际活跃 warp / 理论最大 warp
Tensor Core util    是否真的在用 Tensor Core 而不是 CUDA core 打满
DRAM throughput     实际显存带宽 / 峰值带宽（memory-bound 工作的关键）
arithmetic intensity 每个 byte 搬运对应的 FLOPs（Roofline 的横轴）
```

这些指标是 **trace / 硬件计数器** 指标，只能从 Nsight Compute / Nsight Systems / PyTorch Profiler 拿，**不能用 wall time 猜**。本模块 benchmark 里这些字段在 profiler 阶段采集，不在这里用 `perf_counter` 冒充。

---

## 4. 判断 bound 类型

看到"慢"之后，第一步是把它归类到下面之一，因为每种 bound 的优化手段完全不同：

| bound 类型 | 症状 | 证据来源 | 典型优化 |
|---|---|---|---|
| CPU-bound | GPU 有空隙，CPU 一直忙 | nsys：GPU idle gap + CPU busy | 减少 Python 开销、C++ runtime |
| launch-bound | 大量 tiny kernel，GPU 忙但每个很短 | nsys：kernel 数量大、每个 <5us | fusion、CUDA Graph |
| memory-bound | DRAM throughput 接近峰值，SM 空闲 | ncu：DRAM 高、compute 低 | kernel fusion、减少搬运 |
| compute-bound | SM 打满、Tensor Core 高 | ncu：compute 高 | 换低精度、更好 GEMM |
| synchronization-bound | GPU 等待 host 的 sync | nsys：`cudaDeviceSynchronize` | 去掉 sync、异步化 |
| I/O-bound | 数据加载慢于 GPU | nsys：H2D / dataloader 占比 | pinned、async H2D、prefetch |

**同一个数字背后可能是完全不同的病**：`GPU-Util = 20%` 可能是 CPU 喂不饱、可能是 launch 间隙、可能是 H2D 阻塞、也可能是 kernel 本身 memory-bound 导致 SM 空转。这正是"看到 GPU 利用率低"不能直接说"加大 batch"的原因。

---

## 5. 本机实测数据（Thor, sm_110，仅供参考）

下面的数字是 `src/inference/benchmark_latency.py` 在 `NVIDIA Thor`（Jetson 平台，统一内存，sm_110）上实测的，仅用于说明**指标的读法**，不构成任何硬件标称。不同机器、时钟、dtype 会完全不同，复现命令见 `src/inference/README.md`。

一个 4 层、hidden=1024、batch=1 的残差 MLP：

```text
wall_latency（host 同步，含 launch 开销）:
  mean 237us  p50 235us  p90 242us  p95 251us  p99 261us  max 333us

event_latency（CUDA event，纯 GPU 时间）:
  mean 229us  p50 228us  p90 235us  p95 243us  p99 249us
```

读法：

1. `wall > event`，差值约 8us 就是 host 侧 launch/dispatcher 开销。batch=1 的小模型里，这个差值是"launch-bound"的征兆。
2. `max(333) 明显高于 p99(261)`，长尾来自偶发的 CPU 调度抖动或 allocator 重分配——这正是只看 mean 会漏掉的信息。

throughput 的 batch sweep（同模型，`benchmark_throughput.py`）：

```text
batch    samples/s      avg batch latency
1        4,499          0.22 ms
8        26,845         0.30 ms
64       102,514        0.62 ms
256      222,284        1.15 ms
```

读法：batch 从 1 到 256，吞吐提升约 49 倍，但单 batch 延迟也从 0.22ms 涨到 1.15ms——**吞吐和单请求延迟是同时上升的**，这就是 03 要定量讨论的 tradeoff。

---

## 6. 指标采集的纪律

1. **warmup**：先跑若干次不进统计，排除 cuBLAS 初始化、JIT、allocator 预热、时钟爬坡。
2. **样本数**：latency 至少 100+ 次，保留原始样本，否则 p99 不可信。
3. **同步边界**：起点和终点都同步；不要把 `loss.item()` 这类隐式同步偷偷混进来。
4. **固定环境**：clocks、dtype、shape、功耗状态尽量固定，跨实验才可比。
5. **拒绝伪造**：拿不到的数字（如无 GPU、无 ncu 权限）写 `Not Validated`，不补 0。

---

## 7. 本模块闭环小结

```text
问题：模型推理慢 / GPU 利用率低 / 吞吐上不去
      ↓
原理：latency 与 throughput 是两个量；GPU-Util 是"忙不忙"不是"忙得有效"
      ↓
指标：mean/p50/p90/p95/p99/max、samples/s、tokens/s、QPS
      ↓
Baseline：src/inference/benchmark_latency.py + benchmark_throughput.py
      ↓
判断：CPU / launch / memory / compute / sync / I/O bound
      ↓
优化：只改一处 → 同口径重测 → 解释差异
```

下一模块：`02_gpu_inference_pipeline.md`，回答"一次推理到底经过了哪些 stage、数据在哪里、同步点在哪、每一段可能产生多少延迟和显存"。
