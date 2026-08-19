# 00. AI Infra / Training Systems：从“GPU 很忙”到可证明的训练优化

> 本章不是术语索引，而是后续所有实验的统一工作方法。目标是：面对吞吐下降、MFU 低、显存爆炸、GPU 空洞或长周期训练抖动时，先建立可证伪的性能模型，再用数据定位、修改和复测。

## 0. 本阶段的边界与实验事实

当前 Stage 1 只研究单 GPU 性能基础：训练指标、GPU 执行与显存、Roofline，以及 PyTorch Profiler / Nsight Systems / Nsight Compute 的分层方法。

本工程当前可验证环境为单张 NVIDIA Thor；因此：

- 单 GPU correctness、benchmark、PyTorch Profiler、Nsight Systems / Compute 的可用部分可以实测；
- 8 / 128 / 1024 GPU、InfiniBand、NCCL 跨机通信和 scaling efficiency 尚不能在本机实测；
- 未采集到的数字必须写成 `N/A (not measured)`，不能用估计值冒充实验值；
- 硬件峰值、计数器名称和 profiler 能力必须从实际机器查询，不能把 A100/H100/B200 的默认值套到 Thor。

对应工程：

```text
note/01_training_performance_metrics.md  <-> src/metrics/
note/02_gpu_execution_and_memory.md      <-> src/gpu_basics/
note/18_profiling_methodology.md         <-> src/profiling/
```

## 1. Training Systems 工程师真正优化的对象

训练并不是一个“GPU kernel”，而是一条跨 CPU、GPU、内存、存储和网络的流水线：

```text
storage / dataset
        ↓ read + decode + augment
CPU DataLoader / pinned host buffers
        ↓ H2D copy
GPU forward
        ↓
loss + backward
        ↓
gradient communication（分布式阶段）
        ↓
optimizer update
        ↓
checkpoint / logging / evaluation
```

用户看到的 step time 是整条关键路径的墙钟时间，不是所有子阶段耗时的简单相加。异步执行和 overlap 会让多个阶段同时发生：

\[
T_{step} = T_{critical\ path}
\]

一个便于诊断、但不能机械相加的分解是：

\[
T_{step} \approx T_{GPU\ useful} + T_{exposed\ input} + T_{exposed\ H2D}
+ T_{exposed\ comm} + T_{bubble/sync} + T_{exposed\ checkpoint}
\]

其中 `exposed` 表示没有被其他有用工作隐藏、真正延长 step 的部分。例如 NCCL kernel 总共运行 20 ms，并不代表它给 step 增加了 20 ms；若其中 15 ms 与 backward compute 重叠，最多只有余下的暴露部分在关键路径上。

### 1.1 CPU 和 GPU 分别在做什么

| 时间段 | CPU 典型工作 | GPU 典型工作 | 常见风险 |
|---|---|---|---|
| 数据准备 | 调度 worker、读取、解码、拼 batch、pin memory | 可能仍在执行上一步 | worker 不足、small files、decode 慢 |
| 发射计算 | Python/C++ dispatcher、选择 kernel、enqueue | 执行已入队 kernel | 大量小 kernel 导致 launch-bound |
| forward/backward | 准备下一次发射或数据 | GEMM、attention、norm、reduction | compute/memory bound、bubble |
| 同步点 | 等待 CUDA event / tensor 结果 / collective | 完成队列或等待依赖 | 隐式同步、CPU 阻塞 |
| optimizer | 发射更新 kernel、管理状态 | 更新参数/状态 | kernel 碎片、显存带宽压力 |

CUDA API 大多是异步的：CPU 上 `op()` 返回，通常只代表工作已入队，不代表 GPU 已完成。因此用普通 Python `time` 包住 CUDA op 而不在正确边界同步，测到的主要是 enqueue 时间。

### 1.2 训练状态在哪里

单 GPU 训练的典型驻留关系如下；实际 dtype、master weight、offload 和 fused optimizer 会改变它：

| 对象 | 常见位置 | 生命周期 | 主要显存风险 |
|---|---|---|---|
| parameters | GPU | 整个训练 | 参数 dtype、额外 master weights |
| gradients | GPU | backward 后到清零/更新 | 是否与参数 storage 复用、梯度累积 |
| activations | GPU | forward 到对应 backward | 常是长序列/视频训练的大头 |
| optimizer state | GPU | 整个训练 | Adam 常有一阶、二阶状态及可能的 FP32 副本 |
| temporary workspace | GPU allocator / library | kernel 或算子阶段 | cuBLAS/cuDNN/attention workspace |
| CUDA context / library / graph pools | GPU | 进程或图生命周期 | 不完全计入 PyTorch allocated bytes |
| input batch / staging | CPU pinned + GPU | 每批 | prefetch 深度和并发 batch |

必须同时区分：

- **live tensor bytes**：当前仍可达 tensor 的 storage；
- **PyTorch allocated**：allocator 正被 tensor 使用的字节；
- **PyTorch reserved**：caching allocator 从系统保留的内存；
- **device/process used**：还包括 CUDA context、其他库和非 PyTorch allocation；
- **peak**：必须在明确测量窗口前 reset，并在窗口后读取。

只看 `nvidia-smi` 的进程显存无法解释 allocator 碎片；只看 `memory_allocated()` 又会漏掉框架外 allocation。

## 2. 指标不是一个数，而是一棵因果树

### 2.1 第一层：业务产出

\[
throughput_{samples} = \frac{global\ batch\ size \times measured\ steps}{elapsed\ time}
\]

\[
throughput_{tokens} = \frac{non\ padding\ tokens\ processed}{elapsed\ time}
\]

训练系统最终应以稳定的 samples/s、有效 tokens/s、视频 clips/s，或达到目标 loss 所需时间为准。不同 sequence length、分辨率、帧数或有效 token 比例的吞吐不可直接比较。

### 2.2 第二层：step time 与分位数

均值会隐藏周期性 checkpoint、dataloader 抖动和 straggler。至少记录：

- warmup 后的 p50 / p90 / p99 step time；
- 最小、最大、均值、标准差和样本数；
- measurement window 内是否包含 checkpoint / eval；
- 每个 step 的 batch、tokens、shape 是否相同。

### 2.3 第三层：硬件效率

若一次 step 按明确 convention 需要 \(F_{model,step}\) 次模型 FLOP，设备所选 dtype 的实测或公开峰值为 \(F_{peak}\)，则：

\[
MFU = \frac{F_{model,step}/T_{step}}{N_{GPU}\,F_{peak}}
\]

HFU 的分子是硬件实际执行的 FLOP。activation recomputation、padding、重复计算和某些通信相关计算会增加 HFU 分子，却不增加“完成多少模型训练工作”的 MFU 分子。因此通常不能把 MFU 和 HFU 混为一谈。

任何 MFU/HFU 必须随结果保存：

```text
FLOP convention
model / input shape
global batch or effective tokens
precision
peak FLOP source
dense or sparsity peak
GPU count
timing boundary
```

如果设备峰值未知，正确结果是 MFU `N/A`，而不是静默回退到 A100。

### 2.4 第四层：资源和时间线

需要解释上层指标的底层证据包括：

- GPU active time、idle/bubble；
- kernel 数量、每个 kernel duration、launch gap；
- SM / Tensor Core 与 DRAM/L2 吞吐；
- CPU enqueue、DataLoader、H2D 和同步等待；
- peak allocated/reserved/device memory；
- 分布式阶段的 NCCL total 与 exposed communication；
- scaling efficiency。

## 3. 为什么 GPU utilization 不等于 MFU

`nvidia-smi` 风格的 GPU utilization 通常回答：“采样窗口内 GPU 是否有 kernel 在运行？”MFU 回答：“相对于所选精度的理论峰值，完成了多少定义明确的模型 FLOP？”两者的分子和时间尺度完全不同。

于是可能出现：

1. **utilization 高、MFU 低**：GPU 一直执行低效 elementwise kernel、内存受限 kernel、recompute 或大量小 kernel；“忙”不等于高价值 Tensor Core 计算。
2. **utilization 抖动、MFU 低**：DataLoader 或 CPU launch 造成时间线空洞。
3. **utilization 高、throughput 仍差**：做了过多 padding、错误 shape、数值检查或重计算；HFU 可能不低，MFU/业务吞吐仍低。
4. **短采样窗口 utilization 看似低**：工具采样周期和短 burst 不匹配；应查看 timeline，而不是只信单个百分比。

因此排查顺序应是：先确认 workload 与吞吐，再看 step timeline，再看 kernel / hardware counters。

## 4. Roofline：先判断“上限由什么决定”

算术强度：

\[
AI = \frac{FLOP}{bytes\ transferred}
\]

Roofline 给出的性能上限：

\[
P_{attainable} \le \min(P_{peak}, AI \times BW_{peak})
\]

拐点为：

\[
AI_{ridge} = \frac{P_{peak}}{BW_{peak}}
\]

- \(AI < AI_{ridge}\)：理论上更容易 memory-bound；减少搬运、融合、提高复用通常比增加算术更重要。
- \(AI > AI_{ridge}\)：理论上可能 compute-bound；Tensor Core 映射、tile、occupancy 和指令流水更关键。

但 Roofline 是必要非充分判断：低 SM、低 DRAM 也可能是 launch latency、依赖链、同步或小 grid 导致，不能仅因算术强度低就宣判“带宽已打满”。字节数还必须说明经过 DRAM、L2 还是 shared memory；不同层级对应不同 roofline。

## 5. 五类常见瓶颈的可验证假设

| 假设 | CPU/GPU timeline | PyTorch Profiler | Nsight Systems | Nsight Compute | 第一项对照实验 |
|---|---|---|---|---|---|
| launch-bound | CPU 连续发射，GPU 上大量短 kernel 与 gap | op/kernel 数量多，自身 kernel 很短 | CUDA API 与 kernel launch 密集、GPU 空隙明显 | 单 kernel 可能没有明显硬件瓶颈 | 合并算子、增大 batch、compile/graph 后比较 kernel 数与 step |
| memory-bound | GPU 持续执行但算术产出低 | copy/elementwise/reduction 占主导 | kernel 连续，未必有 idle | DRAM/L2 吞吐接近该工作负载可达上限，算术吞吐相对低 | 减少 bytes、融合中间张量，检查 duration 与 bytes |
| compute-bound | 大 kernel 占满主要时间 | GEMM/attention 占主导 | 长计算 kernel 连续 | 对应精度计算管线/Tensor Core 利用高，DRAM 非主限制 | 降 FLOP 或换更优 GEMM/shape/precision |
| synchronization-bound | CPU/GPU 或 streams 互相等待 | `cudaDeviceSynchronize`、item、copy 等同步可见 | 同步 API、依赖箭头、空洞明显 | NCU 只分析单 kernel，不能单独证明系统级同步 | 移除/推迟同步，比较关键路径而非只比 kernel |
| CPU/input-bound | GPU 有大段无 kernel | DataLoader/CPU op 很长 | CPU thread、I/O、H2D 与 GPU gap 对齐 | NCU 通常不是第一工具 | synthetic data 或预生成 batch A/B |

“接近峰值”不能使用跨架构固定阈值。应比较同一机器、同一 shape 的 baseline/optimized，并同时引用实际 counters、规则和 workload 上限。

## 6. GPU bubble 是什么

Bubble 是本可用于完成当前训练 step、但关键路径上 GPU 没有执行有用工作的时间。它可能来自：

- CPU 尚未发射下一个 kernel；
- DataLoader/H2D 未准备好；
- 显式或隐式同步；
- 跨 stream 依赖；
- 分布式 collective 的暴露时间；
- pipeline parallel 的调度空洞；
- allocator、checkpoint 或 Python GC 造成的停顿。

粗略的 timeline 定义为：

\[
bubble\ ratio = 1 - \frac{useful\ GPU\ busy\ time\ on\ critical\ window}{window\ duration}
\]

必须明确“useful”的分类规则。若把 NCCL、memcpy、recompute 或错误的 padding kernel 都算作有用 compute，bubble 会被低估。

## 7. 性能证据闭环

每一个优化都必须交付同一条链：

```text
Problem statement + falsifiable hypothesis
        ↓
Baseline correctness
        ↓
Controlled benchmark
        ↓
System timeline（PyTorch Profiler / Nsight Systems）
        ↓
Hotspot kernel counters（必要时 Nsight Compute）
        ↓
One controlled change
        ↓
Correctness / numerical tolerance
        ↓
Same benchmark again
        ↓
Same profiler again
        ↓
End-to-end impact + caveats + stability check
```

### 7.1 Benchmark contract

每次 A/B 实验应固定并记录：

- code revision、命令和随机种子；
- GPU、driver、CUDA、PyTorch、profiler 版本；
- model/input shape、dtype、layout、batch、sequence/video dimensions；
- eager / compile、CUDA Graph、autocast 和 TF32 状态；
- warmup 次数、测量次数、同步边界；
- power/thermal 状态及同机干扰；
- 输出正确性容差；
- 原始逐 step 数据，而不只是一行均值。

不能让 baseline 用冷启动而 optimized 用热缓存；不能改变 batch 后只比较 samples/s 而不说明；不能只优化一个 2% 热点便宣称端到端大幅提速。

### 7.2 最低比较表

| 指标 | Baseline | Optimized | 证据源 |
|---|---:|---:|---|
| step p50 / p90 / p99 | measured | measured | benchmark raw samples |
| samples/s 或有效 tokens/s | measured | measured | workload counter / wall time |
| MFU / HFU | measured 或 N/A | measured 或 N/A | FLOP convention + peak source |
| GPU active / bubble | measured 或 N/A | measured 或 N/A | timeline + 分类规则 |
| exposed communication | 单 GPU N/A | 单 GPU N/A | 多卡 timeline |
| peak allocated / reserved | measured | measured | PyTorch memory stats |
| kernel count / duration | measured | measured | profiler |
| scaling efficiency | 单 GPU N/A | 单 GPU N/A | 多卡 benchmark |

## 8. 三层 profiler：不要拿错工具

### 8.1 PyTorch Profiler：框架语义

回答“哪个 PyTorch op / module / step 占时、分配显存、调用了哪些 kernel”。适合快速建立 operator-level baseline，但 profiler 本身有开销，带 profiler 的绝对吞吐不能代替无 profiler benchmark。

### 8.2 Nsight Systems：系统关键路径

回答 CPU threads、CUDA API、GPU kernels、memcpy、NVTX、NCCL 和 I/O 在时间轴上如何交错。先用它确认 hotspot、bubble、同步和 overlap，再选择代表性 kernel。

### 8.3 Nsight Compute：单 kernel 机制

回答代表性 kernel 的 launch geometry、occupancy、SM/Tensor pipeline、DRAM/L2、warp stalls、register/shared-memory 限制。它会 replay kernel，不能用 NCU 采集时的应用墙钟时间当真实 step time。

在当前 Thor 上遵循：

1. 先 `--set basic`；
2. 仅在问题需要时升级 `detailed` / source counters / `full`；
3. 先枚举报告里实际存在的 metric；缺失是 `unavailable`，不是 0；
4. CUDA Graph 需要 node 级 profiling；NVTX push/pop include 名称的尾部 `/` 不能遗漏；
5. 任何 kernel 优化最后都回到无 profiler 的端到端 benchmark。

## 9. 从 1 GPU 到 1024 GPU，为什么结论会改变

单卡阶段测得的 compute/memory/launch 上限仍是分布式建模的基础，但系统关键路径会变化。

### 8 GPU（单机）

- NVLink / NVSwitch / PCIe topology 开始决定 collective 带宽；
- backward compute 与 NCCL overlap 决定 exposed communication；
- batch per GPU 变小可能让 GEMM shape 和单卡效率变差；
- rank 间 dataloader、clock、thermal 差异会放大为 straggler。

### 128 / 1024 GPU（多机）

- 多层网络、rail、NIC/NUMA affinity、拥塞和 collective 算法进入关键路径；
- 最慢 rank 决定同步 step，尾延迟比平均值更重要；
- checkpoint 元数据/并发写入、作业启动、容错与恢复成为一等瓶颈；
- 固定 global batch 下 weak/strong scaling 语义不同；
- 小概率故障会在大量设备和长时间下变成常态。

因此不能用“8 卡 NCCL 总时长占比”直接外推 1024 卡，也不能将通信总时间直接等同于通信增加的 step time。

## 10. 生产问题的排查顺序

面对“训练变慢 / MFU 低”，按以下顺序保护第一现场：

1. **确认问题定义**：吞吐、loss、batch/tokens、shape、precision、world size 是否真的相同。
2. **保存环境与原始数据**：revision、配置、rank 日志、step samples、GPU/CPU/网络状态，不要先重启销毁证据。
3. **看时间序列**：何时开始、持续还是周期性、所有 rank 还是单 rank、是否与 checkpoint/eval 对齐。
4. **建立无 profiler baseline**：固定 workload，warmup 后重复，报告分位数和抖动。
5. **看系统 timeline**：CPU、GPU、copy、sync、后续阶段 NCCL 的关键路径。
6. **隔离变量**：synthetic data、关闭 checkpoint/eval、固定 shape、单卡/单机对照。
7. **只对真实 hotspot 用 NCU**：记录精确 kernel、launch index、shape 和报告原件。
8. **一次改一个主要变量**：先 correctness，再同条件 benchmark/profile。
9. **验证稳定性**：更长窗口、真实数据、多 shape、内存峰值和数值行为。
10. **形成可回滚方案**：监控指标、启用条件、fallback 和故障恢复步骤。

### 10.1 高频误诊

- GPU utilization 高就断定“GPU 没问题”；
- `torch.cuda.synchronize()` 塞进每个 op 后仍把结果称为真实流水性能；
- 用 CPU wall clock 测异步 CUDA kernel；
- 把 allocated、reserved 和 device used 混成一个“显存”；
- 用理论 FLOP 公式但不写 convention；
- 只看 NCU 单 kernel，不看它在端到端时间中的占比；
- 把 profiler 注入后的慢速直接当生产吞吐；
- 报告不存在的硬件 counter 为 0；
- 看到通信 kernel 就把它全部算作通信开销。

## 11. 本阶段完成标准

完成本阶段后，应能独立回答并用实验验证：

- step time 和 throughput 的测量窗口是否正确；
- MFU 分子、分母和 FLOP convention 是否可审计；
- 为什么 GPU utilization 与 MFU 不等价；
- CUDA async 为什么让普通计时失真；
- 参数、梯度、activation、optimizer state、workspace 和 allocator cache 各占多少；
- Roofline 如何给出上限，以及为什么低利用率不一定是 bandwidth-bound；
- timeline 中哪些 gap 是 CPU/input、launch、sync 或 GPU bubble；
- 何时止步于 PyTorch Profiler，何时进入 Nsight Systems，何时才用 Nsight Compute；
- 如何用 baseline/profile/optimization/re-profile 证明变化来自目标机制；
- 哪些结论已在当前硬件实测，哪些必须留到真实多卡/集群验证。

后续所有 DDP、FSDP、Megatron、DiT、MoE、DataLoader 和 Checkpoint 模块，都复用本章的 measurement contract 与证据闭环。
