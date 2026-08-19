# 01｜训练性能指标：从“GPU 很忙”到可证伪的性能结论

## 本模块解决的问题

面对“训练慢、MFU 低、显存高”时，第一步不是调参数，而是建立不会混淆的测量契约：测量边界是什么、分子分母是什么、数据来自哪里、误差有多大。目标是形成下面的最小证据闭环：

```text
固定 workload 与正确性
        ↓
同步边界下测 baseline（原始样本 + 分位数）
        ↓
用 timeline 判断 CPU / launch / compute / memory / sync 空洞
        ↓
只对已识别瓶颈做一个改动
        ↓
同口径重测、重做 profile、解释差异
```

本章的代码是测量载体，不预设 `torch.compile` 一定更快，也不提交任何虚构数字。

---

## 1. 先定义一个 step

训练系统常见的 step 边界至少有三种：

1. **device-only step**：只含当前 iteration 的 GPU 工作；适合研究 kernel 和通信关键路径。
2. **trainer step**：从取得一个已经准备好的 batch，到 optimizer update 完成。
3. **end-to-end step**：还包括 DataLoader 等待、H2D、日志、周期性评估和 checkpoint 摊销。

如果两次实验使用不同边界，任何 speedup 都无效。本模块 benchmark 的边界是：

```text
CUDA synchronize
  → optimizer.zero_grad
  → forward
  → loss
  → backward
  → optimizer.step
  → CUDA synchronize
```

这是**隔离的完整 step latency**，会阻止跨 step 的 host/device pipeline；适合建立稳定基线，但不等于生产 trainer 的自然流水执行。生产测量还应增加一个“不逐 step 同步、只在测量窗口末同步”的长窗口吞吐实验。

### CUDA async 为什么会让朴素计时出错

CPU 发起 CUDA kernel 后通常立即返回。下面的代码大多只测到 launch 时间：

```python
t0 = time.perf_counter()
loss.backward()
elapsed = time.perf_counter() - t0  # GPU 可能仍在执行
```

正确方法是使用 CUDA event，或在 host timer 的边界同步。同步必须在起点和终点都做：起点排除之前遗留的工作，终点等待本次工作完成。不要把 `loss.item()` 这种隐式同步偷偷混入某一版本。

### latency 应报告什么

至少保留：

- 原始 step 样本；
- mean、p50、p90、p99、min、max、标准差；
- warmup 次数、有效样本数、异常样本是否剔除及规则；
- shape、batch、dtype、seed、软件和设备元数据。

p99 是样本分位数，不是少量 iteration 就能宣称的稳定 SLO。尾延迟常来自 DataLoader 抖动、Python GC、日志、allocator 扩容、其他进程争用、网络重传或 straggler；只看平均值会掩盖这些问题。

---

## 2. Throughput：分子必须和 step 边界一致

全局吞吐定义为：

\[
\text{samples/s}=\frac{B_{global}}{t_{step}}, \qquad
\text{tokens/s}=\frac{T_{global}}{t_{step}}
\]

其中 `global` 很重要。数据并行时：

\[
B_{global}=B_{micro}\times \text{gradient accumulation steps}\times N_{DP}
\]

视频/DiT 场景必须说明 token 定义：是 text token、latent spatial token、spatiotemporal token，还是每秒视频帧。改变分辨率、帧数、序列长度或 packing 后，samples/s 可能上升但总计算反而下降；因此必须同时报告 workload shape 与 tokens/s。

吞吐最好用整个稳定窗口的总样本数除以总墙钟时间，而不是对每步 `1/t` 取平均。后者会因非线性产生偏差。本模块用 mean step time 计算吞吐，并同时保留所有 latency。

---

## 3. FLOP 口径：MFU 争议通常先是记账争议

### 3.1 基础约定

本工程默认约定：

- 一次 fused multiply-add（`a*b+c`）计 **2 FLOPs**；
- FLOP 是算法运算量估计，不是 kernel 数、Tensor Core 指令数或 GPU cycle；
- MFU 分子是一次标准 forward + backward 的**模型 FLOPs**；
- HFU 分子在相同模型 FLOPs 上加入实际声明的 activation recomputation；
- optimizer、通信、数据搬运不会自动计入模型 FLOPs；
- 非 matmul 工作只有在明确建模或 profiler 统计时才加入，禁止含糊地“补一个系数”。

跨报告比较前必须确认 FMA、forward/backward、attention、embedding、loss、optimizer、checkpoint recompute 的口径完全相同。

### 3.2 Dense Transformer 的 `6PT` 是近似，不是定律

令参数量为 `P`、本 step 有效 token 数为 `T`。对参数化 dense matmul，forward 近似：

\[
F_{fwd,param}\approx 2PT
\]

matmul backward 通常包含 input gradient 和 weight gradient，约是 forward 的两倍：

\[
F_{bwd,param}\approx 4PT
\]

所以不做 activation checkpoint 时：

\[
F_{model}\approx 6PT
\]

这个近似没有自动覆盖：

- attention 的 `QK^T` 与 `AV` 二次项；
- softmax、normalization、RoPE、激活函数；
- embedding、loss、optimizer；
- 稀疏/MoE 的实际 routed token；
- padding、sequence packing 与无效 token；
- kernel 为对齐而执行的额外运算。

以 dense self-attention 为例，仅 `QK^T` 和 `AV` 的 forward 项约为：

\[
F_{attn,fwd}\approx 4\,B\,L\,S^2\,H
\]

其中 `L` 是层数。长上下文或视频 token 下，它不能继续被 `2PT` 忽略。应按模型结构加入显式 `extra_forward_flops`，或使用经过验证的模型 FLOP 计算器，并把版本和公式写入结果。

### 3.3 Activation checkpointing：MFU 与 HFU 分子不同

令被重算的 forward FLOP 比例为 `r`，`0≤r≤1`。本项目采用：

\[
F_{MFU numerator}=F_{fwd}+F_{bwd}
\]

\[
F_{HFU numerator}=F_{fwd}+F_{bwd}+rF_{fwd}
\]

在 `2PT/4PT` 近似下：

```text
无 checkpoint：model = 6PT，hardware estimate = 6PT
全量重算：      model = 6PT，hardware estimate = 8PT
部分重算：      model = 6PT，hardware estimate = (6 + 2r)PT
```

因此 checkpoint 后 step 变慢但 HFU 上升并不矛盾：GPU 执行了更多重算。MFU 保持“完成同一模型训练语义的有效工作”口径，更适合比较 checkpoint 策略；HFU 反映设备被多少算术工作占用。`r` 必须按 forward FLOPs 加权，不能直接用“checkpoint 了多少层”的比例代替。

有些文献把 HFU 定义为 profiler 实测 FLOPs/峰值，有些使用上述算法估计。报告必须写清是哪一种。本模块输出的是 **estimated hardware FLOPs**，不冒充指令级实测。

---

## 4. MFU / HFU：峰值分母不能猜

\[
MFU=\frac{F_{model}/t_{step}}{N\times Peak_{device}}
\]

\[
HFU=\frac{F_{hardware}/t_{step}}{N\times Peak_{device}}
\]

`Peak_device` 必须与以下条件匹配：

- 设备的准确 SKU，而不只是家族名；
- FP32 / TF32 / FP16 / BF16 / FP8 等实际计算 dtype；
- dense 还是结构化 sparse 峰值；
- Tensor Core 是否真的被使用；
- 当前功耗、时钟和 MIG/分区模式；
- 厂商规格中 FLOP/FMA 的计数口径。

**禁止看到设备名就默认 A100 的 BF16 峰值。** 自动发现设备型号并不能可靠推导当前运行模式的峰值。本模块只有显式传入经过查证的 `--peak-tflops` 才计算 MFU/HFU，否则返回 `null` 和原因。

若 MFU > 100%，优先排查：

1. global token/batch 重复乘了 DP 或 accumulation；
2. step time 没有同步；
3. 峰值 dtype/稀疏模式不匹配；
4. FMA 口径不一致；
5. parameter count 或 attention FLOP 被重复计算；
6. 多 GPU 分母没有乘实际参与设备数。

MFU 也不是跨模型的唯一效率标准：非 GEMM 比例高、形状小、稀疏路由、低 arithmetic intensity 的模型，其可达到 roof 本来就低于 dense Transformer。

---

## 5. GPU utilization 不等于 MFU

`nvidia-smi utilization.gpu`/NVML 通常是在采样窗口中“GPU 上有 kernel 活跃”的时间比例。它不告诉你每个周期做了多少有效 FLOP，也不证明 Tensor Core 饱和。

```text
GPU utilization 高 + MFU 高：可能是 compute-bound 且形状良好
GPU utilization 高 + MFU 低：可能是 memory-bound、小 kernel、低 occupancy、通信 kernel
GPU utilization 低 + CPU 忙：可能是 DataLoader/launch/Python/编译/日志瓶颈
GPU utilization 锯齿：      可能是同步、I/O、周期性工作或 straggler
```

一个持续执行低带宽效率的 elementwise kernel 也能让 GPU utilization 接近 100%，同时 MFU 很低。反过来，短而高效的 burst 中 Tensor Core 很忙，但 burst 之间 CPU launch gap 很大，平均 GPU utilization 和 MFU都会低。

生产 dashboard 应同时看：step time、throughput、MFU、功耗/时钟、SM active、Tensor Core/compute pipeline、DRAM 吞吐、GPU active union、CPU launch gap 和通信暴露时间。

---

## 6. Roofline：判断“低 MFU 是否合理”

算术强度：

\[
AI=\frac{FLOPs}{Bytes\ moved}
\]

Roofline 上限：

\[
Performance\le \min(Peak_{compute},\ AI\times Peak_{bandwidth})
\]

ridge point 为 `Peak_compute / Peak_bandwidth`：

- `AI` 小于 ridge point，kernel 倾向 **memory-bound**；
- `AI` 大于 ridge point，才有机会 **compute-bound**。

这里的 bytes 必须说明层级：DRAM、L2、shared memory 的 roof 不同。理想公式中的 tensor bytes 不等于 Nsight Compute 观测到的 DRAM bytes；cache miss、重复读取、临时 tensor、写回和非合并访问都会改变它。

训练 step 是多个 kernel 的混合，不能只用整个模型的 FLOP/参数字节就宣称 compute-bound。正确流程是：Nsight Systems 找关键路径热点，再用 Nsight Compute 对代表 kernel 查看 FLOP/byte、DRAM/L2 throughput、Tensor Core、occupancy 和 stall reason。

### 五类瓶颈的证据

| 类型 | timeline / counter 现象 | 下一步验证 |
|---|---|---|
| launch-bound | 大量短 kernel，CPU launch 紧邻，GPU 间有微小空洞 | kernel 数、median kernel duration、CUDA Graph/compile 对照 |
| memory-bound | GPU 持续忙，DRAM/L2 接近可达上限，compute pipeline 低 | NCU roofline、bytes、访问合并、融合前后 |
| compute-bound | 长 GEMM，Tensor Core/compute throughput 接近可达上限 | shape、tile、occupancy、时钟、precision 对照 |
| synchronization-bound | `cudaDeviceSynchronize`、event/stream wait 后出现空洞 | 找同步调用栈，删除/移动同步做 A/B |
| CPU-bound | GPU 长空洞，CPU decode/Python/launch 工作位于关键路径 | CPU sampling、DataLoader queue、worker/批处理对照 |

“接近可达上限”不一定等于厂商理论峰值；受 shape、功耗、时钟、指令混合和 occupancy 影响，应先用相同硬件上的代表性 microbenchmark 建立 empirical roof。

---

## 7. 显存：tensor live bytes、allocated、reserved 不是同一个数

显存至少拆成：

```text
parameters
+ gradients
+ optimizer states
+ FP32 master parameters（若存在且不与别的副本重合）
+ saved activations
+ temporary workspace / communication buffers
+ allocator slack / fragmentation
+ CUDA context、library 与非 PyTorch allocation
```

仅 tensor storage 的基本公式是：

\[
Memory=numel\times bytes(dtype)
\]

一个**明确声明**的 mixed-precision Adam 示例可能是：BF16 parameter 2 B、BF16 gradient 2 B、FP32 master 4 B、两个 FP32 moments 8 B，共 `16 B/parameter`，尚未含 activation 和临时 buffer。但不同框架可能保持 FP32 gradient、复用参数副本、使用 fused optimizer 或量化 state，不能把 16 B/P 当默认真理。代码因此要求逐项显式传入。

PyTorch allocator 指标：

- `memory_allocated`：当前被 live tensor 占用的 managed bytes；
- `max_memory_allocated`：测量窗口内 allocated 峰值；
- `memory_reserved`：caching allocator 向 CUDA 保留的内存；
- `max_memory_reserved`：reserved 峰值。

`reserved - allocated` 不是纯碎片：还包含可复用 cache。`nvidia-smi` 进程显存通常还包含 context、库 workspace 和其他非 allocator 内存。OOM 排查要同时保存 memory snapshot/timeline、shape、发生阶段和各指标，不能只看 step 末尾 allocated。

activation 往往随 `B × S × H × layers` 增长，attention 中间量在朴素实现下可有 `S²` 项。peak 取决于 tensor 生命周期与重叠，而不是所有组件静态相加；activation checkpoint 通过重算缩短保存生命周期，代价是额外 FLOPs。

---

## 8. GPU bubble：要算 interval union，不能相加 kernel duration

定义 step 窗口内，至少一个相关 GPU kernel 活跃的时间并集为 `T_active_union`：

\[
T_{bubble}=T_{step}-T_{active\_union}
\]

\[
Bubble\ ratio=\frac{T_{bubble}}{T_{step}}
\]

多个 stream 能并发，直接把每个 kernel duration 相加可能超过 step time。必须在同一设备 timeline 上合并区间。代码中的 `analyze_timeline` 正是做 interval union/intersection。

单卡 bubble 可能来自 CPU/DataLoader、launch gap、同步、allocator、日志和 H2D。多卡还会加入 collective wait、pipeline bubble、负载不均和 straggler。

### NCCL 总时间为什么不等于 step 增量

通信 interval 若与 backward compute 重叠，其 duration 仍会被 profiler 计入 NCCL 总时间，但不一定增加同等 step time。可先计算：

```text
communication overlap = compute intervals ∩ communication intervals
unhidden communication = communication union - overlap
```

即便 `unhidden communication` 也不是严格因果增量：它可能与其他 stream 工作重叠，或者 compute 依赖通信而改变调度。真正的 **exposed communication time** 是通信位于关键路径、无法被有效计算隐藏的部分；最好结合 dependency timeline，并用 bucket/collective 禁用或延迟的受控实验验证。绝不能把各 rank、各 stream 的 NCCL duration 求和后直接从 step time 中相减。

---

## 9. CPU 和 GPU 各自在做什么

在本模块的训练 step 中：

- CPU/Python：进入 module、调度 ATen op、launch kernel、维护 autograd/optimizer；
- GPU：执行 Linear/GELU/loss、backward 和 SGD kernel；
- parameters/gradients/optimizer state：由所选 device 上的 PyTorch tensor 持有；
- activations：forward 在 GPU 产生，供 backward 使用后释放；
- 通信：单进程单卡实验没有 NCCL collective；H2D 也被 synthetic GPU input 排除。

这使实验专注于指标和 GPU execution。它**不能**证明真实 DataLoader、PCIe、NCCL 或多机行为。后续模块必须把这些路径逐一加入，同时保持相同 measurement contract。

---

## 10. Benchmark 与 profiler 的使用顺序

### Correctness

```bash
PYTHONPATH=Work/src python -m unittest discover -s Work/src/metrics/tests -v
PYTHONPATH=Work/src python -m metrics.correctness --device cuda --candidate compiled
```

第二条比较 eager 与 compiled 的 forward、loss 和一次 optimizer update。容差随 dtype 声明。候选不正确就停止性能比较。

### Baseline / optimized benchmark

```bash
PYTHONPATH=Work/src python -m metrics.benchmark \
  --device cuda --variant both --warmup 10 --iterations 100 \
  --output /tmp/metrics_benchmark.json
```

输出含环境、raw latency、p50/p90/p99、samples/s、tokens/s、FLOP 口径、peak allocated/reserved。未提供可靠 `--peak-tflops` 时 MFU/HFU 为 `null`，这是正确结果，不是缺陷。

应该观察：

1. warmup 已承担编译、lazy initialization、allocator growth；
2. eager/compiled 结果使用相同 shape/dtype/step 边界；
3. p50 与 tail 是否稳定，raw sample 是否有周期性异常；
4. compiled 的 kernel/launch 改变是否真的转化为 step/throughput 改变；
5. peak allocated 与 reserved 是否改变；
6. 不把一次运行的随机差异写成结论，至少独立重复并报告方差。

### PyTorch Profiler

```bash
PYTHONPATH=Work/src python -m metrics.profile --backend torch \
  --device cuda --steps 10 --trace /tmp/metrics_trace.json
```

先用 operator table 定位 CPU/CUDA 时间和 allocation，再打开 Chrome trace 看 CPU launch、stream、kernel、同步与空洞。Profiler 会扰动执行，profile run 用于结构诊断，基准数字仍来自低扰动 benchmark。

### Nsight Systems

用 `profile.py --backend nvtx`，筛选 `metrics_measured_region` 和 `train_step`。应该看到：

- CPU op/launch 与 CUDA kernel 的对应关系；
- step 间和 step 内 GPU 空洞；
- Linear 对应的 GEMM、elementwise、reduction、optimizer kernel；
- eager 与 compiled 的 kernel 数和 launch gap 是否变化；
- 是否有意外同步。

### Nsight Compute

只选 Nsight Systems 已确认的热点 kernel。至少检查：Tensor Core/compute pipeline、DRAM/L2 throughput、achieved occupancy、registers/thread、shared memory/block、warp stall reasons、grid size 和 memory access pattern。NCU 重放会显著改变时间，不能把 NCU 下的 end-to-end step time 当生产 benchmark。

完整命令见 `src/metrics/README.md`。

---

## 11. 8 GPU、128 GPU、1024 GPU 的 scaling 预留

全局吞吐扩展效率：

\[
Efficiency(N)=\frac{Throughput_N/Throughput_{ref}}{N/N_{ref}}
\]

报告时必须声明：

- strong scaling（固定 global workload）还是 weak scaling（固定 per-device workload）；
- global/per-device batch、accumulation 和收敛语义；
- precision、并行策略、网络拓扑、rank placement；
- min/median/max rank step time 和 collective tail；
- compute/communication overlap 与 exposed communication；
- 数据、checkpoint 和控制面的规模效应。

8 GPU 单机可能主要走 NVLink/NVSwitch，collective latency 较低；128/1024 GPU 会跨节点、跨交换层，网络拥塞、拓扑不匹配、慢 rank、collective tail、启动/元数据服务和故障率都会放大。一个 rank 慢就可能让同步 step 的所有 rank 等待。因此单卡 MFU 高或 8 卡 scaling 好，不能外推到千卡。

当前 `scaling_from_throughput` 只提供口径正确的计算函数；本阶段没有多卡实测，明确不提交 scaling 数字。

---

## 12. 工业排查顺序

吞吐突然下降时，按下面顺序保存第一现场并缩小范围：

1. **确认任务语义**：commit、config、shape、batch、dtype、accumulation 是否变化。
2. **确认指标正确**：step 边界、同步、global 分子、FLOP/peak 口径是否一致。
3. **看时间序列**：何时开始、所有 rank 还是局部、是否周期性、是否伴随功耗/时钟/显存变化。
4. **拆 end-to-end**：data wait、H2D、forward、backward、optimizer、communication、checkpoint。
5. **Nsight Systems 看关键路径**：CPU gap、GPU bubble、同步、NCCL overlap、straggler。
6. **热点下钻**：代表 kernel 用 NCU，CPU/I/O 用相应 sampling 与队列指标。
7. **提出一个可证伪假设**：例如“短 kernel launch-bound”，而不是“GPU 有问题”。
8. **单变量优化并重测**：同一 correctness、shape、环境和统计方法。
9. **稳定性验证**：长跑、tail、peak memory、数值、故障恢复，不能只看 20 个好看的 step。

典型错误包括：未同步得到虚假 speedup、把 GPU utilization 当 MFU、拿稀疏峰值计算 dense MFU、把 reserved 当 live tensor、把 NCCL 总时间当 exposed time、profile 结果与 benchmark 结果来自不同 workload。

---

## 13. 每次优化的证据模板

```text
Workload identity:
  commit / config / model shape / data shape / dtype / seed
Hardware & software:
  GPU SKU / clocks & power / driver / CUDA / PyTorch / topology
Metric conventions:
  step boundary / global batch & tokens / FLOP formula / verified peak source
Baseline:
  raw latency / p50 p90 p99 / throughput / MFU-HFU or null / peak memory
Profile evidence:
  active union / bubble / kernel count / hotspot / exposed communication
Hypothesis:
  one bottleneck, predicted profiler and metric change
Change:
  code/config delta
Optimized:
  same measurements and profile
Correctness & stability:
  numerical tolerance / repeated runs / long-run tail / OOM or fault behavior
Conclusion:
  speedup with variance; whether hypothesis was supported; remaining bottleneck
```

只有完成这个闭环，“更快”才是工程结论。
