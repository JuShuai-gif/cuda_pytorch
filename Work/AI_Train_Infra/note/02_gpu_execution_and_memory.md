# 02｜GPU 执行、CUDA 异步、显存与 Roofline：从 timeline 判断瓶颈

## 本模块解决的问题

训练 step 变慢时，第一件事不是“打开某个优化开关”，而是把端到端时间分成 CPU 供给、GPU kernel、内存搬运、同步等待和空洞，并回答：临界路径在哪里，理论上限是什么，哪项改动真正缩短了 step。

本模块建立以下最小闭环：

```text
成对正确实现 → 同 shape/dtype benchmark → timeline 定位 → kernel counter 验证
             → 实施候选优化 → 重新 benchmark/profile → 检查正确性与显存
```

配套代码见 `src/gpu_basics/`。仓库不保存或声称任何未实际测得的性能数字。

## 1. GPU utilization 为什么不等于 MFU

常见的 GPU utilization 是采样窗口中 GPU 是否至少有一个 kernel 在执行的比例。一个只跑 3% 峰值 FLOPs 的低效 kernel，也可让 utilization 接近 100%；反过来，短而高效的 Tensor Core burst 与 CPU gap 交替，utilization 可能不高。

- **GPU utilization**：设备“忙不忙”的采样信号，不能说明忙得是否有效。
- **MFU**：模型一次迭代按约定计算的有效 FLOPs，除以 step time 和硬件该精度峰值 FLOP/s。它是模型级效率。
- **HFU**：实际执行 FLOPs（包括 activation recompute 等）除以同一峰值。checkpointing 后 HFU 可能高于 MFU，因为额外计算不属于模型有效 FLOPs。
- **带宽效率**：实际或估算 bytes/s 除以可持续显存带宽。它回答 memory-bound 工作，不应拿 MFU 代替。

因此诊断至少同时保留 step time、samples/tokens/s、模型 FLOPs、MFU、GPU active/idle、kernel/NCCL 临界路径和 peak memory。单 GPU 本模块没有 NCCL；该字段应标为不适用，而非填 0 冒充测量。

## 2. 一次 PyTorch CUDA op 底层发生什么

以 `y = relu(x @ w + b)` 为例：

1. Python/dispatcher 检查 dtype、shape、device，选择 ATen/CUDA 实现。
2. CPU 可能调用 cuBLAS、请求 caching allocator 块，并把 kernel/memcpy 提交到某条 CUDA stream。
3. launch 返回通常只表示工作已入队；CPU 可以继续准备下一 op。
4. GPU front end 调度 thread block 到 SM。warp 发射 Tensor Core/FP/LDST 指令；数据经 HBM→L2→L1/shared/register，结果写回。
5. 同一 stream 内按序，独立 stream 之间没有天然数据依赖。allocator、autograd 和 framework 会插入 event 维护部分跨 stream 生命周期。

### CPU 与 GPU timeline

```text
CPU: dispatch A | launch A | dispatch B | launch B | .item() wait........ | next
GPU:             kernel A | kernel B | idle while CPU is blocked/prepares |
```

CPU 快于 GPU 时形成深队列，host enqueue 时间很短但 GPU 仍在运行。CPU 慢于 GPU 时队列耗尽，timeline 出现 GPU bubble。`.item()`、`.cpu()`、打印 CUDA tensor、同步 D2H、显式 `synchronize()` 和某些动态 shape 控制流会建立 CPU←GPU 依赖，截断流水。

### stream 与 event 的正确性

同一 stream 中事件只覆盖该 stream 在两事件之间的工作：

```python
torch.cuda.synchronize()
start.record()
work()
end.record()
end.synchronize()
elapsed_ms = start.elapsed_time(end)
```

跨 stream 不能仅在 default stream 记录 end event，否则 side stream 可能尚未结束。必须让计时 end 所在 stream `wait_event(done_on_side_stream)`，或最终全设备同步。生产 benchmark 同时报告：

- **同步 wall time**：包含 Python/launch/同步和 GPU 临界路径，最接近 step latency；
- **CUDA Event time**：设备 timeline 范围，适合 kernel 序列；
- **未同步 host time**：只能标作 enqueue time，绝不能叫 GPU latency。

`async_timing.py` 用独立运行展示三者。同步本身会改变流水，所以只在 benchmark 样本边界同步；分析连续 step overlap 时用 trace，不在每步插同步。

## 3. GPU execution model 与性能上限

kernel 启动一个 grid，grid 有 blocks，block 有 warps（NVIDIA 每 warp 32 threads）。block 占用 SM 的寄存器和 shared memory；资源或最大 resident warp/block 数会限制 occupancy。occupancy 是隐藏延迟的条件，不是性能目标：寄存器少但发生 spill，或 occupancy 高而 DRAM 饱和，都可能更慢。

吞吐受多层约束：

- grid 太小：block 数不足，许多 SM 无工作；
- 单 block 资源过大：resident blocks 少；
- warp divergence：分支路径串行化；
- 不合并访存：更多 memory transaction；
- 数据依赖/指令延迟：eligible warp 不足；
- launch 数多且 kernel 极短：CPU/CUDA front-end 成为上限；
- GEMM shape、layout 或 dtype 不满足 Tensor Core 快路径。

Nsight Compute 中把 `achieved occupancy` 与 registers/thread、shared memory/block、grid size 一起看；warp stall reason 是线索，不能脱离“发出的有效 work”和吞吐单独优化。

## 4. 显存到底由什么组成

训练进程可见显存近似为：

```text
parameters + gradients + optimizer states + saved activations
+ temporary workspaces + communication buffers + allocator fragmentation
+ CUDA context/library/module/graph memory + other framework allocations
```

若 P 个参数均为 BF16，模型参数约 `2P` bytes；梯度若 BF16 约 `2P`，若 FP32 约 `4P`；Adam 的一阶/二阶矩通常各 `4P`，还可能有 FP32 master weights `4P`。是否存在 master weight、梯度 dtype、fused optimizer 的 state 布局必须从实际实现核对，不能机械背“16 bytes/parameter”。activation 取决于 `B×S×H×layers`、attention 中间量、autograd 保存策略、checkpointing 和 kernel 融合，常随 batch/sequence 快速增长。

PyTorch 指标含义：

- `memory_allocated`：活跃 tensor 占用；
- `memory_reserved`：caching allocator 从 CUDA 驱动保留的 segment；
- `max_memory_allocated`：复位后活跃 tensor 峰值；
- NVML/`nvidia-smi` process memory：还含 context、库和非 PyTorch CUDA 分配，通常更大。

`empty_cache()` 只释放未被活跃 tensor 使用的缓存块给驱动，不会释放仍有引用的 tensor，也不是正常 step 内的性能优化。reserved≫allocated 可能是正常缓存，也可能是尺寸变化导致碎片；应结合 memory snapshot、allocation timeline 和 OOM 信息判断。

`memory_demo.py` 同时报告参数、梯度、AdamW 首步后 state、autograd logical saved bytes 与每阶段 allocated/reserved/peak。saved bytes 可能对 alias 重复计数，allocator 差值也会受临时 buffer 重用影响，二者都是证据而非完美归因。

参数和 optimizer state 在本阶段均位于单 GPU；CPU 只持 Python 对象/元数据并提交工作。没有分布式通信。后续 DDP/FSDP 中这些对象的位置与通信量会改变。

## 5. Roofline：先建模型，再看 profiler

Arithmetic intensity（AI）定义为：

```text
AI = FLOPs / 从目标内存层传输的 bytes
性能上限 = min(峰值计算 FLOP/s, AI × 峰值带宽 byte/s)
```

ridge point 为 `peak FLOP/s ÷ peak byte/s`。AI 低于它通常落在带宽屋顶，减少 bytes/融合 pass 比增加计算单元更重要；AI 高于它才可能 compute-bound。

关键陷阱：bytes 必须说明是哪一层。算法最低 HBM bytes、L2 实测 bytes、DRAM counter 不是一回事。矩阵乘 `M×K` 乘 `K×N` 的 FLOPs 约 `2MKN`，算法最低 bytes 是读取 A/B、写 C，但 tiling/reuse 决定实际 HBM/L2 流量。pointwise 每元素少量 FLOPs 却至少读写一次，通常 AI 很低。

本模块 JSON 的 bytes 是解析下界/近似，明确带 caveat；只有用户输入对应 dtype 的峰值 FLOPs 与带宽才计算 roofline efficiency。最终结论应由 NCU 的 DRAM/L2 throughput、Tensor Core pipe 和指令数据校验。

## 6. 五类瓶颈如何区分

### Launch-bound

症状：大量微小 kernel；每个 kernel 数微秒，GPU 计算/带宽均低；CPU CUDA API 与 kernel 间隙占比高。增加 tensor 大小后吞吐显著改善，或融合后 kernel 数下降且 step 缩短。

验证：Nsight Systems 数 kernel、看 API→kernel 和 gap；NCU 通常 grid 小、各吞吐低。`launch` baseline 的多次 add 对比一次代数折叠。注意折叠改变浮点加法顺序，必须做 tolerance correctness。

### Memory-bound

症状：SM 算术 pipe 未满，DRAM 或 L2 throughput 接近该 shape 的可持续上限；AI 低。优化方向是融合、减少 materialization、连续访问、复用缓存、降低 dtype，而不是盲目增加 occupancy。

`memory` baseline 的冗余 clone 是可控病理。NCU 看 DRAM/L2 bytes 与 sectors/request；优化后应看到 pass/kernel/bytes 下降。小 tensor 可能先是 launch-bound，因此需要 size sweep。

### Compute-bound

症状：大型 GEMM/attention 占临界路径，Tensor Core/计算吞吐高，扩大计算量近似线性增加时间，而带宽未成为主上限。看实际 dtype、TF32 开关、matrix shape/layout、Tensor Core 指令、waves 和尾效应。

`gemm` 比较 `einsum` 与 `mm`；它们可能落到同一 backend、没有速度差。这正是“不凭 API 名宣称优化”的实验。小 512 shape 未必饱和，正式实验需 sweep，但逐步放大避免 OOM。

### Synchronization-bound

症状：CPU timeline 出现 `cudaStreamSynchronize`/D2H wait，GPU 队列被切断，active kernel 总和明显小于 step wall time。`sync` baseline 每轮 `.item()`；候选只在最后 `.item()`。优化后不是“kernel 总时间必降”，而应是同步次数和 exposed GPU bubble 降低。

### CPU-bound / input-bound

症状：GPU kernel 间存在长空洞，同时 CPU 在 Python、decode、DataLoader、GC、锁或调度上工作，GPU 队列深度不足。`cpu_gap` 用 sleep 构造可重复第一现场；真实系统还要看 OS runtime、DataLoader worker、H2D 和 pinned memory。

### GPU bubble 的严格定义

bubble 是观察区间内 GPU 本可执行训练关键路径工作却没有执行的时间。不能简单用 `step time - sum(kernel duration)`：并发 streams 会使 kernel duration 重叠，memcpy/NCCL 可能占设备，不同 rank 的等待也不同。应在 Systems timeline 上按临界路径和 stream 并集计算 active/exposed idle，并保留定义。

## 7. 性能证据闭环怎么做

1. 固定 commit、环境、GPU、clocks/power、shape、dtype、seed 与 warmup。
2. correctness test，明确数值 tolerance 和语义变化。
3. baseline 多次测同步 wall、Event、throughput、peak allocated，保存 mean/std/p50/p90/p99 与 raw samples/JSON。分组边界同步测的是独立调用 latency，会切断自然异步 pipeline；连续 step 的 overlap/bubble 以未逐步同步的 trace 为准。
4. PyTorch Profiler 找 op/shape/内存粗热点；其 instrumentation 会扰动时间。
5. Nsight Systems 选稳定 step：看 CPU launch、GPU stream、memcpy、同步、active/bubble、kernel 数与临界路径。
6. 用 NCU 只采代表 kernel：先 `--set basic`，有具体问题才升级 `detailed/source/full`，且每次使用唯一 report 名保存原始证据。检查 Tensor Core、DRAM/L2、occupancy、register、shared memory、warp stall、grid、访存模式；metric 缺失记 unavailable，不能填 0 或套用别的架构阈值。NCU replay 的端到端时间不可当吞吐。
7. 写出瓶颈假设和可证伪预测。例如“launch-bound，所以融合后 kernel 从 N 到 1、gap 降且 wall 降”。
8. 只改一个主要变量，重复同一 benchmark/profile。
9. 报告中并列 baseline/optimized 的 step time、throughput、有效 FLOP/s、peak memory、kernel 数/时长、GPU active/bubble。多 GPU 再加入 exposed NCCL 和 scaling efficiency。
10. 做 shape sweep、长跑稳定性、数值收敛与回归阈值；一次最快样本不是生产结论。

## 8. 8 GPU 与 128/1024 GPU 为什么会不同

本模块代码没有伪装成多 GPU 验证。单 GPU 的 launch/memory/compute 问题到集群仍存在，但规模增大后：

- rank straggler 的尾延迟被 collective 放大；
- CPU/NUMA、PCIe/NVLink/NIC 拓扑和跨节点 oversubscription 进入临界路径；
- 每 rank 小 batch 让 kernel 更小，原本 compute-bound 可能变 launch-bound；
- 通信可与 backward 重叠，NCCL kernel 总时长不等于它增加的 step time；真正要算未被计算遮蔽的 exposed communication；
- 1024 GPU 上罕见 ECC、网络抖动、进程退出和 checkpoint stall 会成为日常事件。

所以 8 GPU 优化不能只外推 speedup；必须在目标层级重测 step 分布、rank skew、通信 overlap、网络 counter、故障率和恢复时间。

## 9. 生产故障与排查顺序

常见第一现场包括吞吐突降、周期性 bubble、reserved memory 上涨、偶发 OOM、NaN、GPU hang/timeout。建议顺序：

1. **保护现场**：job/rank/host/GPU、step、配置/commit、stderr、OOM/NCCL 日志、最近 checkpoint；不要先重启抹掉证据。
2. **确认口径**：是单 step、滑窗均值还是全局最慢 rank；数据/shape 是否改变。
3. **定位层级**：所有 rank 还是单 rank/单节点；CPU、GPU、I/O、网络、checkpoint 哪条时间线异常。
4. **看 timeline 临界路径**：空洞前 CPU 在做什么、是否同步、哪个 stream/event/collective 阻塞。
5. **显存分类**：allocated/reserved/NVML、active tensor、峰值发生阶段、是否 shape/碎片/泄漏。
6. **kernel 深挖**：只在已确认热点上跑 NCU，避免一上来收全量昂贵 counter。
7. **最小复现与对照**：固定输入、关掉一个变量、与健康节点/前一版本比较。
8. **临时恢复后长期修复**：回滚/降 batch 只是恢复；最终要有监控、回归 benchmark、fault test 和 runbook。

## 10. 实验命令与应观察到什么

```bash
export PYTHONPATH="$PWD/Work/src"
python -m unittest discover -s Work/src/gpu_basics/tests -v
python -m gpu_basics.benchmark --device cuda --output /tmp/gpu_basics.json
python -m gpu_basics.async_timing --device cuda
python -m gpu_basics.memory_demo --device cuda
python -m gpu_basics.profile_workloads --device cuda --workload sync --variant both
bash Work/src/gpu_basics/profile_nsys.sh /tmp/sync --workload sync --variant both
```

应验证而非预设：launch 候选 kernel 数减少；memory 候选 copy/bytes 减少；sync 候选 host wait 次数减少；cpu_gap 候选 GPU 空洞减少；GEMM 两 API 可能相同。若机器无 CUDA，tests 与 CPU benchmark 仍可运行，但 Event、CUDA allocator、Nsight、Tensor Core/DRAM 指标全部标记未验证。

## 下一模块

在已经会区分 CPU/GPU、计算/内存/同步之后，下一步把同一证据协议应用到 PyTorch Profiler、Nsight Systems 与 Nsight Compute：定义稳定 profile window、从 op 下钻到 kernel/counter，并把 trace-only 指标自动纳入前后对比。
