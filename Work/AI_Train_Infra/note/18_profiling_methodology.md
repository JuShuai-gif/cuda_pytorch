# 18｜训练系统 Profiling 方法论：从时间线到可证伪优化

## 本模块解决的问题

训练变慢时，“GPU utilization 只有 60%”不是根因。它既可能是 CPU 没及时提交，也可能是大量短 kernel、显式同步、内存延迟、数据输入空洞，或计算/通信不能 overlap。本模块建立一条工业可复现的证据链：

```text
定义问题与 step 边界
  → 无 profiler 的 baseline
  → PyTorch Profiler 建立语义映射
  → Nsight Systems 确认 CPU/GPU 时间线和真实热点
  → NCU basic 检查一个代表 launch
  → 按问题升级 detailed / source / full
  → 一次只做一个优化
  → 相同条件重新 benchmark/profile
  → correctness、性能和稳定性共同验收
```

Profiler 是测量仪器，不是答案生成器。任何“memory-bound”“overlap 很好”“快了 20%”都必须绑定 workload、范围、指标值和原始报告。本仓库不记录未经运行的数字。

## 1. 先把测量对象说清楚

一个训练 step 的墙钟时间可粗略分解为关键路径上的：

```text
T_step = T_cpu_exposed + T_h2d_exposed + T_gpu_compute
       + T_comm_exposed + T_sync + T_io_exposed + T_checkpoint_exposed
```

这里不能把每项 profiler duration 直接相加，因为 compute、copy、NCCL、CPU 准备可能重叠。真正影响 step 的是关键路径上的 **exposed time**。同理，所有 NCCL kernel 的总时长不等于 NCCL 增加的 step time：藏在 backward compute 下的通信不是完全 exposed；只有延长关键路径的部分才是。

测量前冻结：commit、模型/shape、batch 和 sequence、dtype、gradient accumulation、compile/CUDA Graph、checkpoint、数据源、warmup、测量 step、GPU 电源/温度、其他进程。明确 step 从哪里开始和结束；不要一会儿包含 DataLoader，一会儿只量 forward/backward。

必须同时保留：

- wall-clock step time 与 raw samples（不能只有均值）；报告 p50/p90/p99 和抖动。
- samples/s 或 tokens/s；工作量定义不能在 A/B 中改变。
- MFU/HFU 的 FLOPs 口径、硬件峰值来源和精度模式。
- GPU active time、idle/bubble、kernel 数与 duration 分布。
- peak allocated/reserved memory；必要时 memory snapshot。
- 分布式场景的 NCCL total 与 exposed communication、scaling efficiency。

## 2. CPU/GPU timeline 与 CUDA async

PyTorch 的 CUDA op 通常是异步的：CPU 做 Python/C++ dispatcher、shape/依赖检查、allocator、kernel launch 和下一批数据准备；GPU 按 stream 顺序执行 kernel/copy。CPU 调用返回只说明工作已入队，不说明 GPU 已完成。

因此下面的 CPU 计时是错的：

```python
t0 = time.perf_counter()
y = model(x)
elapsed = time.perf_counter() - t0  # 多半只测到 enqueue
```

隔离 latency 可在边界同步，或用同一 stream 上的 CUDA events；完整训练吞吐则应连续运行多个 steady-state step，只在测量窗口末端同步。每 invocation 都同步会破坏异步提交和 overlap，所以本模块的 `benchmark.py` 只用于 isolated latency，真实 timeline 使用连续的 `profile_target.py`。

GPU bubble 是定义好的 step 范围内“GPU 本可做关键路径工作却没有工作”的时间片，不能简单用 `100% - nvidia-smi GPU-Util` 推导。`nvidia-smi` 是粗采样的 busy 指示器：一个低效 kernel 持续运行也可显示高 utilization；GPU 很忙不代表接近 Tensor Core 峰值，更不代表高 MFU。

常见强制同步包括 `.item()`、把 CUDA tensor 打印/拷回 CPU、同步 H2D、`torch.cuda.synchronize()`、某些 allocator/错误检查、跨 stream 依赖。要从 timeline 证明它位于关键路径，而不是看到一个 sync 名字就删除。

## 3. 三层 profiler 分工

### 3.1 PyTorch Profiler：谁在调用

它把 Python/ATen/autograd/module 语义关联到 CUDA kernel，适合回答：哪个 op 最重、forward/backward 如何组成 step、allocation 来自哪里、DataLoader/CPU 是否形成空洞。

使用 schedule 跳过冷启动并只采 active window：

```bash
cd /home/guhaoran/code/cuda_pytorch/Work/src/profiling
scripts/run_torch_profiler.sh --device cuda --case launch --variant baseline \
  --wait 1 --warmup 1 --active 3
```

先看 self CPU/CUDA time、call count、shape，再看 trace 中 step/N、op、kernel 和 memcpy。`record_shapes`、`profile_memory`、`with_stack` 会增加开销；只在相应问题需要时打开。Profiler 下的绝对 latency 不是最终性能数字，结论须回到无 profiler benchmark。

### 3.2 Nsight Systems：什么时候执行，关键路径在哪里

Nsight Systems 是训练系统诊断的第一现场。观察：

- CPU thread 是否持续提交；CUDA API 与 GPU launch 间是否有长空洞。
- kernel 是否密集但过短（launch-bound），还是存在长 GEMM/attention。
- memcpy、compute、NCCL 是否位于不同 stream，依赖是否允许 overlap。
- `cudaDeviceSynchronize`/stream wait 是否把流水线串行化。
- backward compute 与 NCCL 是否真正重叠；通信尾巴延长 step 多少。
- NVTX step、forward、backward、optimizer、input、checkpoint 范围是否闭合。

```bash
scripts/run_nsys.sh --device cuda --case launch --variant baseline
scripts/run_nsys.sh --device cuda --case launch --variant optimized
```

先用它选中一个代表性 kernel，记录 demangled name、launch index、duration、grid/block、所在 step、shape/precision、是否 graph node。对 CUDA Graph 要保持 A/B 的 graph mode 相同，需要 node 粒度时使用 `--cuda-graph-trace=node`。

### 3.3 Nsight Compute：一个 kernel 为什么这样执行

不要对完整训练直接 `--set full`。NCU 会 replay kernel，成本高且可能扰动上下文；它回答 kernel 内部问题，不能替代 end-to-end timeline。

本项目强制顺序：

1. Nsight Systems 证明热点及其 end-to-end 占比。
2. NCU `basic`：duration、launch geometry、occupancy、SpeedOfLight、work distribution。
3. 针对问题才用 `detailed` 看 scheduler/memory；用 source counters 做 source/SASS 归因；`full` 仅在 replay 成本合理时。
4. 优化后同时重跑 NCU 和无 profiler end-to-end benchmark。

本机 NCU 2025.3 没有 `source` set，因此仓库 `--stage source` 实际运行 `--set detailed --section SourceCounters`；source attribution 需要 `-lineinfo`。版本不同先运行 `ncu --list-sets`/`--list-sections`，不能假设命令和 metric 名恒定。

优先检查且以本报告实际存在为准：

| 维度 | 证据 |
|---|---|
| duration/launch | `gpu__time_duration.sum`、grid/block、waves/SM |
| 资源 | registers/thread、shared memory/block、cluster、occupancy limit |
| 并发 | achieved vs theoretical occupancy、eligible/active warps |
| compute | SM/pipe/tensor 指标与执行指令；和同 shape baseline 比 |
| memory | DRAM/L2/L1 throughput、hit rate、sector/request、local load/store |
| stall | long/short scoreboard、barrier、wait、not-selected 等实际可用项 |
| 分布 | per-SM active cycles、partial wave、PM sampling tail |

Thor/sm_110 必须现场查询 SM 数、resident blocks、register/shared memory、cluster 和 metric availability。不能套 B200 的峰值阈值、launch tile 或“Tensor Core 超过某百分比才合格”。统一内存还要求固定 CPU traffic、page migration、电源与散热。先 `action.metric_names()`；缺失 metric 是 unavailable/null，不是数值 0，也不能用无关 counter 替代。极短 kernel 的 PM sample 可能不足，同样要标记不可用并用 aggregate/timeline 交叉验证。

## 4. 五类瓶颈如何下结论

| 类型 | 必要证据组合 | Nsight Systems 预期 | NCU/Profiler 关注 | 优化后应变化 |
|---|---|---|---|---|
| launch-bound | 大量短 kernel；host launch/dispatch 占关键路径；非 DRAM/compute 饱和 | CUDA API 与成串小 kernel，可能 GPU 等 CPU | call count、kernel duration；超短 kernel 通常没必要先 full NCU | fusion/batching/compile 后 kernel 数和 gap 降，step time 降 |
| memory-bound | arithmetic intensity 低且实测 memory subsystem 接近该 workload 的 roof；stall/access 证据一致 | 长 elementwise/reduction，GPU 持续 active | DRAM/L2/L1、bytes、sector/request、long scoreboard | fusion/reuse 后 bytes 与 duration 降；不能只看 occupancy |
| compute-bound | 计算 pipe 为主要限制，矩阵 shape/precision 适合，memory 非主限 | 长 GEMM/attention 占主导 | tensor/SM pipe、tile、指令、occupancy/resource | 更合适 dtype/library/layout 后 kernel 变短且正确性保持 |
| synchronization-bound | 同步/依赖直接造成关键路径空洞 | sync API、stream wait、串行 copy/compute | CPU self time、barrier stalls；先区分 host sync 与 kernel barrier | 不必要 sync 消失、overlap 增加、bubble 降 |
| CPU-bound | GPU 空洞前 CPU 没提交；CPU/DataLoader 范围占 exposed time | Python/DataLoader/decode gap、CUDA queue 饿死 | CPU stacks、OS runtime、workers；NCU 通常不是第一工具 | CPU gap/GPU idle 降，连续 submission 增加 |

单个信号不充分。例如 high long-scoreboard + low DRAM bandwidth 常是 memory latency/ILP 不足，不是 bandwidth-bound；低 occupancy 也可能是设计选择。`not_selected` 往往表示有可调度并行度，不是坏事。诊断必须把 timeline、SpeedOfLight、stall、access pattern 和源码对应起来。

Roofline 只提供上限：

```text
arithmetic_intensity = useful FLOPs / bytes_from_relevant_memory_level
attainable = min(compute_peak_for_dtype, bandwidth × arithmetic_intensity)
```

bytes 必须说明是算法估计、profiler counter 还是 DRAM/L2 哪一层；FLOPs 要说明 forward-only 或 training，是否含 recompute。低于 roofline 可能来自 launch、latency、依赖、occupancy、tail 或 sync，不能自动归为 memory-bound。

## 5. A/B 证据闭环与验收

一次合格优化实验：

1. correctness test 先通过；训练还需 loss/gradient/参数更新和数值容差。
2. baseline 至少保存 raw sample、p50/p90/p99、timeline、环境和命令。
3. profiler 找到一个最大且可行动的 exposed bottleneck。
4. 写下假设：“改 X 会经机制 Y 使指标 Z 改变”；一次只改一个主变量。
5. optimized 在完全相同条件重跑；随机化/交替 A/B，避免温度和时钟漂移。
6. 同时比较 step time、throughput、MFU（口径不变）、GPU active/bubble、NCCL exposed、peak memory、kernel count/duration 和 scaling efficiency。
7. 多次独立 run 给分布/置信范围；检查长时间稳定性和回归 shape。

Kernel speedup 的 end-to-end 上限受热点占比约束。若热点只占 step 的一小部分，即使 kernel 快很多，step 收益也有限；必须回到 Nsight Systems/benchmark 实测，不能把 NCU rule 的 estimated speedup 当成最终收益，也不能把多个 rule estimate 相加。

## 6. 当前实验应该看到什么

`src/profiling` 的五个 workload 用于建立视觉记忆：

- `launch/baseline` 是多个小 add；optimized 是一个 add。预期 kernel/call count 减少。
- `memory/baseline` 有中间 tensor 和两次 pass；optimized 是单次 affine pass。预期 kernel、allocation/bytes 下降。
- `compute/baseline` 重复 GEMM；optimized 做一次 GEMM 再 scale。预期 GEMM 次数和总 GPU time 降。这是教学用公共子表达式消除，不代表生产模型总能这样改。
- `sync/baseline` 每次 launch 后同步；optimized 只在窗口末等待。预期 host/device serialization 消失。
- `cpu/baseline` 故意 sleep 模拟 decode/input/control gap；optimized 去掉 gap。预期 GPU submission 空洞消失。

这些 signature 必须在本机实际 trace 中验证；若没有 CUDA/Nsight，只运行 CPU correctness/benchmark，并把 CUDA、NCU、Tensor Core、显存、NVLink/NCCL 部分标为未验证，提供真实机器命令而不填数字。

## 7. 规模扩大为何结论会变

单 GPU 优化减少 0.5 ms，不保证 1024 GPU 同样获益。8 GPU 时通信可能被 backward compute 隐藏；128/1024 GPU 时更小 local batch、collective latency、拓扑跨域、straggler 和同步尾部会放大，热点从 compute 转成 exposed communication/CPU launch。大规模分析必须给每个 rank 对齐 step/NVTX，比较快慢 rank，分层看 intra-node 与 inter-node，并以关键路径 exposed time 衡量 NCCL。

参数、梯度、activation、optimizer state 的“在哪里”不能从 GPU utilization 推断。需结合框架状态、memory snapshot、FSDP/TP/PP 策略和 timeline；当前单 GPU workload 没有 NCCL，也不产生分布式通信，不能用它声称多卡 overlap 或 scaling 数据。

## 8. 生产排障顺序

1. 保存第一现场：job/rank/host、commit/config、最近 step、错误、GPU/网络/系统日志，不要先重启抹掉证据。
2. 判断 correctness/stability 还是纯性能；确认 step 定义和吞吐是否真的变化。
3. 查环境与输入是否变了：shape、数据、dtype、compile/graph、电源/温度、共租户。
4. 应用级指标定位时间区间和异常 rank；看 p50/p99，而非单步。
5. PyTorch Profiler 做 op/模块语义映射；Nsight Systems 找 exposed 关键路径。
6. 只对已证明热点用 NCU basic，按问题升级 counter/source。
7. 形成可证伪假设，最小改动 A/B；先 correctness，后性能与长跑。
8. 保存原始报告、命令、版本与结论；未采到的指标明确写 unavailable。

## 运行入口

```bash
cd /home/guhaoran/code/cuda_pytorch/Work/src/profiling
python3 -m unittest discover -s tests -v
python3 benchmark.py --device cpu --case all --variant both
python3 profile_torch.py --device cuda --case launch --variant baseline
bash scripts/run_nsys.sh --device cuda --case launch --variant baseline
bash scripts/run_ncu.sh --kernel 'HOTSPOT_REGEX' --stage basic -- APPLICATION ARGS...
```

具体参数、artifact 布局、NCU source 兼容规则见 `src/profiling/README.md`。
