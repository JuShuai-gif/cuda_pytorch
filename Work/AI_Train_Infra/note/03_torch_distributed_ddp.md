# 03｜torch.distributed 与 DDP：从梯度同步到 exposed communication

## 本模块解决的问题

本模块不以“会写 `torchrun` 命令”为完成标准，而是回答：

- DDP 每张 GPU 保存、计算和通信什么？
- backward 与 NCCL 是否真的重叠？
- 为什么 NCCL kernel 总时长不等于它增加的 step time？
- bucket 变小为什么既可能增加 overlap，也可能因为 collective 过碎而变慢？
- 如何通过 benchmark、PyTorch Profiler 和 Nsight Systems 给出可证伪结论？

当前机器只有一张 NVIDIA Thor。双进程 Gloo correctness 和单 GPU NCCL 控制路径已验证；真实多 GPU NCCL、NVLink、IB/RoCE 未验证，本文不会给出虚构数据。

## 1. 一次 DDP step 的执行过程

设模型参数量为 `P`，每元素参数/梯度字节数为 `b_p/b_g`，rank 数为 `N`。

```text
CPU                         GPU compute stream          NCCL stream
DataLoader / dispatch  ->   forward
autograd scheduling    ->   backward(layer L)
                             gradient bucket ready  ->  all-reduce(bucket)
                             backward(layer L-1)        all-reduce(bucket)
                             ...                        ...
optimizer launch       ->   wait reduced gradients
                             optimizer step
```

CPU 负责 Python 训练循环、autograd 调度、kernel/collective enqueue、进程组状态和错误处理。GPU 负责 forward/backward kernel、梯度写入和 NCCL kernel。CUDA enqueue 是异步的，所以 CPU 上 `loss.backward()` 的耗时不能直接当 GPU backward 完成时间。

DDP 在每张 GPU 上保存完整模型，不是参数分片：

- 参数：约 `P b_p`；
- 梯度：约 `P b_g`；
- optimizer state：SGD momentum 约 `P b_o`，Adam 常见约 `2P b_o`，是否存在 FP32 master weight 取决于精度方案；
- activation：与 local batch、sequence、layer、checkpoint policy 有关，记作 `A_local`；
- DDP bucket：实现与配置相关。`gradient_as_bucket_view=False` 可能需要独立 bucket buffer；开启后第一轮之后 gradient 可成为 bucket view，减少一份复制/峰值，但不能对 gradient 做不兼容的 `detach_()`。

因此 DDP 的基本显存量级仍是：

```text
M_rank ≈ P b_p + P b_g + optimizer_state + A_local + DDP buckets + temporary/workspace
```

增加 GPU 不会分摊参数、梯度或 optimizer state；它主要增加 global batch/吞吐。这正是后续 FSDP/ZeRO 要解决的问题。

## 2. DDP 通信量

每个参与训练的参数通常产生一份梯度 payload：

```text
M_grad = P b_g
```

若 NCCL 使用经典 ring all-reduce，reduce-scatter + all-gather 的每 rank 单向发送量近似：

```text
V_sent,rank ≈ 2 (N - 1) / N · M_grad
```

接收量同样约为 `V_sent,rank`；如果统计 NIC 上 send+receive 总字节，应再乘 2。报告必须说明采用哪一种口径。

这是算法 payload 模型，不是网卡端实际字节数：协议头、分块、通道、拓扑和 NCCL 选择的 tree/ring/CollNet 等都会改变现场。通信时间可粗略建模为：

```text
T_comm ≈ K · α + V_sent,rank / BW_effective
```

`K` 是 collective/bucket 数，`α` 是每次启动与网络延迟。bucket 很大时 `K` 小但首个 collective 晚；bucket 很小时通信更早，却可能被 `Kα`、小消息低带宽效率和更多 launch 吞掉。

## 3. Bucket 为什么控制 overlap

autograd 按反向依赖逐层产生梯度。某个 bucket 中所有梯度 ready 后，reducer 才能发起该 bucket 的 all-reduce。

大 bucket baseline：

```text
backward compute =========================
all-reduce                                =========
```

小 bucket candidate：

```text
backward compute =========================
all-reduce         ===  ===  ===  ===  =====
```

candidate 是否更快取决于：参数注册/反向 ready 顺序、bucket 重建后的实际分组、layer compute 时长、collective latency、通信 stream 优先级和 GPU/网络拓扑。只看 `bucket_cap_mb` 配置不能证明发生了 overlap。

`static_graph=True` 适合每轮使用参数集合与控制流固定的模型，可减少动态图检查并支持一些重入 backward 场景；动态 unused parameter 图不能盲开。`find_unused_parameters=True` 会遍历 autograd 图并可能增加开销，只有模型确实存在 unused 参数时才使用。

## 4. NCCL 总时间为什么不等于 step 增量

假设 step 中 NCCL kernel interval union 为 8 ms，其中 6 ms 与 backward compute 重叠：

```text
NCCL total = 8 ms
overlap = 6 ms
unoverlapped / exposed NCCL = 2 ms
```

不能把 8 ms 全部算成通信 penalty。多 stream kernel 时长还必须先求 interval union，不能简单相加，否则并发 kernel 会被重复计数。

但 2 ms 也不自动等于因果 penalty：

- overlap 时 NCCL 可能争用 SM、copy engine、L2、DRAM 或内存带宽，使 compute 变慢；
- “被覆盖”的 compute 可能不在最终关键路径；
- optimizer/下一 step 可能等待最后一个 bucket；
- rank skew 会使其他 rank 在 collective 中表现为等待；
- profiler 会引入开销。

所以至少需要两类证据：

1. timeline：`NCCL total / overlap / exposed`；
2. 端到端 A/B：相同 workload 下 step time、throughput、GPU active/bubble 是否改善。

## 5. 如何从 timeline 判断 overlap

正确流程：

```text
correctness
  -> synchronized benchmark
  -> PyTorch Profiler 定位 DDP op/bucket
  -> Nsight Systems 查看 compute stream 与 NCCL stream
  -> interval union/intersection
  -> 调整 bucket/graph policy
  -> 同条件重新 benchmark + profile
```

Nsight Systems 中应确认：

- 每个 rank 的 NVTX step/backward 边界；
- backward kernels 位于 compute stream；
- `ncclDevKernel*` 等 collective kernels 位于通信 stream；
- collective 的开始时间是否随梯度 ready 逐步出现；
- 最后一个 NCCL 是否延伸到 backward 尾部之后；
- CPU 是否有 launch gap、同步或 rank straggler；
- baseline 与 candidate 的 kernel 数、collective 数和 exposed 区间是否按假设变化。

`src/distributed_basics/analyze_nsys.py` 使用 CPU NVTX 范围内 CUDA runtime launch 的 correlation ID 找到 GPU kernels，再做 interval union。自动分类只是辅助；最终必须打开 raw timeline 复核 kernel 名和依赖。

## 6. Benchmark 如何设计

Baseline 与 candidate 必须固定：

- global/local batch、sequence、hidden、layers；
- dtype、loss scaling、optimizer；
- rank 到 GPU/NIC/NUMA 的绑定；
- warmup、测量轮数、时钟/功耗/温度；
- DataLoader 和 checkpoint 是否进入边界；
- NCCL 环境变量和算法选择；
- profiler 关闭后的最终复测。

分布式 step time 应同时保存每 rank raw 样本，并关注同一步最慢 rank。平均一个快 rank 会隐藏集群关键路径。至少报告：

- mean/p50/p90/p99 step time；
- global samples/s 或 tokens/s；
- MFU（峰值匹配时才算）；
- peak allocated/reserved；
- NCCL kernel union、overlap、exposed；
- GPU active/bubble；
- collective 数量/尺寸；
- scaling efficiency。

强 scaling：global batch 固定，GPU 增加；weak scaling：local batch 固定，global batch 随 GPU 增加。两者不能混报。

## 7. PyTorch Profiler 与 Nsight Compute 分工

PyTorch Profiler 用来回答哪个 autograd op、DDP collective、CPU launch 或内存分配对应异常；它不如 Systems 适合跨进程、跨 stream 的全局关键路径。

Nsight Compute 不用于证明 NCCL 与 backward overlap。只有 Systems 已证明某个 backward compute kernel 是热点时，才从 `basic` 开始检查：

- Tensor Core 是否实际使用；
- DRAM/L2 throughput；
- achieved/theoretical occupancy；
- registers/thread、shared memory/block；
- warp stall、grid/waves、访问合并。

当前 Thor 是 20 SM、CC 11.0 的现场快照，不能套用 B200/H100 阈值。当前用户没有 NCU performance-counter 权限，未采集项必须写 `unavailable`。

## 8. 8 GPU、128 GPU、1024 GPU 为什么不同

8 GPU 单机可能主要走 NVLink/NVSwitch，带宽高、延迟低，较小 bucket 能有效隐藏通信；但 PCIe-only 机器会呈现不同拐点。

128/1024 GPU 时会出现：

- 跨节点 IB/RoCE 带宽、交换机 oversubscription、拥塞与路由；
- 节点内/节点间分层 collective；
- rank placement、NIC affinity、NUMA 和 GPUDirect RDMA；
- `Kα` 随 collective 数放大；
- 极慢 rank、数据抖动和网络重传控制整个 step；
- 任一 rank crash/collective mismatch 影响整个 job。

因此“2 GPU 最佳 bucket”不能直接推广到 1024 GPU。必须按拓扑层次分别测消息尺寸曲线和 end-to-end scaling。

## 9. 高频故障与排查顺序

症状：所有 rank 卡在 backward/NCCL。

1. 保存第一现场：每 rank stack、最后 step、主机/GPU/rank 映射、NCCL logs；
2. 检查所有 rank 是否进入相同 collective、tensor shape/dtype/count 是否一致；
3. 检查某 rank 是否先 OOM、NaN、DataLoader 异常或进程退出；
4. 检查 unused parameter/条件分支导致的 reducer 不一致；
5. 再查 GPU/NIC link、RDMA、RoCE loss/ECN/PFC、IB port error 和 NCCL timeout；
6. 最后才调整 timeout；延长 timeout 不能修复 collective mismatch。

症状：吞吐下降但无 hang。

1. 先比较每 rank step 分布和 straggler；
2. 看 exposed NCCL，而不是 NCCL 总和；
3. 检查 bucket 数量/尺寸和首个/末个 collective；
4. 检查 CPU/DataLoader gap、GPU thermal/power 和后台流量；
5. 对照单卡、单机多卡、跨机逐层定位性能断崖。

## 10. 本模块实验与下一步

```bash
PYTHONPATH=Work/src python -m unittest discover \
  -s Work/src/distributed_basics/tests -v

Work/src/distributed_basics/scripts/run_correctness.sh \
  --device cpu --backend gloo --nproc-per-node 2
```

本模块完成后，应能够解释并测量 DDP 的梯度同步、bucket readiness、通信量、compute/communication overlap 与 exposed communication。下一模块将进入 NCCL collective microbenchmark、GPU topology、NVLink/PCIe 与网络分层，而不是继续堆 DDP 启动参数。
