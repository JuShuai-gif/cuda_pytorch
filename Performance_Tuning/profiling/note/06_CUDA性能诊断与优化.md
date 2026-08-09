# CUDA性能诊断与Bad/Good优化

## 总体流程

```text
nvidia-smi/tegrastats → nsys（哪里慢）→ ncu（为什么慢）→ Roofline → 优化 → nsys复验E2E
```

GPU Util高只表示采样窗口内有GPU工作，不代表kernel高效。

## Kernel Hotspot与NVTX

`cuda_13_kernel_hotspot_bad_good`包含fast/medium/slow kernel和NVTX。本机nsys确认slow_kernel占GPU kernel时间80.7%。

```bash
nsys profile -t cuda,nvtx,osrt -o hotspot ./src/build/cuda_13_kernel_hotspot_bad_good
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum hotspot.nsys-rep
ncu --query-metrics
ncu --set full --kernel-name regex:slow_kernel --launch-count 1 ./src/build/cuda_13_kernel_hotspot_bad_good
```

当前ncu因ERR_NVGPUCTRPERM未完成计数器采集，不应sudo修改。

## Memory Bound vs Compute Bound

Memory Bound候选：AI低 + Memory/DRAM吞吐高 + SM低。Compute Bound候选：AI高 + SM/指令吞吐高 + DRAM低。吞吐率单项不能定案。

- `cuda_02_memory_bound`：Triad/elementwise，低AI。
- `cuda_03_compute_bound`：重复FMA，高AI。

## Global Memory与Coalescing

`cuda_04_uncoalesced_access`与`cuda_05_coalesced_access`比较stride和连续映射。关注transactions/sectors per request、L1/L2、DRAM和Long Scoreboard。必须先`ncu --query-metrics`，不要照搬legacy metric。

## Shared Memory Bank Conflict

`cuda_07_shared_memory_bank_conflict`用padding消除冲突。常见GPU有32 bank，但具体映射以目标架构为准。关注shared transactions、bank conflict和warp stall。

## Warp Divergence

`cuda_06_branch_divergence`让同一warp奇偶线程走不同复杂路径。Good通过数据/线程映射让warp内路径一致。它不同于CPU branch predictor miss。

## Occupancy

`cuda_08_low_occupancy`制造高register pressure。关注Theoretical/Achieved Occupancy、Registers/Thread、Shared/Block、Active/Eligible Warps和stall。

Occupancy 100%不保证最快，50%也不自动说明问题。真正的问题是eligible warp是否不足以隐藏Long Scoreboard等延迟；降低register若导致spill可能更慢。

## Launch、Transfer与Sync

- `cuda_09_kernel_launch_overhead`：大量tiny kernel；优化为fusion/CUDA Graph。
- `cuda_10_h2d_d2h_transfer`：分析bytes、duration和copy engine。
- `cuda_11_cpu_gpu_sync`：循环device synchronize vs批量异步。
- `cuda_14_pipeline_overlap_bad_good`：serial vs pinned memory、async、双buffer、双stream。

cudaDeviceSynchronize等待整个device先前工作；cudaStreamSynchronize只等指定stream；cudaEventSynchronize只等事件依赖到达。异步API不等于实际重叠，必须从nsys timeline证明。本机pipeline约13.73→4.93ms。

## 正确计时

```cpp
cudaEventRecord(start);
kernel<<<grid, block>>>();
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&ms, start, stop);
```

未同步的std::chrono通常只测launch。

## nsys时间线阅读顺序

1. 先看E2E NVTX阶段宽度；
2. 看CPU线程是否持续提交工作；
3. 看CUDA API中的sync和launch；
4. 看H2D/D2H次数与持续时间；
5. 看GPU kernel排列、gap和stream；
6. 选占比高且可优化的kernel进入ncu。

cudaMalloc首次调用可能包含context初始化，不能把它误判为稳态每帧成本。应在warm-up后截取active窗口。

## ncu指标组

不同版本指标名字不同，但概念稳定：

| 类别 | 要回答的问题 |
|---|---|
| Speed of Light | SM还是Memory更接近上限 |
| Memory Workload | DRAM/L2/L1/shared流量如何 |
| Scheduler | Active/Eligible Warps是否足够 |
| Warp State | Long/Short Scoreboard、Barrier等 |
| Launch Stats | block、register、shared、occupancy |
| Source Counters | 哪行源码对应stall与指令 |

ncu可能replay kernel。带有随机、外部IO或状态修改的kernel需确保replay安全。

## Warp Stall解释

- Long Scoreboard：常等待global/texture memory；
- Short Scoreboard：较短数据依赖或shared memory；
- Barrier：warp等待同步；
- Not Selected：有其他eligible warp被选中，不一定坏；
- Wait：固定延迟/执行依赖；
- MIO Throttle：memory input/output pipeline压力；
- LG Throttle：local/global指令队列压力。

Stall百分比不是“删除这一项就能等比例加速”，必须结合eligible warp和吞吐。

## Coalescing推理

若stride版本runtime高、global sectors/request高、DRAM transactions增加且checksum一致，支持非合并访问根因。若L2命中很高掩盖DRAM差异，应扩大工作集或减少重复。

## Occupancy资源约束

每SM可驻留block数量受threads、warps、register file、shared memory和架构上限共同限制。优化register数可能造成local-memory spill；减少shared memory可能增加global traffic。最终看runtime和stall，不追求单一最大occupancy。

## Kernel Fusion权衡

Fusion可减少launch和中间tensor流量，但会增加寄存器、代码体积和编译复杂度，甚至降低occupancy。Transformer中LayerNorm、bias、activation等融合常有效；大GEMM不应盲目与所有逻辑融合。

## Async与Overlap条件

要发生H2D/compute重叠通常需要：

- pinned host memory；
- cudaMemcpyAsync；
- 非default或合适stream；
- 硬件copy engine支持；
- 足够大的工作量；
- 不存在隐式同步；
- buffer生命周期正确。

API返回异步只说明CPU未等待，不证明copy与kernel并发。

## CUDA Graph适用性

Graph适合拓扑重复、kernel细碎、CPU launch overhead明显的工作负载。动态shape、复杂控制流或频繁更新图会降低收益。先用nsys确认launch gap再决定。

## CUDA Correctness

每个Bad/Good应：

1. 初始化相同输入；
2. warm-up不污染最终输入，或重新初始化；
3. 检查cudaGetLastError；
4. 同步后检查runtime error；
5. D2H比较误差；
6. 浮点结果按合理rtol/atol，而非bitwise强求。

## CUDA实验练习

1. nsys过滤13号Demo，计算slow kernel占比。
2. 查询本机ncu metrics，而不是复制文档旧名字。
3. 对14号pipeline检查stream之间是否真实重叠。
4. 将tiny kernel融合，比较kernel count、API time和E2E。

## 目标机ncu采集脚本

```bash
cd /home/ghr/code/cuda_pytorch/Performance_Tuning/profiling
./src/scripts/collect_ncu_target.sh \
  ./src/build/cuda_13_kernel_hotspot_bad_good slow_kernel ./src/ncu_slow
```

脚本先保存当前ncu版本的metric列表，再按kernel regex采集并导出ncu-rep。遇到ERR_NVGPUCTRPERM只报告并退出，不修改系统权限。
