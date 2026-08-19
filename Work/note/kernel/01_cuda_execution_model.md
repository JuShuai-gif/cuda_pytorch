# 01｜CUDA 执行模型：Thread / Warp / Block / Grid / SM 与内存层次

## 本模块解决的问题

要从"会写 PyTorch"走到"能看懂一个 kernel 为什么慢"，必须理解 GPU 的硬件执行模型：一个 kernel 是怎么被拆成 thread、warp、block，怎么被调度到 SM，数据怎么从 HBM 一路流到 register。本章回答：

```text
一个 kernel 启动后，硬件到底发生了什么？
occupancy 是什么，为什么它决定能否掩盖延迟？
register / shared memory 用多了会发生什么？
memory coalescing / bank conflict 为什么影响带宽？
```

配套代码：`src/kernel/cuda_async/`、`src/kernel/cuda_graph/` 用 PyTorch 侧验证异步与 launch 行为；真正的 warp/bank/register 计数要用 `ncu` 看（见 `note/profiling/02_ncu_kernel_profiling.md`）。

---

## 1. 执行层级：Thread → Warp → Block → Grid → SM

```text
Kernel（一个函数，一次 launch）
   └── Grid（由多个 block 组成，覆盖整个输出空间）
          └── Block（调度到单个 SM 上的一组 thread）
                 └── Warp（block 内每 32 个 thread 一组，调度基本单位）
                        └── Thread（最小执行单元）
```

关键规则：

- **Warp 是调度和执行的原子单位**：SM 以 32 个 thread 为一个 warp 发射指令。同一 warp 内 thread 走 SIMT（Single Instruction Multiple Threads）。
- **Block 被整体分配到一个 SM**，SM 上可以同时驻留多个 block（只要 register/shared memory 够）。
- **一个 block 不会跨 SM**，一个 SM 可以跑多个 block。block 之间无天然通信（协作靠 global memory，或单 block 内靠 shared memory + `__syncthreads`）。

---

## 2. 内存层次与访问成本

```text
register       每 thread 私有，最快（~1 cycle），容量极小（几十 KB/SM）
shared memory  block 内共享，L1 旁边（~几十 cycle），可编程
L1/L2 cache    片上缓存，不可直接控制（L2 32MB 本机 Thor）
global/HBM     片上之外的大显存，~几百 cycle，带宽是主要瓶颈
constant/texture  只读缓存，广播访问快
```

直觉：**能放 register 就别放 shared，能放 shared 就别反复读 global**。但每级容量有限，用多了就会掉到下一级。

---

## 3. Memory Coalescing（访存合并）

GPU 访存按 **32-byte / 128-byte 的段** 传输。一个 warp 的 32 个 thread 如果访问**连续的** global memory 地址，可以被合并成一次（或少数几次）宽传输；如果各自跳着访问（stride 访问），则拆成 32 次传输，带宽利用率暴跌。

```text
coalesced：thread i 访问 a[i]     → 一次段传输
strided ： thread i 访问 a[i*K]   → 多次段传输（K 大时接近 32 次）
```

这是很多 memory-bound kernel 慢的根因。ncu 里看 `gld_efficiency` / `L2 sectors` 可以量化合并程度。

---

## 4. Bank Conflict（共享内存冲突）

shared memory 被分成 32 个 bank（每 bank 4 字节，不同架构略有差异）。一个 warp 的 thread 如果访问同一 bank 的不同地址，就会发生 bank conflict，访问被串行化为多拍：

```text
无冲突：thread i 访问 s[i]           → 1 拍
2-way 冲突：thread i 访问 s[i*2]     → 2 拍
全冲突：所有 thread 访问 s[i*32]     → 32 拍
```

广播（所有 thread 读同一地址）不算冲突。经典矩阵转置、reduction 的 shared memory 布局都要小心这个问题。

---

## 5. Occupancy（占用率）

occupancy = 实际活跃 warp / SM 理论最大 warp。它决定**能否用并行度掩盖延迟**：

```text
SM 每次访存 global memory 都要等几百 cycle。
如果 SM 上有足够多 warp 可切换，等访存的 warp 挂起时，其它 warp 继续算 → 延迟被隐藏。
warp 太少 → 访存时 SM 空转 → 慢。
```

**occupancy 是手段不是目的**：高 occupancy 能掩盖延迟，但一个 register 特别多（register pressure）导致 occupancy 骤降的 kernel，可能因为更少访存反而更快。所以优化目标是"在访存和并行度之间找平衡"，不是盲目堆 occupancy。

### 影响 occupancy 的两个压力

```text
register pressure  每 thread 用的 register 越多，SM 能容纳的 thread 越少
shared pressure    每 block 用的 shared memory 越多，SM 能容纳的 block 越少
```

ncu 里直接看 `achieved occupancy`、`registers/thread`、`shared memory/block`、`block limit` 四个数就能判断是被哪个限制住了。

---

## 6. 为什么这些对推理重要

推理优化的很多手法，本质都是在对抗执行模型的某个瓶颈：

| 手法 | 对抗什么 |
|---|---|
| kernel fusion | 减少 launch 次数 + 减少 global 中间张量搬运 |
| 提高 batch | 提高 GEMM 的 Tensor Core 利用率（更多并行） |
| 低精度 fp16/bf16 | 同样带宽搬更多数据，缓解 memory-bound |
| 调整 BLOCK_SIZE / num_warps | 平衡 occupancy 与 register/shared 压力 |
| 避免 bank conflict | 提高 shared memory 有效带宽 |
| CUDA Graph | 消除大量 launch 开销（见 02 篇） |

**判断一个 kernel 是 memory-bound 还是 compute-bound**，是下一步所有优化（量化、fusion、GEMM 选型）的前提，方法是看 ncu 的 DRAM throughput 与 Tensor Core utilization 的相对关系，见 `note/profiling/02_ncu_kernel_profiling.md`。

---

## 7. 编译链路：CUDA C++ → PTX → SASS

理解"kernel 真正跑的是什么"要走过这条链：

```text
CUDA C++（.cu）
   ↓ nvcc -ptx
PTX（虚拟 ISA，可移植）
   ↓ ptxas（随 driver/JIT 针对具体架构）
SASS（sm_110 机器码，真正执行）
   ↓
GPU
```

- **PTX**：中间表示，跨架构可移植，但还没绑定具体 GPU。
- **SASS**：针对某架构（如 sm_110）生成的真实指令。`cuobjdump --dump-sass` 能反汇编看。
- register 数量、shared memory 用量、指令调度都在 ptxas 阶段决定，这也是 register pressure 的来源。

`nvcc --resource-usage` 或 `--ptxas-options=-v` 能直接打印每个 kernel 的 register/shared 用量，是判断 occupancy 限制的第一手材料。

---

## 8. 本机硬件锚点（Thor, sm_110）

```text
SM 数               20
max threads / SM    1536
L2 cache            32 MB
内存               统一内存（CPU/GPU 共享物理 DRAM），~128GB
compute capability  sm_110
```

这几个数直接决定 occupancy 上限（20 SM × 每 SM 上限）、L2 命中收益（32MB 能装多少权重/激活）和 memory-bound 的判定（统一内存的带宽特性不同于离散 HBM）。

---

## 9. 本模块闭环小结

```text
问题：kernel 为什么慢
      ↓
原理：warp 是调度单位，SM 靠多 warp 隐藏访存延迟
      ↓
瓶颈：memory-bound（coalescing/bank/带宽）vs compute-bound（Tensor Core/occupancy）
      ↓
测量：ncu 看 occupancy、register、shared、DRAM、Tensor Core、stall
      ↓
优化：block/warp 配置、访存布局、fusion、低精度
```

下一模块：`02_cuda_async_and_stream.md`，回答"CPU 怎么把工作交给 GPU、stream/event/graph 如何影响延迟与重叠"。
