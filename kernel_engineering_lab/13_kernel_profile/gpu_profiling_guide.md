# GPU 端性能分析指南

## 工业背景

GPU kernel engineering 的核心技能就是 profiling。你写的每一行 CUDA 代码，最终都要通过 ncu 和 nsys 来验证性能。

在实际工作中：
- **Nsight Systems（nsys）** 是你的"侦察兵"——先找到哪个 kernel 慢、哪个阶段是瓶颈
- **Nsight Compute（ncu）** 是你的"显微镜"——深入单个 kernel，看 occupancy、memory throughput、tensor core 利用率

以下是本书中最重要的性能分析技能。

---

## 目录

- [Nsight Compute（ncu）详解](#nsight-computencu-详解)
- [Nsight Systems（nsys）详解](#nsight-systemsnsys-详解)
- [性能优化决策树](#性能优化决策树)
- [常见性能反模式](#常见性能反模式)

---

## Nsight Compute（ncu）详解

ncu 是 GPU kernel 级性能分析的终极工具。它通过读取 GPU 硬件计数器，告诉你每个 kernel 的**精确性能瓶颈**。

### 安装

```bash
# 方式 1：apt 安装（推荐，CLI only）
sudo apt update
sudo apt install nsight-compute

# 方式 2：从 NVIDIA 官网下载完整包（含 GUI）
# https://developer.nvidia.com/nsight-compute
# 下载 .deb 或 .run 文件
sudo dpkg -i nsight-compute-<version>.deb
# 或
sudo sh nsight-compute-<version>.run

# 验证安装
ncu --version

# 安装后常见路径
# CLI: /opt/nvidia/nsight-compute/<version>/ncu
# GUI: /opt/nvidia/nsight-compute/<version>/ncu-ui
```

### 基础用法

```bash
# 对所有 kernel 做完整分析
ncu --set full python my_benchmark.py

# 只分析特定名称的 kernel
ncu --kernel-name "flash_attention_fwd" --set full python my_benchmark.py

# 使用正则匹配多个 kernel
ncu --kernel-name regex:"attention|matmul" --set full python my_benchmark.py

# 只分析第 1 次 kernel launch
ncu --launch-skip 0 --launch-count 1 python my_benchmark.py

# 只分析第 3 到第 5 次 launch
ncu --launch-skip 2 --launch-count 3 python my_benchmark.py

# 导出为 CSV 做自动化分析
ncu --set full --csv --log-file profile.csv python my_benchmark.py

# 保存为 .ncu-rep 文件（可在 GUI 中打开）
ncu --set full -o my_profile python my_benchmark.py
```

### 预定义 Section 集

ncu 提供了多个预定义的 profiling section 集合：

| --set 参数 | 包含的 section | 开销 | 适用场景 |
|------------|---------------|------|----------|
| `--set basic` | SpeedOfLight, LaunchStatistics, Occupancy | 低 | 快速判断瓶颈类型 |
| `--set full` | 所有 section | 高 | 完整深入分析 |
| `--set roofline` | 生成 roofline 分析所需的所有指标 | 中 | 判断 memory-bound vs compute-bound |
| `--set memory` | MemoryWorkloadAnalysis | 中 | 专注内存访问模式分析 |
| `--set compute` | ComputeWorkloadAnalysis, SchedulerStats | 中 | 专注计算效率和调度 |
| `--set occupancy` | Occupancy | 低 | 只看 occupancy |

```bash
# 快速判断瓶颈类型（推荐第一步）
ncu --set basic python my_benchmark.py

# 内存问题深入分析
ncu --set memory --section MemoryWorkloadAnalysis_Chart python my_benchmark.py

# Roofline 分析
ncu --set roofline python my_benchmark.py
```

### 只分析特定 Section

```bash
# 只看内存工作负载
ncu --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables python my_benchmark.py

# 只看计算工作负载
ncu --section ComputeWorkloadAnalysis --section InstructionStats python my_benchmark.py

# 只看调度统计
ncu --section SchedulerStats --section WarpStateStats --section Occupancy python my_benchmark.py

# 只看出生率
ncu --section SpeedOfLight python my_benchmark.py
```

### 基线对比（A/B 测试）

```bash
# 保存基线
ncu --set full --save baseline.ncu-rep python baseline.py

# 对比新版本
ncu --set full --import-source yes --compare baseline.ncu-rep python optimized.py
```

---

### 关键 Metric 解读

ncu 的输出包含数百个指标，以下是最重要的几类。每个指标都直接对应硬件行为。

#### 1. Speed of Light 汇总（第一步必看）

这是 ncu 自动生成的汇总页面，告诉你 kernel 的瓶颈是 memory 还是 compute。

```
Speed of Light 报告：
┌──────────────────────────────────────────────────────┐
│ Section: SpeedOfLight                                │
│                                                      │
│ Memory Throughput:      ████████████░░░░  82.3%      │  ← 显存带宽使用率
│ Compute (SM) Throughput: ██░░░░░░░░░░░░░  14.7%      │  ← SM 计算利用率
│                                                      │
│ 瓶颈诊断：Memory-bound（显存带宽是瓶颈）               │
└──────────────────────────────────────────────────────┘
```

**解读规则**：
- Memory Throughput > 80% 且 Compute Throughput < 50%：**Memory-bound**。优化内存访问。
- Compute Throughput > 80% 且 Memory Throughput < 50%：**Compute-bound**。优化计算。
- 两者都低：**Latency-bound**。提高 occupancy、减少同步、增加并行度。
- 两者都高：**Good job**。Kernel 已接近硬件峰值。

#### 2. Memory Workload Analysis（内存工作负载）

这是分析 memory-bound kernel 的核心模块。

```
关键指标：

dram__bytes.sum              总 DRAM（显存）读写字节数
dram__throughput.max.pct_of_peak_sustained_elapsed    DRAM 带宽使用率（占峰值的百分比）
l1tex__throughput.max.pct_of_peak_sustained_elapsed   L1 Cache 吞吐率
l2tex__throughput.max.pct_of_peak_sustained_elapsed   L2 Cache 吞吐率

sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed   Load/Store 指令占比
sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed   FMA 指令占比
```

**解读规则**：

| 指标 | 正常范围 | 需要关注 | 含义 |
|------|---------|----------|------|
| `dram__throughput` | < 30%（compute-bound 时正常） 或 > 70%（memory-bound 时正常） | 80%-95%（memory-bound 下接近峰值，说明内存访问已优化得很好） | 显存带宽使用率 |
| `l1tex__throughput` | 越高越好（L1 cache 命中意味着没有访问显存） | < 20% 可能表示 L1 cache 命中率低 | L1 cache 吞吐率 |
| `l2tex__throughput` | 中等（L2 弥补 L1 miss） | > 80% 意味着大量 L1 miss 被 L2 承接，但不是免费的 | L2 cache 吞吐率 |
| `sm__inst_executed_pipe_lsu` | < 30%（非 memory-bound 时） | > 60% + dram throughput > 70%，确认 memory-bound | Load/Store 指令占比 |
| `l1tex__data_pipe_lsu_wavefronts_shared` | 0 | > 0 意味着 shared memory bank conflict | Shared memory bank conflict 计数 |

**memory-bound kernel 的特征**：
- `dram__throughput` > 70%
- `sm__inst_executed_pipe_lsu` 占比高（> 50%）
- `sm__throughput`（计算利用率）低
- Speed of Light 明确显示 Memory 是瓶颈

**compute-bound kernel 的特征**：
- `sm__inst_executed_pipe_fma` 占比高
- `dram__throughput` 低（因为大部分数据在 cache/shared memory 中）
- `sm__throughput` 高

#### 3. Scheduler Statistics（调度统计）

告诉你 warp 调度效率。

```
关键指标：

sm__warps_active.avg.pct_of_peak_sustained_elapsed    活跃 warp 的百分比
sm__occupancy.avg.pct_of_peak_sustained_elapsed        Occupancy（理论 warp 数 vs 实际活跃 warp 数）
sm__inst_executed.avg.pct_of_peak_sustained_elapsed    指令执行率

sm__warps_issue_stalled_long_scoreboard_per_warp_active  因等待内存而 stall 的 warp
sm__warps_issue_stalled_short_scoreboard_per_warp_active 因等待执行结果而 stall 的 warp
sm__warps_issue_stalled_not_selected_per_warp_active     因调度器选择其他 warp 而 stall
sm__warps_issue_stalled_barrier_per_warp_active          因 __syncthreads() 而 stall
```

**Occupancy 深度解读**：

| Occupancy 范围 | 判断 | 原因与对策 |
|----------------|------|-----------|
| > 75% | 优秀 | — |
| 50-75% | 正常 | 对于 register 密集型或 shared memory 密集型 kernel 来说很正常 |
| 25-50% | 偏低 | 检查 register 使用量是否过高（> 128/thread on A100），或 shared memory 使用过多 |
| < 25% | 严重偏低 | 每个 SM 上活跃 warp 太少，无法隐藏延迟。需要减少 register 或 shared memory 使用 |

**Occupancy 为什么重要？**

GPU 通过大量 warp 之间的上下文切换来隐藏延迟。当一个 warp 在等待内存时，SM 会切换到另一个 warp 继续执行。如果活跃 warp 太少，SM 没有足够的 warp 来切换，就会出现"空转等待"。

**计算公式**：
```
Occupancy = active_warps / max_warps_per_SM

max_warps_per_SM = f(registers_per_thread, shared_memory_per_block, block_size)
```

#### 4. Compute Workload Analysis（计算工作负载）

```
关键指标：

sm__throughput.avg.pct_of_peak_sustained_elapsed    SM 整体利用率
sm__pipe_tensor.avg.pct_of_peak_sustained_elapsed   Tensor Core 利用率
sm__pipe_fma.avg.pct_of_peak_sustained_elapsed      FMA（浮点乘加）利用率
sm__pipe_alu.avg.pct_of_peak_sustained_elapsed      ALU（整数/逻辑）利用率
sm__pipe_fp64.avg.pct_of_peak_sustained_elapsed     FP64 利用率
```

**解读**：
- `sm__pipe_tensor` > 0 说明 kernel 使用了 Tensor Core（只有 mma 指令会用到）
- `sm__pipe_fma` 高但 `sm__pipe_tensor` = 0，说明 kernel 在做大量 FMA 但没用 Tensor Core
  → 考虑用 `mma.sync` 或 `wmma` 切换到 Tensor Core（在支持的数据类型下性能提升 2-4x）
- `sm__throughput` 低但 `dram__throughput` 也低 → latency-bound

#### 5. Launch Statistics（启动统计）

```
关键指标：

gpu__time_duration.sum                 所有 kernel 总执行时间（ns）
sm__cycles_elapsed.avg.per_second      SM 平均每秒周期数
registers_per_thread                   每个线程使用的寄存器数
shared_memory_per_block                每个 block 使用的 shared memory（bytes）
grid_size                              启动的 grid 维度
block_size                             启动的 block 维度

launch__registers_per_thread           Kernel 请求的寄存器数
launch__shared_mem_per_block           Kernel 请求的 shared memory（bytes）
```

**Register Spilling 判断**：

```
如果 registers_per_thread > 128（A100），检查：
  - l1tex__data_pipe_lsu_wavefronts_shared 是否 > 0
  - 如果是，说明有 register spilling（寄存器溢出到 local memory / L1 cache）

register spilling 的影响：
  - 增加 L1 cache 流量 → 减少有效 cache 容量
  - 增加内存延迟（local memory 在 L1/L2 cache 或显存中）
  - 可能显著降低性能
```

**Registers per Thread 与 Occupancy 的关系（以 A100 为例）**：

| registers/thread | 最大 threads/SM | 最大 blocks/SM | Occupancy（256 threads/block） |
|------------------|-----------------|----------------|-------------------------------|
| ≤ 32 | 2048 | 32 | 100% |
| 40 | 1536 | 24 | 75% |
| 48 | 1280 | 20 | 62.5% |
| 64 | 1024 | 16 | 50% |
| 80-96 | 768 | 12 | 37.5% |
| 128 | 512 | 8 | 25% |
| > 128 | 512 | 8 | ≤ 25% |

可以看出，registers/thread 从 32 增加到 128，occupancy 从 100% 降到 25%。在某些场景下，occupancy 低是可以接受的（如果每个 warp 的计算密度足够高）。

---

### Roofline 分析

Roofline 模型帮你判断一个 kernel 是 memory-bound 还是 compute-bound。

**方法 1：使用 ncu 内置的 roofline 分析**

```bash
# 一行命令生成 roofline 分析
ncu --set roofline python my_benchmark.py
```

**方法 2：手动计算 Arithmetic Intensity**

```
Arithmetic Intensity（算术强度）= Total FLOPs / Total Bytes Moved

如果 AI < 设备的 "ridge point"（转折点）：
  → Memory-bound（性能受限于显存带宽）
如果 AI > ridge point：
  → Compute-bound（性能受限于计算吞吐）
```

**A100 的 Roofline 关键数字**：

```
峰值计算吞吐（FP16 Tensor Core）：312 TFLOPS
峰值显存带宽（HBM2e）：1555 GB/s（约 1.55 TB/s）

Ridge Point（转折点）= 312,000 GFLOPS / 1,555 GB/s ≈ 200 FLOPs/Byte
```

**典型 kernel 的算术强度**：

| Kernel 类型 | 近似算术强度（FLOPs/Byte） | 瓶颈判断 |
|-------------|--------------------------|----------|
| Elementwise Add | ~0.08 | 强烈 memory-bound |
| RMSNorm | ~0.33 | 强烈 memory-bound |
| SiLU/GELU | ~0.08-0.16 | 强烈 memory-bound |
| Vector Add | ~0.04 | 强烈 memory-bound |
| Matmul（未 tiling） | ~2-10 | memory-bound |
| Matmul（已 tiling） | ~50-200 | compute-bound（接近 ridge point） |
| Matmul（Tensor Core, M=N=K=8192） | ~1000+ | 强烈 compute-bound |
| Flash Attention（fwd） | ~50-150 | 混合，取决于 seq_len |
| Reduction（sum） | ~0.04 | memory-bound |

**Roofline 优化策略**：

```
如果 kernel 在 roofline 图上位于：
├─ 对角线下方（memory-bound 区）
│  → 减少全局内存访问：
│     - 使用 shared memory / register 保存中间结果
│     - 增加 tiling block size（更多数据重用）
│     - 使用 vectorized load/store（一次加载更多字节）
│     - 减少 unnecessary load/store
│
├─ 水平渐近线下方（compute-bound 区）  
│  → 增加计算效率：
│     - 使用 Tensor Core（mma/wmma 指令）
│     - 减少非计算指令（分支、指针操作）
│     - 优化 warp 级指令流
│
└─ 接近两条线交点
   → 良好优化的 kernel，使用 ncu 做微调
```

---

### 实战示例 1：分析 Memory-bound Kernel（Elementwise Add）

```bash
# 运行 profiling
ncu --kernel-name "vector_add" --set full python benchmark.py
```

**ncu 输出关键片段**：

```
Speed of Light:
  Memory Throughput:      89.2%   ← 显存带宽几乎用满
  Compute (SM) Throughput:  4.1%  ← SM 计算单元几乎空闲

Memory Workload:
  dram__bytes.sum:        268,435,456  (256 MB)
  dram__throughput:       89.2% of peak
  l1tex__throughput:      12.3% of peak
  l2tex__throughput:      85.7% of peak  ← L2 在弥补 L1 miss

Scheduler:
  sm__warps_issue_stalled_long_scoreboard: 78.3%  ← 78% 的 stall 是在等内存
  sm__occupancy:          62.5%  ← occupancy 正常，但 kernel 的瓶颈不是这里

Instruction Mix:
  sm__inst_executed_pipe_lsu: 91.2%  ← 91% 的指令是 load/store
  sm__inst_executed_pipe_fma:  1.2%
```

**诊断**：
- Memory throughput 89.2%，接近硬件峰值
- 91% 的指令是 load/store，SM 几乎不做计算
- 78% 的 warp stall 是因为等待内存

**结论**：这是一个典型的 memory-bound kernel。优化方向：
1. 如果可能，**与后续操作融合**（减少读写的内存量）
2. **使用 vectorized load/store**（如 `float4` 替代 `float`）
3. 这个 kernel 本身已经接近硬件极限，不需要更多优化

---

### 实战示例 2：分析 Compute-bound Kernel（Matmul）

```bash
ncu --kernel-name "tiled_matmul" --set full python benchmark.py
```

**ncu 输出关键片段**：

```
Speed of Light:
  Memory Throughput:      15.3%   ← 显存带宽未用满
  Compute (SM) Throughput: 72.1%  ← SM 计算单元繁忙

Memory Workload:
  dram__bytes.sum:         67,108,864  (64 MB)
  dram__throughput:        15.3% of peak
  l1tex__throughput:       89.2% of peak  ← L1 大量命中（数据复用）
  l2tex__throughput:       34.5% of peak

Scheduler:
  sm__warps_issue_stalled_short_scoreboard: 45.2%  ← 45% stall 在等计算依赖
  sm__occupancy:           37.5%  ← occupancy 偏低

Instruction Mix:
  sm__inst_executed_pipe_fma:  68.3%  ← FMA 占主要指令
  sm__inst_executed_pipe_lsu:  12.1%
  sm__inst_executed_pipe_tensor: 0%  ← 未使用 Tensor Core

Launch:
  registers_per_thread:    96    ← 96 个寄存器/线程！限制了 occupancy
  shared_memory_per_block: 49152 bytes (48 KB)
```

**诊断**：
- SM throughput 72.1%，但未用 Tensor Core
- occupancy 37.5% 偏低，`registers_per_thread=96` 是原因
- 45% warp stall 在等待计算依赖（不是等内存），确认是 compute-bound

**优化建议**：
1. **降低 register 使用量**（96 → 64），提高 occupancy
2. **检查 tiling size**—可能 block 太小导致 registers 浪费
3. **考虑用 Tensor Core**（如果数据是 FP16/BF16）—可提升 2-4x 计算吞吐
4. 检查 `float4` 向量化加载是否被正确使用

---

### CSV 导出与自动化分析

```bash
# 导出所有 kernel 的数据
ncu --set full --csv --log-file all_kernels.csv python benchmark.py

# 只用 basic section 加速
ncu --set basic --csv --log-file quick_scan.csv python benchmark.py

# 导出为 JSON（Python 脚本分析用）
ncu --set full --print-summary per-kernel --log-file output.txt python benchmark.py
```

**CSV 自动化分析示例**：

```python
import pandas as pd

# 读取 ncu 导出的 CSV
df = pd.read_csv("all_kernels.csv")

# 关键指标列（不同 ncu 版本列名可能不同）
KEY_COLS = [
    "Kernel Name",
    "gpu__time_duration.sum",
    "dram__throughput.max.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__occupancy.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.max.pct_of_peak_sustained_elapsed",
]

# 找到 memory-bound 的 kernel
memory_bound = df[df["dram__throughput"] > 70]

# 找到低 occupancy 的 kernel
low_occupancy = df[df["sm__occupancy"] < 30]
```

---

## Nsight Systems（nsys）详解

nsys 做的是**系统级分析**：CPU 和 GPU 的时间线、kernel 启动/执行顺序、H2D/D2H 传输、CUDA API 调用、Python 开销等。

### 安装

```bash
# apt 安装
sudo apt update
sudo apt install nsight-systems

# 或从 NVIDIA 官网下载
# https://developer.nvidia.com/nsight-systems
sudo dpkg -i nsight-systems-<version>.deb

# 验证
nsys --version
```

### 基础用法

```bash
# 生成 timeline 报告（.nsys-rep 文件，需 GUI 打开）
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=timeline_report \
    python my_benchmark.py

# 终端统计摘要
nsys profile \
    --trace=cuda,nvtx \
    --stats=true \
    python my_benchmark.py

# 指定 GPU 设备
nsys profile \
    --trace=cuda,nvtx \
    --gpu-metrics-device=0 \
    --output=gpu_metrics \
    python my_benchmark.py

# 强制覆盖已有输出
nsys profile \
    --trace=cuda,nvtx \
    --force-overwrite=true \
    --output=my_report \
    python my_benchmark.py
```

### Timeline 分析

nsys 的 `.nsys-rep` 文件需要用 Nsight Systems GUI 打开（`nsys-ui`），或者导出为 SQLite/CSV 后用脚本分析。

**Timeline 中能看到的关键信息**：

```
时间线（从上到下）：
┌──────────────────────────────────────────────────────┐
│ CUDA API (CPU)     [cudaLaunchKernel] [cudaMemcpy]  │  ← CPU 端 API 调用
│ CUDA Kernels (GPU)   ████████████████  ██████        │  ← GPU kernel 执行
│ CUDA Memcpy (GPU)         ██████            ██████    │  ← H2D/D2H 传输
│ NVTX Ranges (CPU)   ├──── FlashAttn ────┤ ├─RMSNorm─┤│  ← 你标记的 range
│ CPU Threads         ░░░░  ░░░░  ░░░░  ░░░░           │  ← CPU 端计算
└──────────────────────────────────────────────────────┘
```

**关注这些 Patterns**：

1. **GPU 空闲间隙**：CUDA Kernels 行中的空白段 → CPU 端有瓶颈或 kernel launch 开销过大
2. **Memcpy 不重叠**：CUDA Memcpy 和 Kernel 执行没有重叠 → 缺少 async copy + stream overlap
3. **大量小 kernel**：很多细碎的 kernel 之间存在间隙 → launch overhead > kernel 本身
4. **NVTX 标记的段过长**：某个 range 占据了主要时间 → 这个操作是瓶颈

### 关键指标

```bash
# 导出 kernel 统计
nsys stats --report cuda_gpu_kern_sum timeline_report.nsys-rep \
    --format csv --output kernel_stats.csv

# 导出 CUDA API 统计
nsys stats --report cuda_api_sum timeline_report.nsys-rep \
    --format csv --output api_stats.csv

# 查看所有可用报告类型
nsys stats --list-reports timeline_report.nsys-rep
```

**重要指标判断**：

| 场景 | 判断标准 | 优化方向 |
|------|---------|----------|
| kernel 时间 / 总时间 < 50% | CPU 端有大量开销 | 检查 Python 代码、数据预处理 |
| H2D 拷贝时间 > kernel 时间 | 数据拷贝是瓶颈 | Pinned memory、数据预处理后置 on GPU |
| 大面积 GPU idle | Launch 或 CPU 端延迟 | 减少 kernel launch 次数、CUDA Graph |
| cudaDeviceSynchronize 时间长 | 不必要的同步 | 移除显式同步、使用 cudaStreamWaitEvent |

### CUDA API 调用分析

在 nsys timeline 中，CUDA API 行显示的是 CPU 端调用的 CUDA 运行时 API。

**关注**：
- `cudaMalloc` / `cudaFree` 占用时间过长 → 使用内存池 / caching allocator
- `cudaMemcpy` 占用时间过长 → 考虑 pinned memory / async copy
- `cudaLaunchKernel` 之间有间隙 → launch overhead 过高

### CPU-GPU 同步开销

在 nsys 中可以看到 `cudaDeviceSynchronize` 和 `cudaStreamSynchronize` 的调用。

```bash
# 统计同步调用的时间
nsys stats --report cuda_api_sum timeline_report.nsys-rep
# 筛选包含 "Synchronize" 的行
```

**Golden Rule**：在生产代码中，除了必要的 checkpoint（如验证正确性后），应该尽量避免显式 `cudaDeviceSynchronize()`。如果有多个任务需要同步，使用 `cudaStreamWaitEvent` 替代全局同步。

### NVTX 标记

NVTX（NVIDIA Tools Extension）允许你在代码中插入标记，让 nsys timeline 显示你的代码段名称。

**Python 用法**：

```python
import torch

# 方式 1：手动 push/pop
torch.cuda.nvtx.range_push("attention_forward")
output = attention(query, key, value)
torch.cuda.nvtx.range_pop()

# 方式 2：上下文管理器
class NvtxRange:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        torch.cuda.nvtx.range_push(self.name)
    def __exit__(self, *args):
        torch.cuda.nvtx.range_pop()

with NvtxRange("fused_ffn"):
    x = gate_proj(x)
    x = up_proj(x)
    x = silu(gate) * up

# 方式 3：装饰器（生产环境中常见）
def nvtx_wrap(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            torch.cuda.nvtx.range_push(name)
            result = func(*args, **kwargs)
            torch.cuda.nvtx.range_pop()
            return result
        return wrapper
    return decorator

@nvtx_wrap("flash_attn_kernel")
def flash_attention(q, k, v):
    ...
```

**C++ 用法**：

```cpp
#include <nvtx3/nvToolsExt.h>

void my_function() {
    nvtxRangePushA("flash_attention_fwd");
    // ... kernel launch ...
    nvtxRangePop();
}

// 或使用 RAII wrapper
class NvtxScope {
public:
    NvtxScope(const char* name) { nvtxRangePushA(name); }
    ~NvtxScope() { nvtxRangePop(); }
};
```

**运行带 NVTX 的 profiling**：

```bash
nsys profile --trace=cuda,nvtx,osrt -o nvtx_report python script.py
```

### 实战：多 Stream Pipeline 的 Overlap 效率分析

```bash
nsys profile \
    --trace=cuda,nvtx,osrt \
    --gpu-metrics-device=0 \
    --output=stream_overlap_report \
    python benchmark_cuda_streams.py
```

**在 timeline 中检查**：

```
预期（良好的 overlap）：
Stream 0: [H2D_0] [██████ Kernel_0 ██████] [D2H_0]
Stream 1:   [H2D_1] [████ Kernel_1 ████] [D2H_1]
Stream 2:     [H2D_2] [███ Kernel_2 ███] [D2H_2]
           ↑ H2D 和 Kernel 重叠，整体时间缩短

不良（没有 overlap）：
Stream 0: [H2D_0]─────────────────[██████ Kernel_0 ██████]──────────[D2H_0]
Stream 1: ─────────────────────────────[H2D_1]───[███ Kernel_1 ███]──[D2H_1]
           ↑ 各 stream 串行执行，整体时间 = 各部分时间之和
```

**overlap 效率计算**：
```
overlap_efficiency = (sum(kernel_times) + sum(transfer_times) - total_wall_time) / min(sum(kernel_times), sum(transfer_times))
```

- > 80%：优秀
- 50-80%：一般
- < 50%：需要优化（检查是否使用了 pinned memory、是否使用了默认 stream）

### nsys 报告导出

```bash
# 导出 kernel 统计数据为 CSV
nsys stats --report cuda_gpu_kern_sum report.nsys-rep --format csv --output kernels.csv

# 导出为 SQLite 数据库（支持复杂查询）
nsys stats --report cuda_gpu_kern_sum report.nsys-rep --format sqlite --output kernels.sqlite

# 列出所有报告类型
nsys stats --list-reports report.nsys-rep

# 常见报告类型
# cuda_gpu_kern_sum    - GPU kernel 执行摘要
# cuda_gpu_mem_time_sum - GPU 内存操作时间摘要
# cuda_api_sum         - CUDA API 调用摘要
# nvtx_sum             - NVTX range 摘要
# osrt_sum             - OS 运行时摘要
```

---

## 性能优化决策树

```
延迟高？
│
├─ 第一步：用 nsys 确认瓶颈位置
│  │
│  ├─ GPU kernel 时间占总时间 < 50%？
│  │  └─ 是 → 瓶颈在 CPU 端
│  │     ├─ 数据预处理（tokenization、padding 等）
│  │     │  → 用 perf 分析热点函数 → 优化算法 / SIMD 向量化
│  │     ├─ Python 开销
│  │     │  → 减少 Python 循环 / 使用 torch.compile / CUDA Graph
│  │     └─ CUDA API 调用开销
│  │        → 减少 cudaMalloc/cudaFree / 使用内存池
│  │
│  └─ GPU kernel 时间占总时间 > 50%？
│     └─ 是 → 瓶颈在 GPU 端，用 ncu 深入分析
│
├─ 第二步：用 ncu 判断瓶颈类型
│  │
│  ├─ Speed of Light 显示 Memory Throughput > 70%？
│  │  └─ 是 → Memory-bound（显存带宽是瓶颈）
│  │     ├─ 使用 operator fusion（减少中间结果读写）
│  │     ├─ 使用 tiling（增加数据复用）
│  │     ├─ 使用 shared memory（减少全局内存访问）
│  │     ├─ 使用 vectorized load/store（float4/float2 替代 float）
│  │     └─ 优化内存访问模式（coalesced access、对齐）
│  │
│  ├─ Speed of Light 显示 Compute Throughput > 70%？
│  │  └─ 是 → Compute-bound（计算是瓶颈）
│  │     ├─ 使用 Tensor Core（mma.sync / wmma）
│  │     ├─ 增加 tiling block size（更大数据复用 → 更多计算密度）
│  │     ├─ warp-level 优化（减少分支、使用 warp shuffle）
│  │     └─ 使用 FP8/INT8 量化（增加计算吞吐）
│  │
│  └─ 两者都低（< 50%）？
│     └─ 是 → Latency-bound（延迟是瓶颈）
│        ├─ 提高 occupancy（减少 register / shared memory 使用）
│        │  ├─ 减少局部变量
│        │  ├─ 减少 shared memory 分配
│        │  └─ 调整 block size
│        ├─ 减少同步 barrier
│        │  ├─ 只在必要时用 __syncthreads()
│        │  └─ 使用 warp-level 原语替代 block-level
│        ├─ 多 stream overlap
│        │  ├─ 使用多个 CUDA stream
│        │  ├─ 异步 H2D/D2H
│        │  └─ 用 cudaStreamWaitEvent 而非全局同步
│        └─ 减少 kernel launch 次数
│           ├─ 合并小 kernel
│           └─ 使用 CUDA Graph
│
└─ 第三步：验证优化效果
   ├─ 用 ncu --compare baseline.ncu-rep 对比优化前后
   ├─ 用 nsys 确认 timeline 改善
   └─ 用 benchmark 测量实际性能提升
```

---

## 常见性能反模式

### 反模式 1：忘记 `.contiguous()` 导致 Strided Access

**问题**：PyTorch tensor 在 permute/transpose/slice 后可能不是连续存储。非连续 tensor 传递给 CUDA kernel 或做 `.to(device)` 时，内存访问变成 strided，性能急剧下降（降低 3-10x）。

```python
# 反模式
x = torch.randn(128, 4096).cuda()
x_t = x.transpose(0, 1)  # shape [4096, 128]，但不是 contiguous！
output = my_kernel(x_t)   # strided memory access → 性能灾难

# 正确做法
x_t = x.transpose(0, 1).contiguous()  # 显式重新排列内存
output = my_kernel(x_t)               # 连续内存访问
```

**ncu 特征**：`l1tex__throughput` 低、`dram__throughput` 非常高、每个 global load 有多个 sectors。

### 反模式 2：默认 Stream 隐式同步

**问题**：PyTorch 默认使用 legacy default stream 模式，在默认 stream 上的任何操作都会与所有其他 stream 隐式同步。

```python
# 反模式
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

with torch.cuda.stream(stream1):
    kernel_a(x1)  # 在 stream1 上

kernel_b(x2)  # 在默认 stream 上 → 隐式等待 stream1 完成！

with torch.cuda.stream(stream2):
    kernel_c(x3)  # 在 stream2 上 → 被 kernel_b 阻塞！
```

**正确做法**：所有并发操作都使用显式 stream。

```python
stream_a = torch.cuda.Stream()
stream_b = torch.cuda.Stream()

with torch.cuda.stream(stream_a):
    kernel_a(x1)
with torch.cuda.stream(stream_b):
    kernel_b(x2)
```

**nsys 特征**：Timeline 上 kernel 串行执行，即使应该可以并行的。

### 反模式 3：过多小 Kernel Launch

**问题**：每次 CUDA kernel launch 有约 5-15 us 的 CPU 端开销。如果一个 kernel 本身只执行 5 us，但 launch 用了 10 us，效率只有 33%。

```python
# 反模式：对每个元素启动一个 kernel
for i in range(1000):
    tiny_kernel(x[i:i+1])  # launch overhead >> kernel 执行时间

# 正确做法：合并为单个 kernel
big_kernel(x)  # 一次 launch 处理所有数据
```

**nsys 特征**：Timeline 上大量微小 kernel 之间有间隙。

**量化判断**：
```python
# 如果 kernel 执行时间 < 5x launch overhead（~50 us），考虑合并
if kernel_time_us < 50:
    print("考虑与其他 kernel 合并以分摊 launch 开销")
```

### 反模式 4：Shared Memory Bank Conflict

**问题**：Shared memory 按 bank（32 个 bank，每个 4 bytes）组织。如果同一个 warp 内的多个线程访问同一个 bank 的不同地址，就会发生 bank conflict，访问串行化。

```cpp
// 反模式：stride=32 导致所有线程访问同一个 bank
__shared__ float smem[1024];
int idx = threadIdx.x;
float val = smem[idx * 32];  // 每 32 个线程访问 bank 0（bank conflict × 32）

// 修复：添加 padding 改变 stride
__shared__ float smem[1024 + 32];  // 多加一行防止冲突
float val = smem[idx * 33];  // stride=33，不同线程访问不同 bank
```

**ncu 特征**：`l1tex__data_pipe_lsu_wavefronts_shared` > 0 说明有 bank conflict。

### 反模式 5：Register Spilling

**问题**：Kernel 使用了太多局部变量（超过 SM 的 register file 限制），编译器被迫将这些变量"溢出"到 L1 cache 或 local memory，每次访问增加几十个周期的延迟。

```cpp
// 反模式：声明过多自动变量
__global__ void my_kernel() {
    float a1, a2, a3, ..., a96;  // 96 个 float，96 个寄存器
    float b1, b2, ..., b64;       // 64 个，总共 160 寄存器 → spill!
    // ... 使用这些变量 ...
}

// 修复：拆分为更小的 kernel 或减少局部变量
__global__ void my_kernel_part1() { /* 第一部分计算 */ }
__global__ void my_kernel_part2() { /* 第二部分计算，复用寄存器 */ }
```

**ncu 特征**：
- `registers_per_thread` 过高（A100 上 > 128）
- `l1tex__data_pipe_lsu_wavefronts_shared` > 0
- occupancy 异常低

### 反模式 6：未使用 Pinned Memory

**问题**：非 pinned host memory 不允许 GPU DMA 引擎直接访问。CUDA 驱动必须先做一次 CPU 端内存拷贝到 staging buffer，然后 GPU 才能读取。这不仅增加了拷贝时间，更重要的是让拷贝变成同步的。

```python
# 反模式
data = torch.randn(1024, 1024)  # 普通 pageable memory
data_cuda = data.cuda()  # 同步拷贝！中间有隐式的 staging copy

# 正确做法
data = torch.randn(1024, 1024, pin_memory=True)  # pinned memory
data_cuda = data.cuda(non_blocking=True)  # 真正的异步拷贝

# 或使用 DataLoader 的 pin_memory
loader = DataLoader(dataset, pin_memory=True)
```

**nsys 特征**：H2D 拷贝时间长，且无法与 GPU kernel 重叠。

### 反模式 7：Warp Divergence

**问题**：同一个 warp 内的线程走不同的分支路径（由于 if/else），导致 SIMT 执行变成串行。

```cpp
// 反模式：data-dependent branch 导致 warp divergence
__global__ void process(float *x, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (x[idx] > 0) {
        x[idx] = sqrt(x[idx]);     // 某些线程走这里
    } else {
        x[idx] = -sqrt(-x[idx]);   // 其他线程走这里 → 串行化
    }
}

// 修复：用符号函数 + 数学运算替代分支
__global__ void process_optimized(float *x, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float sign = (x[idx] > 0) ? 1.0f : -1.0f;
    x[idx] = sign * sqrt(fabsf(x[idx]));  // 所有线程统一的计算路径
}
```

**ncu 特征**：`sm__warps_issue_stalled_short_scoreboard` 较高；`sm__sass_branch_s diverging` 显示分支发散。

---

## 快速诊断流程总结

```
发现性能问题
  │
  ├─ 1. nsys profile → 找到延迟最长的阶段（CPU or GPU？哪个 kernel？）
  │
  ├─ 2. 如果瓶颈是 GPU kernel：
  │     ncu --set basic → 看 Speed of Light
  │     ├─ Memory Tput > 70% → Memory-bound
  │     │    → ncu --set memory → 看 cache hit rate / coalescing
  │     ├─ Compute Tput > 70% → Compute-bound
  │     │    → ncu --set compute → 看 FMA/Tensor Core 利用率
  │     └─ 两者都低 → Latency-bound
  │          → ncu --section Occupancy → 看 register/shared memory
  │
  ├─ 3. 如果瓶颈是 CPU：
  │     perf stat → 看 IPC
  │     ├─ IPC < 1 → perf record → 找热点函数 → perf annotate
  │     └─ IPC ≈ 2+ → SIMD 效率好，可能是调度/API 调用问题
  │          → nsys 看 CUDA API timeline
  │
  └─ 4. 优化后 → ncu --compare baseline.ncu-rep → 确认指标改善
```
