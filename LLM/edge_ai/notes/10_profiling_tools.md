# 性能剖析工具实战指南

## 1. 为什么要掌握 Profiling 工具

在机器人系统性能工程中，优化的第一步永远是**测量**。没有数据的优化是盲目的。一套完整的 profiling 工具链能帮助你回答以下问题：

- CPU 在干什么？哪些函数消耗了最多的时间？
- GPU 是否在空转等待？kernel 启动延迟是多少？
- 内存分配是否成为瓶颈？是否存在内存泄漏？
- 系统调度的抖动来自哪里？中断处理占用了多少时间？
- I/O 等待是否阻塞了关键路径？

本章将系统介绍从应用层到内核层的完整 profiling 工具栈。

## 2. NVIDIA Nsight Systems：GPU 时间线分析

### 2.1 工具定位

Nsight Systems 是 NVIDIA 官方提供的系统级性能分析工具，适用于 CUDA 应用程序。它能以时间线（timeline）形式展示 CPU 和 GPU 之间的交互，帮助识别 GPU 空闲、kernel 启动开销、数据传输延迟等问题。

### 2.2 基本用法

```bash
# 采集性能数据
nsys profile -o output_report ./your_cuda_app

# 指定采集范围（CUDA API 追踪 + OS 运行时）
nsys profile --trace=cuda,osrt,nvtx,cublas ./your_app

# 生成时间线报告
nsys stats output_report.qdrep
```

### 2.3 时间线解读要点

打开 `.qdrep` 文件后，关键关注以下几个维度：

- **CUDA API 行**：显示 cudaMalloc、cudaMemcpy、cudaLaunchKernel 等 API 调用。注意 API 调用和实际 GPU 执行之间的**异步间隙**。
- **CUDA Kernel 行**：显示 GPU 上 kernel 的实际执行时间。如果这行有大量空白，说明 GPU 在空转。
- **CUDA Memcpy 行**：数据传输（Host ↔ Device）。Pinned memory 可以显著缩短此时间。
- **NVTX 区间**：你在代码中插入的标记（`nvtxRangePush/Pop`），用于关联源码逻辑与时间线。

### 2.4 常见性能问题诊断

| 症状 | 可能原因 | 解决方向 |
|------|----------|----------|
| GPU kernel 之间有长间隙 | kernel launch 开销过大 | 合并小 kernel，使用 CUDA Graph |
| Memcpy 占用大量时间 | 使用了 pageable memory | 改用 pinned (page-locked) memory |
| 连续 kernel 之间 GPU 空闲 | CPU 端串行准备数据 | 使用多 stream 并发 |
| 第一个 kernel 延迟特别大 | CUDA context 惰性初始化 | 预热 (warm-up) 一次 |

## 3. Linux perf：CPU 性能计数器

`perf` 是 Linux 内核自带的性能分析工具，利用硬件性能计数器（PMC）提供微观层面的 CPU 行为洞察。

### 3.1 perf stat：快速统计

```bash
# 基础统计：执行时间、context switch、CPU 迁移
perf stat ./your_program

# 关注 IPC（Instructions Per Cycle）和缓存行为
perf stat -e cycles,instructions,cache-references,cache-misses,branch-misses ./your_program

# 实时监控（类似 top）
perf top
```

**IPC 解读**：IPC < 1.0 表示 CPU 大量时间在等待内存（memory-bound）；IPC > 2.0 表示代码 compute-bound 且流水线效率高。

### 3.2 perf record + perf report：采样分析

```bash
# 采样 CPU 调用栈（默认 4000 Hz）
perf record -g ./your_program

# 生成报告
perf report

# 火焰图数据导出
perf script > out.perf
```

### 3.3 关键硬件事件

| 事件 | 含义 | 高值说明 |
|------|------|----------|
| `cache-misses` | L1/L2/L3 缓存未命中次数 | 数据访问模式差 |
| `branch-misses` | 分支预测失败次数 | 大量不可预测的 if/switch |
| `context-switches` | 上下文切换次数 | 线程过多或 I/O 密集 |
| `cpu-migrations` | 线程在不同核心间迁移 | 应使用 CPU affinity |
| `page-faults` | 缺页异常 | 内存分配或 mmap 问题 |
| `stalled-cycles-frontend` | 前端停顿周期 | 指令缓存未命中 |
| `stalled-cycles-backend` | 后端停顿周期 | 内存带宽瓶颈 |

## 4. eBPF：内核可观测性

eBPF (extended Berkeley Packet Filter) 允许你在内核中运行沙箱化的程序，无需修改内核源码即可实现深度可观测性。

### 4.1 bpftrace：高级追踪语言

```bash
# 追踪某个函数的调用延迟
bpftrace -e 'kprobe:vfs_read { @start[tid] = nsecs; }
             kretprobe:vfs_read /@start[tid]/ {
                 @lat_us = hist((nsecs - @start[tid]) / 1000);
                 delete(@start[tid]); }'

# 统计系统调用分布
bpftrace -e 'tracepoint:syscalls:sys_enter_* { @[probe] = count(); }'

# 追踪进程调度延迟
bpftrace -e 'tracepoint:sched:sched_switch { @lat_ns = hist(nsecs - args->prev_state); }'
```

### 4.2 BCC 工具集

BCC (BPF Compiler Collection) 提供即开即用的追踪工具：

- `execsnoop`：追踪所有新进程的创建
- `biolatency`：块设备 I/O 延迟分布
- `tcptop`：TCP 流量监控
- `funclatency`：函数级延迟分析
- `cachestat`：页面缓存命中率

### 4.3 在机器人场景中的应用

- **函数延迟追踪**：对控制循环中的关键函数（如 MPC 求解器）进行延迟直方图分析
- **调度延迟监控**：确认 `SCHED_FIFO` 线程是否被及时调度
- **I/O 分析**：检查日志写入是否阻塞了实时线程
- **内存分配追踪**：确认是否在实时路径中存在 `malloc` 调用

## 5. FlameGraph：火焰图

Brendan Gregg 发明的火焰图是性能剖析可视化的黄金标准。

### 5.1 CPU 火焰图（On-CPU）

```bash
# 采集 30 秒的调用栈数据（99 Hz 采样）
perf record -F 99 -g -p <PID> -- sleep 30

# 生成火焰图
perf script | stackcollapse-perf.pl | flamegraph.pl > cpu_flamegraph.svg
```

**解读**：X 轴宽度代表函数在采样中的占比，Y 轴代表调用栈深度。宽且平的"高原"是优化重点。

### 5.2 Off-CPU 火焰图

```bash
# 记录进程被阻塞的位置
perf record -e 'sched:sched_switch' -e 'sched:sched_stat_sleep' \
    -e 'sched:sched_stat_blocked' -g -p <PID> -- sleep 30
```

Off-CPU 火焰图展示的是**线程不在 CPU 上运行**的时间花在哪里（等锁、等 I/O、主动睡眠等），对于分析延迟问题至关重要。

### 5.3 其他火焰图变体

- **Memory 火焰图**：追踪内存分配调用栈
- **I/O 火焰图**：追踪磁盘 I/O 延迟
- **Cold/Warm 火焰图**：差分火焰图，对比两次 profiling 结果

## 6. 内存 Profiling 工具

### 6.1 Valgrind / Cachegrind

```bash
# 缓存模拟分析
valgrind --tool=cachegrind ./your_program

# 查看缓存未命中率
cg_annotate cachegrind.out.<pid>
```

Cachegrind 模拟 L1/L2 缓存的访问模式，输出每个函数/行的缓存命中率。适合精确诊断缓存未命中热点。

### 6.2 Heaptrack：堆内存分析

```bash
# 采集内存分配数据
heaptrack ./your_program

# 图形化分析（火焰图+调用树）
heaptrack_gui heaptrack.your_program.<pid>.gz
```

Heaptrack 可以回答：谁分配了最多内存？是否存在持续增长的内存占用？哪些分配是临时的（可优化掉的）？

### 6.3 Massif（Valgrind 工具）

```bash
valgrind --tool=massif --time-unit=B ./your_program
ms_print massif.out.<pid>
```

Massif 展示堆内存的快照随时间变化，适合诊断内存泄漏和内存使用模式。

## 7. 工具选择决策树

面对性能问题时，快速选择正确的工具：

```
性能瓶颈类型？
├── CPU 计算密集
│   ├── 需要宏观热点 → perf record + FlameGraph
│   ├── 需要微观分析 → perf stat (IPC, cache miss)
│   └── 需要 GPU 视角 → Nsight Systems
├── 内存瓶颈
│   ├── 缓存效率 → cachegrind
│   ├── 分配频率 → heaptrack
│   └── 内存泄漏 → massif / valgrind memcheck
├── I/O 瓶颈
│   ├── 磁盘 → biolatency (BCC)
│   └── 网络 → tcptop (BCC)
├── 调度/延迟抖动
│   ├── 宏观 → perf sched
│   ├── 微观 → bpftrace (sched tracepoints)
│   └── 端到端 → cyclictest / oslat
└── 不确定
    └── 先用 perf top → 定位热点类别 → 选择子工具
```

## 8. 性能优化工作流

**核心循环：假设 → 测量 → 修复 → 验证**

1. **假设阶段**：基于对系统的理解，形成性能瓶颈的假设。例如："Planning 阶段慢是因为 RRT 搜索空间太大"。
2. **测量阶段**：使用上述工具采集数据，验证或推翻假设。
   - 不要跳过测量直接"优化"——你可能在优化错误的代码。
   - 确保测量环境稳定：关闭无关进程，固定 CPU 频率，禁止动态调频。
3. **修复阶段**：基于数据实施优化。优先优化热点路径（二八原则）。
4. **验证阶段**：重新测量，确认优化效果。记录每次优化的前后对比数据。
5. **回归保护**：将性能指标纳入 CI/CD，防止后续改动引入性能退化。

**最重要的原则**：永远不要猜测性能瓶颈在哪里。数据驱动，让工具告诉你真相。
