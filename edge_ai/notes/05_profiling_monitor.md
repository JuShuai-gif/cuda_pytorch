# 性能剖析与监控

## 1. 延迟分解分析方法论

### 1.1 核心思想：每一微秒都需归因

性能优化的第一原则是**测量优先**。在不知道瓶颈在哪的情况下做优化，等于蒙眼射箭。延迟分解分析（Latency Breakdown）的目标是将端到端延迟拆解为各子阶段的耗时，找到最大的贡献者。

### 1.2 逐阶段打点法

```cpp
// Instrument every stage boundary
auto t0 = now();
preprocess(data);
auto t1 = now();
detect(data);
auto t2 = now();
track(detections);
auto t3 = now();
plan(tracks);
auto t4 = now();

// Report breakdown: preprocess=t1-t0, detect=t2-t1, track=t3-t2, plan=t4-t3
```

**关键**：打点要包围完整的处理逻辑，包括排队等待时间（在队列中取到数据的时间也算在对应 stage 内）。

### 1.3 Tracing 全链路追踪

分布式系统中使用 Trace ID + Span ID 在全链路跟踪每一帧。每个 Span 记录：
- start/end timestamp
- stage name
- parent span ID

所有 Span 构成一棵树，总延迟 = 根 Span 的 duration。

**工具**：Jaeger、Zipkin、Perfetto（Android/Chrome）、LTTng（Linux 内核级）。

## 2. 火焰图（Flame Graph）

### 2.1 如何阅读火焰图

- X 轴 = 采样占比（按字母排序合并同类调用栈），宽度越大 = CPU 时间占比越高
- Y 轴 = 调用栈深度，从下到上是调用链
- 颜色无特殊含义（通常随机或按函数名 hash）

**查找瓶颈的方法**：从顶部平台（plateau）向下看，宽而平的部分是自耗 CPU 多的函数。

### 2.2 生成火焰图

```bash
# 采样（99Hz，避免与定时器同频）
perf record -F 99 -g -p <pid> -- sleep 30

# 生成火焰图
perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg
```

### 2.3 火焰图类型

| 类型 | 工具 | 用途 |
|------|------|------|
| CPU (on-CPU) | `perf` + flamegraph | 找 CPU 热点函数 |
| Memory | `perf mem record` | 找内存访问热点 |
| Off-CPU | `perf record -e sched:sched_switch` | 找阻塞等待原因 |
| I/O | `perf trace` + flamegraph | 找 I/O 延迟来源 |

### 2.4 Off-CPU 分析的重要性

CPU 火焰图只能看到**正在执行**的代码。如果线程大部分时间在等锁、等 I/O、等 GPU，CPU 火焰图看不到这些。Off-CPU 火焰图显示线程在阻塞状态时的调用栈，是定位锁竞争和 I/O 等待的关键工具。

## 3. GPU 时间线分析（Nsight Systems）

### 3.1 Nsight Systems 视图

- **Timeline 视图**：显示 CPU 线程和 GPU 流（stream）、kernel、memcpy 的时间分布
- **关键观察**：CPU 和 GPU 之间的"气泡"（空白区域）= 两者同时在等待，说明并行度不足
- **CUDA Stream 重叠**：理想情况下，多个 stream 的 kernel 和 memcpy 应交替执行，不串行等待

### 3.2 常见 GPU 时间线问题

1. **Kernel 序列化（Serialized Kernels）**：所有 kernel 在默认 stream 中依次执行 → 使用多个 CUDA stream
2. **大 Memcpy 阻塞**：CPU→GPU 数据传输期间 GPU 空闲 → 使用异步 memcpy + stream 重叠
3. **Kernel Launch 间隙**：kernel 之间存在较大空白 → 合并 kernel 或使用 CUDA Graph
4. **CPU 端瓶颈**：GPU 空闲但 CPU 在忙 → CPU 端计算需要优化或多线程化

## 4. 插桩方法

### 4.1 手动插桩（Timer-based）

```python
class TimerContext:
    def __enter__(self):
        self.start = time.perf_counter()
    def __exit__(self, ...):
        self.elapsed = time.perf_counter() - self.start
```

**优点**：精确、零歧义、可自定义上报维度（stage name、frame id）。
**缺点**：侵入式（需修改代码）、插桩本身有开销（通常 < 100ns，可忽略）。

### 4.2 编译器插桩（Compiler Instrumentation）

`gcc -finstrument-functions` 在每个函数入口/出口插入 `__cyg_profile_func_enter` / `__cyg_profile_func_exit` 调用。

**优点**：全自动，覆盖所有函数。
**缺点**：开销大（每个函数调用增加 ~200ns），输出海量数据。通常只在特定编译单元启用。

### 4.3 采样（Sampling-based）

`perf record` 以固定频率中断 CPU，记录当前指令指针和调用栈。

**优点**：零代码侵入，生产环境安全（开销 < 3%），统计意义上精确。
**缺点**：无法追踪单次调用的延迟分布（只能看比例），短函数可能被遗漏（采样偏差）。

### 4.4 混合策略

推荐：**采样定位热点 → 手动插桩精细分析**。先用 perf/火焰图找到可疑函数，再在关键路径上添加精确计时。

## 5. 关键指标

### 5.1 核心指标

| 指标 | 含义 | 目标 |
|------|------|------|
| Frame Time | 单帧端到端处理时间 | < 延迟预算 |
| Frame Rate (FPS) | 每秒处理帧数 | > 目标 FPS |
| Queue Depth | 各阶段队列中等待的数据量 | < 2-3（避免排队延迟累积） |
| GPU Utilization | SM 活跃占比 | 80%+ 但需配合计算效率 |
| Memory Bandwidth | DRAM 读/写吞吐 | < 理论带宽 80% |
| CPU Utilization | 核心使用率 | 70-80%，留余量应对突发 |

### 5.2 延迟分布统计

均值不可靠！必须关注 P50、P95、P99、P99.9 尾延迟。对于实时系统，P99.9 才是真实用户体验。

```python
sorted_latencies = sorted(all_frame_latencies)
p50  = sorted_latencies[int(len * 0.50)]
p99  = sorted_latencies[int(len * 0.99)]
p999 = sorted_latencies[int(len * 0.999)]
```

### 5.3 抖动（Jitter）

定义：`jitter = max_latency - min_latency` 或标准差。

来源：
- 操作系统调度抖动（timer interrupt, RCU callback）
- 垃圾回收（GC）暂停
- 动态频率调整（DVFS, thermal throttling）
- 缓存/TLB 抖动（cache pollution from co-running workloads）

**缓解**：`isolcpus` 隔离核心、`nohz_full` 关闭定时中断、关闭 CPU 频率调节、使用 real-time 调度（SCHED_FIFO）。

## 6. 构建生产级剖析基础设施

### 6.1 架构设计

```
App → [Metrics SDK] → [Aggregation Daemon] → [Time-Series DB] → [Dashboard]
       ├── LatencyTracker (P50/P99/histogram)
       ├── QueueMonitor (depth gauge)
       └── ThroughputCounter (FPS counter)
```

**原则**：
1. 指标采集开销 < 1%（用原子变量 + 无锁队列）
2. 聚合在独立线程，不阻塞热路径
3. 支持动态采样率（运行时调整，减少生产环境开销）
4. 数据写入时间序列库（InfluxDB/Prometheus），用 Grafana 展示

### 6.2 自诊断能力

当检测到延迟超过阈值时，自动触发栈采样（如 `SIGPROF` + backtrace），保存 crash dump 级别的时间线快照。

## 7. 常见剖析反模式

| 反模式 | 问题 | 正确做法 |
|--------|------|---------|
| 只看平均值 | 掩盖尾延迟问题 | 必须看 P99/P99.9 分布 |
| 只看 CPU 火焰图 | 忽略 I/O/锁阻塞 | 补充 off-CPU 火焰图 |
| perf 采样在虚拟机上 | hypervisor 偷时间 | 在裸金属上测试 |
| 插桩代码在热循环中 | printf 开销压倒实际计算 | 用无锁 ring buffer + 批量写入 |
| 只测单个函数 | 忽略排队和同步等待 | 测量端到端路径（含等待） |
| GPU 只看 nvidia-smi | 不清楚 kernel 级细节 | 用 Nsight Systems 看时间线 |
| 优化前不建立基线 | 不知道是否真的变快了 | 每次优化前后用相同负载测量 |
| 只在一个负载下测试 | 低负载快 != 高负载也快 | 用梯度负载（10%/50%/100%）测试 |
