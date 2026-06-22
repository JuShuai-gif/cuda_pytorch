# 端到端系统流水线性能优化

## 1. 什么是机器人/自动驾驶的端到端流水线

端到端流水线（End-to-End Pipeline）是指从传感器采集原始数据到执行器输出控制指令的完整处理链路。在自动驾驶系统中，典型流程为：

```
传感器(摄像头/LiDAR/Radar/IMU) → 感知(Perception) → 决策(Planning/Decision) → 控制(Control) → 执行器(油门/刹车/转向)
```

每个阶段都是一个有向无环图（DAG）中的节点，数据以流的形式在节点间传递。流水线的核心目标是：**在给定硬件资源下，最大化吞吐量（throughput）的同时保证端到端延迟（latency）满足实时性约束**。

## 2. 延迟、吞吐量与尾延迟

### 2.1 延迟（Latency）

**定义**：单个数据包从进入流水线到离开流水线的时间间隔，即端到端响应时间。

- 在自动驾驶中，端到端延迟 = 传感器曝光时刻 → 控制指令发出的时刻
- 典型要求：L4 级自动驾驶 < 100ms（一些安全关键路径 < 50ms）
- 延迟 = 数据采集时间 + 传输时间 + 各阶段处理时间 + 排队等待时间

### 2.2 吞吐量（Throughput）

**定义**：单位时间内流水线能处理的帧数（FPS）。

- 如果流水线是纯串行的，吞吐量 ≈ 1 / 最大单阶段延迟
- 流水线并行化后，吞吐量由最慢阶段决定：T = 1 / max(stage_i)
- 注意：吞吐量和延迟是**不同维度**的指标，优化一个可能损害另一个

### 2.3 尾延迟（Tail Latency / p99 / p999）

**定义**：p99 延迟 = 99% 的请求延迟小于此值。p999 同理。

- 平均延迟（p50）可能很低，但 p99 很高意味着系统有偶发卡顿
- 尾延迟的来源：
  - **GC 暂停**：垃圾回收导致的 stop-the-world
  - **缓存未命中**：冷启动时的缓存回填
  - **锁竞争**：高并发下的互斥等待
  - **中断处理**：硬件中断抢占 CPU
  - **电源管理**：CPU 频率调节延迟
  - **内存带宽争抢**：GPU 与 CPU 同时访问内存
- 尾延迟的缓解：
  - 使用无锁数据结构减少竞争
  - 预分配内存，避免运行时分配
  - 绑核（CPU affinity）+ 中断隔离
  - 关闭 CPU 频率调节（performance governor）
  - 使用实时内核（PREEMPT_RT）

## 3. 抖动（Jitter）来源与缓解

**抖动定义**：延迟在时间轴上的变化幅度，即延迟的方差或标准差。

### 3.1 主要抖动来源

| 来源 | 描述 | 典型量级 |
|------|------|----------|
| 操作系统调度 | CFS 调度器的时间片轮转、抢占 | 100μs ~ 10ms |
| 缓存/TLB 未命中 | 数据不在 CPU cache 中 | 10ns ~ 100ns per miss |
| 内存带宽竞争 | 多个核心同时访问 DRAM | 10% ~ 50% 性能下降 |
| 中断（IRQ） | 网络、磁盘、定时器中断 | 10μs ~ 100μs |
| GPU 内核启动 | CUDA kernel launch overhead | 5μs ~ 20μs |
| 功耗/热限制 | DVFS 降频、thermal throttling | 影响可达 50% |

### 3.2 缓解策略

1. **绑核 + 中断亲和性**：将关键线程绑定到隔离的 CPU 核心，将中断导向其他核心
2. **使用实时调度策略**：`SCHED_FIFO` 或 `SCHED_RR` 替代默认 `SCHED_OTHER`
3. **锁存数据**：预加载关键数据到 L1/L2 缓存（prefetch）
4. **忙等待替代阻塞**：在延迟敏感路径用 spin-wait 替代 mutex`cond_wait
5. **固定频率运行**：设置 CPU governor 为 `performance`，GPU 为固定频率

## 4. 感知 → 决策 → 控制流水线结构

### 4.1 感知阶段（Perception）

**输入**：传感器原始数据（图像、点云、IMU 数据等）
**输出**：环境模型（目标检测框、车道线、可行驶区域、目标跟踪轨迹）

典型子任务：
- 图像预处理（去畸变、resize、normalize）
- 目标检测（YOLO、CenterNet 等）
- 语义分割（道路、行人、车辆分类）
- 深度估计 / 3D 目标检测
- 多传感器融合（Camera + LiDAR 前融合/后融合）
- 目标跟踪（Kalman Filter、匈牙利匹配）

**时延预算参考**：30ms ~ 80ms（GPU 推理为主）

### 4.2 决策/规划阶段（Planning/Decision）

**输入**: 环境模型 + 自车状态
**输出**：轨迹（trajectory）、行为决策（换道/跟车/停车）

典型子任务：
- 行为预测（周围车辆/行人未来 5s 轨迹）
- 行为决策（有限状态机 or 强化学习）
- 路径规划（A* / RRT / Lattice Planner）
- 速度规划（S-T 图 + 动态规划）
- 轨迹优化（平滑、曲率约束）

**时延预算参考**：20ms ~ 50ms（CPU 计算为主）

### 4.3 控制阶段（Control）

**输入**：目标轨迹 + 车辆状态
**输出**：执行器指令（油门、刹车、转向角）

典型子任务：
- 横向控制（Pure Pursuit / Stanley / LQR / MPC）
- 纵向控制（PID / MPC）
- 车辆动力学模型（自行车模型）
- 控制指令平滑与限幅

**时延预算参考**：5ms ~ 10ms（要求确定性低延迟）

## 5. 各阶段测量与优化方法

### 5.1 测量手段

- **应用层打点**：在代码中插入 `std::chrono::high_resolution_clock` 时间戳
- **系统级profiling**：`perf record`、`perf stat` 分析 CPU 事件
- **GPU profiling**：`nvprof` / `nsys` / `ncu` 分析 GPU 时间线
- **端到端追踪**：在每个数据包中携带时间戳，记录各阶段耗时
- **分布式追踪**：类似 Jaeger/Zipkin 的思想，跨进程/跨芯片追踪

### 5.2 优化手段

**感知阶段**：
- 模型量化（FP16/INT8 推理）
- TensorRT / ONNX Runtime 图优化
- 算子融合（卷积 + BN + ReLU）
- 多流并发处理不同传感器
- ROI 裁剪减少推理像素

**决策阶段**：
- 搜索空间剪枝
- 增量规划（复用上一帧结果暖启动）
- 多线程并行评估候选轨迹
- 查表法替代在线计算

**控制阶段**：
- 固定执行频率（如 100Hz），多退少补
- 将控制循环绑定到专用核心
- 减少控制指令的 jitter

## 6. 流水线并行概念

### 6.1 时间并行（Temporal Parallelism / Pipelining）

同一时刻，流水线的不同阶段处理不同帧的数据：

```
时刻 t0: Frame0 → Perception
时刻 t1: Frame0 → Planning,  Frame1 → Perception
时刻 t2: Frame0 → Control,    Frame1 → Planning,  Frame2 → Perception
```

吞吐量 = 1 / max(Stage_i_duration)，延迟 = sum(Stage_i_duration)。

### 6.2 空间并行（Spatial Parallelism / Data Parallelism）

将一帧数据拆分给多个计算单元并行处理。例如将图像切分为多个 tile 分别做检测。

### 6.3 关键设计考量

- **缓冲队列大小**：队列太小导致背压、丢帧；队列太大增加延迟
- **反压机制（Back-pressure）**：下游处理不过来时通知上游减速
- **时钟同步**：多传感器时间戳对齐（硬件同步 or 软件插值）
- **掉帧策略**：当负载过重时，有选择地丢弃非关键帧

## 7. 自动驾驶实时性能参考数据

| 指标 | L2/L2+ (ADAS) | L3 (Conditional) | L4 (Highway Pilot) |
|------|---------------|-------------------|--------------------|
| 端到端延迟上限 | < 200ms | < 150ms | < 100ms |
| 感知阶段 | 60~100ms | 40~80ms | 25~50ms |
| 规划阶段 | 30~50ms | 20~40ms | 10~25ms |
| 控制阶段 | 10~20ms | 5~10ms | 1~5ms |
| 传感器帧率 | 10~30Hz | 10~30Hz | 10~30Hz |
| 计算平台功耗 | 10~30W | 50~150W | 200~500W |
| CPU 核心数 | 4~8 | 8~16 | 16~32+ |
| GPU 算力 | 10~30 TOPS | 30~100 TOPS | 200~500+ TOPS |

**关键认知**：不同级别的自动驾驶对实时性的要求差异巨大。L4 级别中，一次超过 100ms 的延迟可能意味着在 100km/h 速度下车辆盲行了 2.8 米——足以导致事故。

## 8. 流水线性能的度量指标总结

1. **P50 延迟**：系统在理想条件下的性能基线
2. **P99 延迟**：系统的稳定性和可预测性
3. **最大延迟**：最坏情况是否能被安全机制兜底
4. **吞吐量**：系统在单位时间内的处理能力
5. **抖动（标准差/方差）**：延迟的一致性
6. **丢帧率**：超过截止时间的帧比例
7. **CPU/GPU 利用率**：硬件资源是否被充分利用
8. **内存带宽利用率**：是否因为内存瓶颈导致计算单元空转

性能工程的本质是：**测量、定位瓶颈、优化、再测量，循环迭代直到满足目标。**

---

## 9. 卡尔曼滤波通俗解释

### 9.1 一句话核心

**你有两个都不太准的信息来源，卡尔曼滤波把它们按"靠谱程度"加权平均，得出一个更靠谱的估计。**

### 9.2 开车进隧道的例子

你开车进隧道，GPS 信号丢了。你现在有两个信息来源：

| 来源 | 怎么来的 | 问题 |
|------|----------|------|
| **预测** | 上一秒你开 60km/h，按这个速度推算现在的位置 | 你可能加减速了，所以不准 |
| **测量** | 隧道里偶尔有个传感器测到你的位置 | 传感器自己也有噪声，也不准 |

卡尔曼滤波做的就是：**我信你一部分，也信传感器一部分**，按各自的"靠谱程度"加权平均：
- 预测方差大 → 多信测量
- 传感器噪声大 → 多信预测

### 9.3 三步循环

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  1. 预测    │ ──→ │  2. 测量    │ ──→ │  3. 更新    │ ──→ 回到 1
│  猜现在在哪  │     │  传感器读数  │     │  加权融合    │
└─────────────┘     └─────────────┘     └─────────────┘
```

1. **预测（Predict）**：根据上一帧状态 + 运动模型（如匀速模型 "位置 += 速度 × dt"），猜现在在哪。同时，不确定性也会增加（因为你不知道是否有加减速）。
2. **测量（Measure）**：拿到传感器观测值（如检测框的 x、y 坐标）。
3. **更新（Update）**：把预测和测量加权融合，得到新状态。同时更新"我有多确定"——越确定，下次越信自己的预测；越不确定，越信新来的测量。

### 9.4 对应到代码中的 KalmanTrack

在 `01_system_pipeline` 中：

```
KalmanTrack {
    x, y, vx, vy    // 4 状态：[位置 x, 位置 y, 速度 x, 速度 y]
    P[4x4]          // 协方差矩阵 = "我有多确定"
    age             // 跟踪了多少帧
    missed          // 连续跟丢了多少帧
}
```

- **predict()**：用匀速模型推算 `x += vx * dt`，`y += vy * dt`，同时 P 增大（不确定性增加）
- **update(measured_x, measured_y)**：用检测框的 x、y 修正状态，P 减小（更确定了）
- **卡尔曼增益 K**：就是那个"加权系数"，公式为 `K = P_predict / (P_predict + R_measurement)`——预测越不确定（P 大），K 越大，越信测量；传感器越噪声（R 大），K 越小，越信预测

### 9.5 为什么比简单平滑更好

| 方法 | 做法 | 问题 |
|------|------|------|
| 低通滤波 | 对历史位置取指数滑动平均 | 不知道物体的真实运动规律，只是"磨平"噪声 |
| 卡尔曼滤波 | 同时利用运动模型 + 测量模型 + 不确定性 | 知道物体在运动，能区分"真走了"和"测错了" |

---

## 10. 代码中其他关键算法

### 10.1 三次样条轨迹（Cubic Spline）

规划阶段生成的轨迹不是折线，而是光滑曲线。三次样条保证：
- 曲线经过所有控制点（路点）
- 在连接处一阶导（速度）和二阶导（加速度）连续
- 使用 Thomas 算法（追赶法）高效求解三对角方程组

### 10.2 PID 控制器

控制阶段使用两个 PID：

| PID | 输入 | 输出 | 含义 |
|-----|------|------|------|
| 横向 PID | 横向误差（到轨迹的符号距离） | steering（-1..1） | 车偏左就往右打方向盘 |
| 纵向 PID | 速度误差（目标速度 - 当前速度） | throttle/brake（0..1） | 慢了加油，快了刹车 |

PID 三个参数的含义：
- **P（比例）**：误差越大，纠正越猛——迅速缩小误差
- **I（积分）**：长期有误差就累加纠正——消除稳态误差
- **D（微分）**：误差变化越快越要收着——防止过冲和震荡

### 10.3 体素网格降采样（Voxel Grid Downsampling）

LiDAR 有 10 万个点，全量处理太慢。体素降采样把空间划分为 10cm × 10cm × 10cm 的小立方体，每个格子只保留一个代表点，大幅减少点数同时保留空间结构。

---

## 11. 流水线两种执行模式对比

本代码实现了两种模式，核心区别在于**是否允许不同帧在不同阶段同时处理**：

```
顺序模式（Sequential）：
Frame0: [感知][规划][控制] → Frame1: [感知][规划][控制] → ...
延迟 = 感知+规划+控制
吞吐 = 1/(感知+规划+控制)

流水线模式（Pipelined）：
Frame0: [感知]
Frame0: [规划] Frame1: [感知]
Frame0: [控制] Frame1: [规划] Frame2: [感知]
延迟 = 感知+规划+控制（不变）
吞吐 = 1/max(感知, 规划, 控制)（大幅提升）
```

关键：流水线不减少单帧延迟，但能成倍提升吞吐量——用空间（多帧同时在处理中）换时间。

---

## 12. 流水线执行会乱序吗？

**不会。** 原因有三：

1. **`std::queue` 是 FIFO**：先入先出。main 线程按帧号 0→1→2 顺序 push，每个 worker 按 push 顺序 pop，不乱序。
2. **每个阶段只有一个消费者线程**：perception/planning/control 各一人，不存在"两个线程抢同一帧"的情况。
3. **单帧内阶段顺序由代码保证**：perception → planning → control 是硬编码的单向数据流。

**但不同帧会同时处于不同阶段**——这不是乱序，正是流水线并行的目的。比如 Frame2 在感知、Frame1 在规划、Frame0 在控制同时运行。每帧内部仍然是 1→2→3 的严格顺序。

---

## 13. 伪共享（False Sharing）分析

**本代码基本没有伪共享问题**，原因：

| 组件 | 分析 |
|------|------|
| **三个队列** | `std::queue` 节点在堆上按需分配，不在同一 cache line |
| **mutex / cv** | `std::mutex` ≈40B、`std::condition_variable` ≈48B，各自独立，远超 64B cache line 对齐边界 |
| **stats 统计** | 统一走 `LatencyStats::record()`，不在工作线程维护本地计数器 |
| **stop_** | `std::atomic<bool>` 只写一次（`stop_ = true`），三个 worker 只读，不构成频繁写入冲突 |

理论上 `in_mutex_` 和 `pq_mutex_` 若类布局恰好共线，planning_worker 加锁 `pq_mutex_` 时可能令 perception_worker 持有的 `in_mutex_` 所在 cache line 失效——但 Linux pthread 实现已对 mutex 做 64 字节对齐，实际不会发生。

---

## 14. 为什么工业落地常用自实现队列而非 std::queue？

| 痛点 | `std::queue + mutex + cv` | 工业方案（Ring Buffer） |
|------|---------------------------|------------------------|
| **锁粒度** | 一个 mutex 锁整个队列，producer 和 consumer 互斥 | 无锁 CAS 原子操作，读写线程无竞争 |
| **内存分配** | 每次 push `new` 一个节点，可能触发 malloc 内部锁甚至 syscall | 预分配连续数组，push/pop 只移头尾指针，零分配 |
| **背压控制** | 默认无限增长，内存可能爆炸 | 带容量上限，队列满时阻塞生产者或丢帧 |
| **缓存友好** | 链表节点散落堆各处，cache miss 多 | 连续内存，预取友好 |

**典型工业选择**：

| 场景 | 方案 |
|------|------|
| 简单够用 | `boost::lockfree::spsc_queue` |
| 性能极致 | 自实现 ring buffer + `std::atomic` |
| 复杂拓扑 | `moodycamel::ConcurrentQueue` |
| ROS/自动驾驶 | ROS2 `rclcpp` 自带无锁消息队列 |

学习阶段用 `std::queue + mutex + cv` 完全合理——先跑通，再 profile，发现瓶颈才优化。

---

## 15. 如何进行 Profile？

### 15.1 先看自己打的点

```bash
./pipeline_sim --frames 100 --verbose
cat pipeline_metrics.json   # mean / p50 / p99 / std / max 一目了然
```

先看哪个阶段耗时占比最大，再深入。

### 15.2 CPU 热点采样（非侵入，不需重编译）

```bash
perf record -g ./pipeline_sim --frames 200
perf report          # 交互查看哪个函数占比最高

# 只看本项目函数（过滤 libc / libstdc++）
perf report --sort dso,symbol | head -30

# 生成火焰图
perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg
```

### 15.3 锁竞争检查

```bash
perf lock record ./pipeline_sim --mode pipelined --frames 500
perf lock report     # 看哪个锁的 wait time 最长
```

### 15.4 调度延迟检查

```bash
perf sched record ./pipeline_sim --frames 200
perf sched latency   # 看线程被调度出去的频率和时长
```

### 15.5 GPU Profiling（后续章节涉及）

```bash
nsys profile ./your_cuda_app    # Nsight Systems 时间线
ncu --set full ./your_cuda_app  # Nsight Compute kernel 细节
```

**核心原则**：先看数据（`pipeline_metrics.json`）定位哪个阶段慢，再用 `perf record` 定位该阶段内部哪行代码慢，最后针对性优化。别上来就乱改。

---

## 16. 发现问题后如何解决？

### 16.1 某个阶段耗时占比过大 → 根本瓶颈

**定位**：`pipeline_metrics.json` 中 `mean_ns` 最大的阶段。

**解决方案**：
- **算法降复杂度**：卡尔曼矩阵求逆用 2×2 闭式解析解，避免通用求逆
- **状态复用暖启动**：每帧从头 `new KalmanTrack` → 改为复用上一帧状态
- **降低处理量**：LiDAR 体素从 10cm → 20cm，点数砍 4 倍；相机 resize 可进一步降到 320×240
- **拆分阶段并行**：最慢阶段拆成子阶段，各自独立线程

### 16.2 P99 远大于 P50 → 尾延迟卡顿

**定位**：`p99_ns` 是 `mean_ns` 的 3 倍以上。

**解决方案**：
- **消除动态分配**：每帧的 `std::vector` 临时分配 → 类构造中预分配 buffer，运行时复用
- **绑核 + 实时调度**：
  ```cpp
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(1, &cpuset);
  pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

  sched_param param{ .sched_priority = 80 };
  pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
  ```
- **关掉 CPU 变频**：
  ```bash
  echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
  ```

### 16.3 锁竞争严重 → 排队开销

**定位**：`perf lock report` 显示某个 mutex wait time 很长。

**解决方案**：
- **缩短临界区**：锁只保护队列操作，计算和 stats 统计移到锁外
- **改用无锁队列（Ring Buffer）**：

  ```cpp
  template <typename T, size_t Size>
  class RingBuffer {
      std::array<T, Size> buffer_;
      std::atomic<size_t> head_{0};  // consumer 读指针
      std::atomic<size_t> tail_{0};  // producer 写指针
  public:
      bool push(const T &item) {
          size_t t = tail_.load(std::memory_order_relaxed);
          size_t next = (t + 1) % Size;
          if (next == head_.load(std::memory_order_acquire))
              return false;  // 满了，背压
          buffer_[t] = item;
          tail_.store(next, std::memory_order_release);
          return true;
      }
      bool pop(T &item) {
          size_t h = head_.load(std::memory_order_relaxed);
          if (h == tail_.load(std::memory_order_acquire))
              return false;
          item = buffer_[h];
          head_.store((h + 1) % Size, std::memory_order_release);
          return true;
      }
  };
  ```
  生产者只动 `tail_`，消费者只动 `head_`，互不干扰，无 mutex。

- **stats 收集避开热路径**：`LatencyStats::record()` 有内部 mutex → 改用 thread-local 缓冲，每 100 帧批量提交一次

### 16.4 吞吐量不够 → 阶段耗时不均

**定位**：`pipeline_metrics.json` 的 `throughput_fps` 低于目标。

**解决方案**：
- **加深流水线深度**：`--depth 5`，增加飞行中的帧数
- **拆分最慢阶段**：感知拆为 `perception_img + perception_lidar` 两个子阶段并行
- **多 CUDA Stream 并行**（后续 GPU 章节）：不同传感器处理用不同 stream 提交
- **丢帧策略**：实时系统允许超截止时间的帧直接丢弃，跳过去处理新帧

### 16.5 标准差不稳 → 抖动（Jitter）

**定位**：`pipeline_metrics.json` 的 `std_ns` 接近甚至超过 `mean_ns`。

| 抖动源 | 解法 |
|--------|------|
| CPU 调度抢占 | `SCHED_FIFO` + 绑核 |
| 中断抢占 | IRQ affinity 隔离：`echo 0 > /proc/irq/../smp_affinity` |
| CPU 变频 | governor 设为 `performance` |
| 内存分配 | 预分配 + 禁止运行时 `malloc` |
| `std::thread` 创建开销 | 用线程池，线程常驻不销毁 |

---

## 17. 性能诊断总路线

```
pipeline_metrics.json  →  定位 哪个阶段慢（mean）、稳不稳（std）、尾延迟多大（p99）
        ↓
perf record/report     →  定位 该阶段内部 哪行代码最耗时
        ↓
perf lock report       →  定位 是不是锁拖累的
        ↓
perf sched latency     →  定位 是不是被调度出去了
        ↓
针对性解决             →  算法降复杂度 / 去锁 / 预分配 / 绑核 / 变频
        ↓
再次运行验证           →  对比优化前后 p50/p99/throughput/std 是否改善
```

---

## 18. 如果改成真实传感数据，需要改动哪些部分？

### 18.1 需改动的 5 处（按数据流方向）

| 位置 | 当前做法 | 真实方案 |
|------|----------|----------|
| 传感器数据生成 | `generate_camera_image()` 随机像素 | 摄像头驱动或 ROS topic (`cv::VideoCapture`) |
| 传感器数据生成 | `generate_lidar_point_cloud()` 随机点云 | LiDAR 驱动或 bag 文件 (`sensor_msgs::PointCloud2`) |
| 检测生成 | `generate_detections()` 随机框 | 2D 检测(YOLO) + 3D 检测(PointPillars) + 融合 |
| 卡尔曼更新 | `kalman_update(track, x+0.01, y+0.01)` 假测量 | 匈牙利匹配检测框与 track，匹配成功才更新 |
| 轨迹规划控制点 | `wp.y = 3.0 * sin(i*0.3)` 硬编码正弦 | A*/Lattice Planner 在可行驶区域内生成 |
| 自车状态 | `ego_x=2.0, ego_speed=11.0` 写死 | 里程计/IMU/GPS 获取真实位姿与速度 |

### 18.2 不需要改动的部分

| 模块 | 原因 |
|------|------|
| 图像预处理（灰度化/缩放/归一化） | 通用操作，真实图像一样用 |
| LiDAR 预处理（距离过滤/体素降采样） | 真实点云也需要 |
| 卡尔曼滤波核心（predict/update 数学） | 算法本身正确 |
| PID 控制器 | 算法骨架对，输入换真值即可 |
| 三次样条生成 | 数学实现正确，控制点来源换真实规划即可 |
| 流水线执行器 + LatencyStats | 与数据内容无关，完全可复用 |

---

## 19. 多传感器融合：图像识别结果如何与 LiDAR 匹配？

### 19.1 核心前提：外参标定

多传感器融合的基础是**外参标定矩阵** `T_lidar_to_camera`（4×4）。标定后可以做：

```
LiDAR 点 (x, y, z) → 乘 T → 相机坐标系 (x', y', z') → 相机内参投影 → 图像像素 (u, v)
```

有了这个投影关系，一个 3D 空间中的点就能对应到图像上的一个像素。

### 19.2 三种融合方案

#### 方案一：前融合（Early Fusion）

直接把 LiDAR 点投影到图像上，把 RGB 颜色值"涂"在点上：

```
LiDAR 点 → 投影 → 取图像像素 RGB → 得到彩色点云 (x, y, z, r, g, b)
对彩色点云直接做 3D 检测
```

- **优点**：不丢信息，原始数据全保留
- **缺点**：计算量大，标定误差敏感（投影偏一个像素颜色就涂错了）

#### 方案二：后融合（Late Fusion）

图像跑 2D 检测器（YOLO），LiDAR 跑 3D 检测器（PointPillars），各自出结果后**匹配**：

```
3D 框 8 个角点 → 投影 → 图像上 8 个 (u, v) → 外接矩形 → "投影 2D 框"
投影 2D 框 vs YOLO 2D 框 → 算 IoU
IoU > 阈值 → 匹配成功，认定为同一物体
```

- **优点**：两个检测器独立最优，实现解耦
- **缺点**：可能漏掉一方没检出的物体（图像检出但 LiDAR 漏了）

#### 方案三：中融合（Middle Fusion / BEVFusion 类）——当前主流

把图像特征和 LiDAR 特征都投影到**共享鸟瞰图（BEV）空间**，在特征层面融合：

```
图像 → 2D CNN 提取特征 → Lift-Splat-Shoot / Cross-Attention → BEV 空间
LiDAR → 3D CNN 提取特征 → 直接落 BEV 空间
两者在 BEV 空间相加/拼接 → 统一做 3D 检测/分割
```

- **优点**：不需要显式做检测框级别的匹配，特征在 BEV 空间自然对齐
- **代表方法**：BEVFusion、BEVFormer、UniAD
- **缺点**：BEV 特征计算量大，常用 GPU 加速

### 19.3 直观对比

| 方案 | 比喻 |
|------|------|
| 前融合 | 把照片贴在 3D 模型上，得到一个"有颜色的雕塑"，然后分析它 |
| 后融合 | 图像说"我看到一辆车"，LiDAR 说"我看到一个立方体在某位置"，核对位置后认定同一物体 |
| 中融合 | 两人把看到的东西画在一张共享地图上，地图上自然标出共有物体 |

### 19.4 关键挑战：标定维护

无论哪种方案都依赖准确的外参。标定一旦偏移，投影就偏，匹配就乱。这是自动驾驶感知里最容易出 bug 的环节——车辆过减速带、传感器支架微变形，标定就可能失效。工业上通常需要**在线标定**持续校正。
