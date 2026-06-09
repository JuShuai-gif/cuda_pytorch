# 机器人系统性能工程师 — 完整学习项目

> 通过 17 个子项目系统掌握机器人/自动驾驶/机械臂/人形机器人/边缘 AI 领域系统性能优化所需全部技能。
>
> 每个子项目包含中文学习笔记（`notes/`）和可直接编译运行的真实场景代码（`src/`）。
>
> 对标岗位：机器人系统性能工程师 / 自动驾驶性能架构师 / 边缘 AI 推理优化 / 机械臂与人形机器人控制工程师（P6-P7）

---

## 项目总览

| # | 子项目 | 技能点 | 语言 | 真实场景 |
|---|--------|--------|------|----------|
| 01 | [系统端到端流水线](src/01_system_pipeline/) | 感知→规划→控制管线、P50/P99、吞吐量与抖动 | C++ | 1920×1080 相机+LiDAR+Voxel、Kalman 跟踪、PID 控制 |
| 02 | [CPU/GPU 混合计算](src/02_cpu_gpu_optimization/) | Stream 重叠、Pinned Memory、零拷贝、吞吐对比 | CUDA/C++ | Conv2D+ReLU+MaxPool 推理、体素降采样、NMS |
| 03 | [系统级瓶颈识别](src/03_system_bottleneck/) | 伪共享、锁竞争、缓存颠簸、内存拷贝开销 | C++ | Kalman 预测竞争、NMS 加锁对比、环形缓冲区 |
| 04 | [高性能 Runtime 设计](src/04_runtime_pipeline/) | DAG 任务图、双缓冲、流水线并行、异步执行 | C++ | 7 节点机器人任务图、Sobel 检测、A* 规划 |
| 05 | [Profiling 与监控](src/05_profiling_monitor/) | 延迟分解、直方图、P95/P99、瓶颈识别 | Python | 图像前处理→CNN→LiDAR 聚类→NMS→JSON 报告 |
| 06 | [跨团队协作优化](src/06_team_collaboration/) | 性能契约 SLA/SLO、跨模块合规验证、违规检测 | C++ | Sobel 感知+A* 规划+PID 控制合约验证 |
| 07 | [C++ 多线程与并发](src/07_cpp_multithread/) | 线程池、无锁队列 MPMC、原子操作、优先级反转 | C++ | 工作窃取线程池+无锁队列压测 |
| 08 | [计算机体系结构](src/08_computer_architecture/) | 缓存行、NUMA、MESI、SIMD、行列遍历 | C++ | 缓存行检测、NUMA 跨节点访问、AVX2 FMA |
| 09 | [GPU 计算与优化](src/09_gpu_optimization/) | 分块矩阵乘、Memory Coalescing、核融合、Stream 流水线 | CUDA | 分块 Matmul、Bias+ReLU 融合、Nsight 标注 |
| 10 | [Profiling 工具实战](src/10_profiling_tools/) | perf stat/record、火焰图、eBPF 追踪 | Shell/Python | perf 封装脚本、FlameGraph 生成、eBPF 延迟直方图 |
| 11 | [复杂流水线优化](src/11_complex_pipeline/) | 自动驾驶 7 阶段管线、A*+Kalman+Stanley 控制 | C++ | 传感器→预处理→检测→跟踪→预测→规划→控制 |
| 12 | [实时系统基础](src/12_realtime_system/) | RMS/EDF 调度、优先级反转/继承、截止时间分析 | C++ | 可调度性分析、超周期模拟、deadline miss 检测 |
| 13 | [边缘端 C++ 优化](src/13_edge_optimization/) | uncached 内存、DMA_BUF_IOCTL_SYNC、NEON、Fail-Closed | C++ | DMA 缓存一致性、FP16→FP32 NEON、分配/复用对比 |
| 14 | [NPU 推理优化](src/14_npu_inference/) | IO 持久化、双 Letterbox、NEON LUT、管线瓶颈拆解 | C++ | 1080p 5 阶段管线、27ms→0.1ms、13fps→29fps |
| 15 | [机械臂实时控制](src/15_manipulator_control/) | 正/逆运动学、DLS/NR IK、梯形/S 曲线轨迹、1kHz 关节控制 | C++ | 7 轴 DH 建模、雅可比伪逆、PID+抗饱和、控制闭环 |
| 16 | [人形机器人全身控制](src/16_humanoid_balance/) | WBC 零空间投影、ZMP/LIPM 平衡、34-DOF 任务栈 | C++ | 层级优先级求解、支撑多边形、步态摩擦锥 |
| 17 | [ROS2 实时通信](src/17_ros2_realtime/) | 无锁 SPSC 环形缓冲、RT 执行器、生命周期 FSM、QoS 策略 | C++ | SCHED_FIFO、多速率匹配、deadline 监控 |

---

## 快速开始

```bash
# 1. 阅读学习笔记（中文）
ls notes/

# 2. C++ 项目编译运行（以 01 为例）
cd src/01_system_pipeline
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./pipeline_sim --mode pipelined --frames 100

# 3. CUDA 项目编译运行（以 09 为例，需要 CUDA Toolkit）
cd src/09_gpu_optimization
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./gpu_optimization_demo

# 4. Python 项目运行
python3 src/05_profiling_monitor/main.py
```

---

## 技能矩阵

### 可直接写入简历的能力线

| 能力方向 | 掌握程度 | 对应项目 |
|----------|----------|----------|
| 端到端管线延迟分析与优化（P50/P99/Jitter） | 实战 | 01, 05, 11 |
| CPU/GPU 混合计算 + CUDA Stream 重叠 | 实战 | 02, 09 |
| C++ 多线程高并发（线程池、无锁队列、原子操作） | 实战 | 07 |
| 系统瓶颈定位（伪共享、锁竞争、缓存命中率、NUMA） | 实战 | 03, 08 |
| 高性能 Runtime 设计（DAG 任务图、双缓冲、流水线并行） | 实战 | 04 |
| Profiling 工具链（perf、火焰图、eBPF、Nsight） | 实战 | 05, 10 |
| 跨团队性能 SLA/SLO 制定与合规验证 | 理解 | 06 |
| 实时调度（RMS/EDF）与优先级反转解决 | 理解 | 12 |
| 边缘端 uncached/DMA 缓存一致性优化 | 实战 | 13 |
| NEON/AVX2 SIMD 手动向量化 | 实战 | 08, 13 |
| NPU 推理管线调优（IO 持久化、配置匹配） | 实战 | 14 |
| GPU Kernel 优化（Tiling、Coalescing、Fusion） | 实战 | 09 |
| 机械臂运动学与 1kHz 实时关节控制 | 实战 | 15 |
| 人形机器人 WBC 全身控制与 ZMP 平衡 | 理解 | 16 |
| ROS2 无锁实时通信与多速率管线架构 | 实战 | 17 |

### 对标面试能力

| 面试题 | 你的回答来源 |
|--------|-------------|
| "如何优化一段端到端延迟从 100ms 到 20ms？" | 01 + 11 的 P50/P99 分解方法论 |
| "GPU 推理如何与 CPU 预处理重叠？" | 02 的 CUDA Stream 流水线 + Pinned Memory |
| "遇到过锁竞争导致性能下降吗，怎么解决的？" | 03 的 SpinLock/Mutex/LockFree 对比 + NMS 案例 |
| "多线程下怎么设计高效的生产者消费者？" | 07 的无锁 MPMC 队列 + 线程池 |
| "怎么排查缓存命中率低的问题？" | 08 的 Cache Line 检测 + 行列遍历 + False Sharing |
| "用过什么 profiling 工具？" | 05 + 10 的 perf/flamegraph/eBPF/Nsight 完整工具链 |
| "实时系统怎么保证 deadline？" | 12 的 RMS/EDF 调度 + 优先级继承 |
| "边缘设备上怎么优化内存访问？" | 13 的 uncached→cached + DMA_BUF_IOCTL_SYNC |
| "NPU 推理从 13fps 到 29fps 怎么做的？" | 14 的 IO 持久化 + 双 Letterbox 消除 |
| "机械臂逆运动学怎么在 1ms 内解算完成？" | 15 的 DLS 阻尼最小二乘 + 奇异点检测 |
| "人形机器人怎么保证 30+ 关节实时控制？" | 16 的 WBC 层级零空间投影 + ZMP 平衡 |
| "ROS2 实时通信怎么避免优先级反转？" | 17 的无锁 SPSC 环形缓冲 + RT 执行器 |

---

## 学习路径建议

```
阶段一（基础）: 08 → 07 → 03       # 体系结构 + 多线程 + 瓶颈识别
阶段二（工具）: 05 → 10             # Profiling 方法论 + 工具实战
阶段三（GPU）: 09 → 02              # GPU Kernel 优化 → CPU/GPU 协同
阶段四（系统）: 04 → 01 → 11        # Runtime 设计 → 小型管线 → 复杂管线
阶段五（实时）: 12 → 17 → 06        # 实时调度 → ROS2 通信 → 团队协作 SLA
阶段六（边缘）: 13 → 14             # 边缘端内存优化 → NPU 推理调优
阶段七（机械）: 15 → 16             # 机械臂 IK+控制 → 人形 WBC+平衡
```

---

## 涉及的内核知识

| 知识点 | 对应项目 | 要求程度 |
|--------|----------|----------|
| DMA_BUF_IOCTL_SYNC（dma-buf 子系统） | 13 | 理解 API 语义即可 |
| NUMA 内存策略（mbind/set_mempolicy） | 08 | 会调用 numactl / libnuma |
| perf_event 子系统 | 05, 10 | 会用 perf stat/record/report |
| eBPF / bpftrace | 10 | 会写 bpftrace 单行脚本 |
| 实时调度类（SCHED_FIFO/SCHED_DEADLINE） | 12 | 理解调度策略 |
| CPU 亲和性（sched_setaffinity） | 07, 08, 17 | 会设置线程绑核 |
| SCHED_FIFO / clock_nanosleep | 12, 17 | 会配置实时线程与精确定时 |

**不要求内核开发**：所有代码均在用户态运行，不需要编写内核模块或修改内核代码。

---

## 项目元信息

- 总文件数：~175
- 总代码行：~28,000
- 语言：C++ / CUDA / Python / Shell
- 构建系统：CMake
- 所有注释与用户界面：中文
- 数据处理：真实图像矩阵、点云降采样、DH 运动学、雅可比伪逆、Kalman 滤波、A* 寻路、PID 控制、WBC 零空间投影、ZMP 平衡
