# ROS2 实时控制与通信模式

模拟 ros2_control 框架中机器人系统性能工程师必须掌握的中间件/基础设施模式。
不依赖实际 ROS2 安装，通过 C++17 标准库和 pthread 纯模拟核心模式。

## 功能

- **无锁 SPSC 环形缓冲区**：原子操作 + 缓存行填充，适用于传感器图像数据传输
- **实时执行器架构**：SCHED_FIFO 调度 + clock_nanosleep(TIMER_ABSTIME) 精确定时
- **生命周期状态机**：UNCONFIGURED → INACTIVE → ACTIVE → FINALIZED 含错误处理
- **QoS 策略模拟**：Reliable / BestEffort 通道 + Deadline 监控 + 速率匹配器
- **多速率管线**：200Hz 传感器 → 30Hz 感知 → 1kHz 控制，latest-is-best + hold-last

## 文件结构

```
17_ros2_realtime/
|-- spsc_ringbuffer.h     # 无锁 SPSC 环形缓冲区 (含 ImageRingBuffer)
|-- rt_executor.h         # 实时执行器 + NonRTThread 声明
|-- rt_executor.cpp       # RTExecutor + NonRTThread 实现
|-- lifecycle.h           # LifecycleNode + LifecycleManager 声明
|-- lifecycle.cpp         # 生命周期状态机实现
|-- qos_demo.h            # QoS 通道 + Deadline 监控 + 速率匹配器
|-- qos_demo.cpp          # QoS 演示辅助函数
|-- main.cpp              # 4 个演示: 环形缓冲区/执行器/生命周期/QoS 管线
|-- CMakeLists.txt
|-- README.md
```

## 推荐阅读顺序

1. **`spsc_ringbuffer.h`** — 基础通信原语：无锁 SPSC 环形缓冲区（原子操作 + 缓存行填充）
2. **`rt_executor.h` + `rt_executor.cpp`** — 实时执行器：SCHED_FIFO 调度 + clock_nanosleep 精确定时
3. **`lifecycle.h` + `lifecycle.cpp`** — ROS2 生命周期模拟（状态机 + 依赖顺序管理）
4. **`qos_demo.h` + `qos_demo.cpp`** — QoS 模拟（BestEffort / Reliable / Deadline 监控）
5. **`main.cpp`** — 最后阅读，演示 1(环形缓冲区)→2(执行器抖动)→3(生命周期)→4(QoS 管线)

## 构建

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## 运行

```bash
# 运行所有演示 (~15 秒)
./ros2_rt_demo

# 使用 sudo 以获得 SCHED_FIFO 实时优先级
sudo ./ros2_rt_demo
```

## 演示说明

### Demo 1: SPSC 环形缓冲区压力测试

1 百万条目的 push/pop 吞吐量测试。验证：
- 原子操作的 acquire/release 内存序正确性
- 缓存行填充防止伪共享
- push_overwrite() 的 latest-is-best 语义

### Demo 2: 实时执行器抖动测量

注册 3 个固定频率回调 (1kHz / 200Hz / 30Hz)，运行 5 秒，输出：
- 每个回调的抖动直方图 (最小/最大/平均/P99)
- 超时次数统计

使用 `clock_nanosleep(TIMER_ABSTIME)` 避免累积漂移。

### Demo 3: 生命周期管理

完整的 ros2_control 生命周期演示：
- 正常状态转换 (UNCONFIGURED → ACTIVE → FINALIZED)
- 错误注入：激活失败时安全回退到 INACTIVE
- 非法状态转换拒绝
- 依赖顺序管理 (硬件接口 → 控制器 → 规划器)

### Demo 4: QoS 多速率管线

5 秒多线程管线模拟：
- 200Hz 传感器线程 (BEST_EFFORT + latest-is-best)
- 30Hz 感知线程 (RELIABLE + 截止时间监控)
- 1kHz 控制线程 (RELIABLE + hold-last)

输出各阶段样本数和平均延迟。

## 指标输出

运行后生成 `ros2_realtime_metrics.json`，包含所有演示的量化指标。

## 关键技术点

| 模式 | 实现 | 对应 ROS2 概念 |
|------|------|---------------|
| SPSC 环形缓冲区 | `std::atomic` + acquire/release | rmw 层数据传输 |
| 实时执行器 | `SCHED_FIFO` + `clock_nanosleep` | ros2_control ControllerManager |
| 生命周期 | 状态机 + 原子状态变量 | rclcpp_lifecycle::LifecycleNode |
| 可靠传输 | 自旋等待 + 环形缓冲区 | RELIABLE QoS |
| 尽力传输 | push_overwrite() | BEST_EFFORT QoS |
| 截止时间 | 周期检查 + 回调 | Deadline QoS |
| 速率匹配 | latest-is-best / hold-last | 多速率节点编排 |

## 注意事项

- 实时调度 (`SCHED_FIFO`) 需要 root 权限或 `CAP_SYS_NICE` capability
- 如果没有实时权限，程序会降级使用 `SCHED_OTHER`，抖动会更大
- 实际部署时还需要：内存锁定 (`mlockall`)、CPU 隔离 (`isolcpus`)、中断亲和性配置
