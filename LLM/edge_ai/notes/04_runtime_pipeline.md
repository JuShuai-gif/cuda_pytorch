# 高性能运行时与流水线设计

## 1. 异步执行模型

### 1.1 Future/Promise 模型

**概念**：`Promise` 是生产者端写入结果的容器，`Future` 是消费者端读取结果的句柄。两者通过共享状态连接。

```cpp
std::promise<int> p;
std::future<int> f = p.get_future();
std::thread([&]() {
    int result = heavy_compute();
    p.set_value(result);  // Producer signals completion
}).detach();
int value = f.get();  // Consumer blocks until ready
```

**优点**：解耦生产者和消费者，支持链式组合（`.then()`），错误传播清晰。
**缺点**：每次 `.get()` 涉及同步开销；大量小任务时分配开销大；`std::future` 不支持 continuation（C++20 之前）。

**适用**：单个异步任务的结果传递，如一次 RPC 调用、一次文件读取。不适合高频流式数据。

### 1.2 协程（Coroutines）

C++20 引入无栈协程，通过 `co_await`/`co_return`/`co_yield` 实现：

```cpp
task<int> async_read() {
    int data = co_await socket.read_async();
    co_return data;
}
```

**优点**：代码像同步写法，无回调地狱；无栈协程内存开销极小（< 100 字节）；可组合。
**缺点**：C++20 标准库未提供常用协程类型（generator、task 需自行实现或使用 cppcoro）；调试困难（调用栈不连续）。

**适用**：I/O 密集型异步逻辑、状态机实现。在高吞吐实时系统中通常用于编排而非热路径。

### 1.3 回调（Callback）

```cpp
void on_frame_ready(Frame f, std::function<void(Result)> next) {
    detect(f, [next](Detection d) {
        track(d, [next](Track t) {
            next(plan(t));
        });
    });
}
```

**优点**：零抽象开销，天然支持事件驱动。
**缺点**：回调地狱（callback hell）；错误处理分散；生命周期管理复杂（野指针风险）。

**适用**：底层驱动、中断处理、异步信号通知。

### 1.4 事件循环（Event Loop）

单线程轮询事件队列，将就绪回调依次执行。Node.js/libuv 和 ROS2 executor 的典型模式。

```
while (running) {
    auto events = poll_for_events();  // epoll / kqueue
    for (auto &ev : events) {
        dispatch(ev);
    }
}
```

**优点**：无锁（单线程），编程模型简单，适合 I/O 密集型。
**缺点**：一个回调阻塞会拖延所有其他事件；无法利用多核。

### 1.5 选择决策表

| 场景 | 推荐模型 | 原因 |
|------|---------|------|
| 单次 RPC/文件操作 | Future/Promise | 简单直接 |
| 高并发 I/O (10k+ 连接) | 事件循环 | 无上下文切换开销 |
| 流水线数据流 (帧处理) | 回调 + 线程池 | 最小延迟 |
| 复杂异步编排 | 协程 | 可读性最佳 |
| GPU kernel 异步 | CUDA Stream + Callback | 原生支持 |

## 2. 流水线并行

### 2.1 原理

将处理流程拆分为多个**阶段（Stage）**，每个阶段由一个独立线程/进程处理。各阶段之间通过队列传递数据：

```
[Stage 0] → [Queue 0→1] → [Stage 1] → [Queue 1→2] → [Stage 2]
```

**关键公式**：
- 串行吞吐量：`T_serial = N / Σ(stage_i)`（N 帧的处理时间）
- 流水线稳态吞吐量：`T_pipeline ≈ N / max(stage_i)`（由最慢阶段决定）
- 流水线延迟 ≈ Σ(stage_i)（首帧延迟，后续帧叠加排队延迟）

**加速比上限** = `Σ(stage_i) / max(stage_i)`（Amdahl 定律的流水线版本）。

### 2.2 阶段划分原则

1. **均衡性原则**：尽量让各阶段处理时间接近（load balance），否则瓶颈阶段前队列堆积
2. **数据依赖原则**：阶段之间只能传递 forward dependency，不能有循环依赖
3. **I/O 边界原则**：在 I/O 等待处切分，让计算和 I/O 重叠
4. **硬件亲和原则**：GPU 密集阶段和 CPU 密集阶段分离，让 CPU 在 GPU 计算期间做预处理

### 2.3 ROS2 Executor 解析

ROS2 使用基于 DDS 的发布/订阅模型。Executor 负责调度 callback：
- **SingleThreadedExecutor**：单线程轮询所有 subscription 的就绪 callback
- **MultiThreadedExecutor**：线程池并发执行 callback
- **EventsExecutor**（ROS2 Humble+）：基于事件图，按依赖关系调度

实际工程中的坑：同一个 callback 被多个线程并发执行（需加锁），不同 priority 的 callback 饿死。

## 3. 双缓冲（Ping-Pong Buffer）

### 3.1 概念

维护两个缓冲区：一个由**生产者**写入（前端缓冲），另一个由**消费者**读取（后端缓冲）。完成后交换指针。

```
Frame 1: Producer → Buffer A, Consumer → Buffer B (idle)
         SWAP
Frame 2: Producer → Buffer B, Consumer → Buffer A (Frame 1 data)
         SWAP
Frame 3: Producer → Buffer A, Consumer → Buffer B (Frame 2 data)
```

### 3.2 实现要点

```cpp
struct DoubleBuffer {
    std::array<Frame, 2> buffers;
    std::atomic<int> front{0};  // Producer writes here
    // Swap: front = 1 - front (after consumer finishes reading)
};
```

**关键约束**：
- Swap 操作必须等待消费者读完（使用 fence/barrier 同步）
- Producer 不能覆盖消费者正在读的 buffer
- 通常使用 `std::atomic` + `memory_order_acquire/release` 实现无锁 swap

**适用**：传感器数据捕获（一帧写入同时上一帧被处理）、GPU 渲染（前缓冲显示/后缓冲绘制）、环形缓冲区变体。

## 4. 任务图（Task Graph / DAG）

### 4.1 概念

将计算表达为有向无环图（DAG）：节点 = 任务，边 = 数据依赖。调度器在依赖就绪后执行节点。

```
     ┌─→ [Preprocess] ─┐
[Input] ─→ [Detector] ─→ [Tracker] ─→ [Planner] ─→ [Output]
     └─→ [LidarProc] ─┘
```

### 4.2 调度策略

- **拓扑排序**：按入度归零顺序执行（静态调度）
- **就绪队列**：维护一个入度为 0 的节点集合，线程池从中取任务执行，完成后递减后继节点的入度
- **优先级调度**：给关键路径上的节点更高优先级
- **异构调度**：将 GPU 节点和 CPU 节点分配到不同的执行队列

### 4.3 TensorRT Inference Pipeline

NVIDIA TensorRT 内部将网络层构建为执行图：
- **Builder 阶段**：将 ONNX/TensorFlow 模型转换为优化后的引擎（engine）
- **Layer Fusion**：合并连续的卷积+偏置+激活为单一 kernel（垂直融合），合并并行分支（水平融合）
- **CUDA Graph Capture**：将完整的推理 launch 序列录制为 CUDA Graph，消除每次 kernel launch 开销

## 5. 背压处理（Backpressure）

### 5.1 问题

当下游阶段处理速度慢于上游时，队列无限增长导致内存耗尽。

### 5.2 策略

| 策略 | 做法 | 适用 |
|------|------|------|
| **阻塞队列** | `bounded_queue`，队列满时生产者阻塞 | 不能丢数据的场景 |
| **丢弃策略** | 队列满时丢弃最旧/最新数据 | 传感器数据（最新优先） |
| **降级策略** | 动态降低处理精度（如降低分辨率） | 负载突增时保延迟 |
| **反压传播** | 下游减速信号逐级向上传播 | 管道式流水线 |

### 5.3 GStreamer 的背压机制

GStreamer 使用 `flow_return` 机制：当 sink 端消费速度跟不上时返回 `GST_FLOW_FLUSHING`，逐级向上游 element 传播，上游暂停生产。

## 6. 实际案例对比

### 6.1 GStreamer

- 插件（element）通过 pad 连接，形成 pipeline
- 每个 element 在独立线程中运行（通过 `queue` element 解耦）
- 支持零拷贝（`GstBuffer` 带引用计数）
- 适用于视频/音频流处理

### 6.2 ROS2 Executor

- 节点（Node）订阅/发布主题（Topic）
- Executor 按 wait-set 等待就绪的 subscription
- 适用于机器人多传感器融合和控制

### 6.3 TensorRT Inference Pipeline

- 模型优化为 plan/engine 文件
- 支持多 stream 并发推理（每个 stream 独立 CUDA stream）
- CUDA Graph 减少 kernel launch 延迟
- 适用于深度学习推理加速

## 7. 模式选择总结

始终从数据流图出发：画出各处理阶段、数据量、延迟要求，然后匹配模式：
- **单生产者单消费者 + 均衡负载**：简单流水线 + 双缓冲
- **多入度/多出度 + 依赖复杂**：任务图（DAG）+ 就绪队列调度
- **突发流量 + 不可丢数据**：有界阻塞队列 + 背压传播
- **GPU 密集 + CPU 轻量**：CUDA Stream 并发 + CPU 线程 pool 重叠执行
