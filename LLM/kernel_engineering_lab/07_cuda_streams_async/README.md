# 07_cuda_streams_async - CUDA Streams 与异步操作

## 工业背景：Streams 在推理服务中的应用

CUDA streams 是 GPU 上重叠计算和数据传输的主要机制。在生产推理服务中，streams 支持：

1. **Continuous batching**（vLLM、Triton Inference Server）：将下一 batch 的 H2D 拷贝与当前 batch 的计算重叠
2. **Pipeline parallelism**：通过模型阶段流式传输不同的微 batch
3. **Model replicas**：并发运行多个独立的模型实例
4. **Speculative decoding**：在独立的 stream 上运行 draft model 和 target model

缺少 streams 时，GPU 在 host-to-device 和 device-to-host 传输期间处于空闲状态，在推理工作负载中浪费 10-30% 的容量。

### Pinned Memory（Page-Locked Memory）

**为什么 pinned memory 如此重要：**

普通（pageable）host memory 随时可能被操作系统换出到磁盘。GPU DMA 引擎无法直接访问 pageable memory，因为物理页可能会变化。CUDA 驱动必须：
1. 分配一个临时的 pinned staging buffer
2. 从 pageable → pinned buffer 拷贝（同步）
3. GPU DMA 从 pinned buffer 读取

使用 pinned memory（`pin_memory=True` 或 `cudaHostAlloc`），GPU DMA 引擎可以直接读写，从而实现真正的异步传输。

**权衡：**
- Pinned memory 不能被换出 → 减少可用的系统 RAM
- 分配速度较慢（必须锁定页面）
- 在支持 GPUDirect 的现代 GPU 上，NUMA 最优位置的 pinned memory 很重要

### 默认 Stream 行为

CUDA 有两种默认 stream 模式：

| 模式 | 行为 | 设置 |
|------|----------|-------|
| Legacy Default Stream | Stream 0 与所有 stream 同步 | 旧版 CUDA 的默认设置 |
| Per-Thread Default Stream | 每个 host 线程拥有自己的默认 stream | `--default-stream per-thread` |

PyTorch 默认使用 legacy 行为，即默认 stream 与所有显式 stream 同步。这意味着在默认 stream 上启动会为所有其他 stream 创建隐式 barrier。

### Stream 优先级

CUDA 允许设置 stream 优先级（较低数字 = 较高优先级）：
```python
high_pri = torch.cuda.Stream(priority=-1)
low_pri = torch.cuda.Stream(priority=0)
```

**警告：** Stream 优先级经常被误用。较高优先级的 stream 可能耗尽低优先级的 stream。在大多数 GPU 上，仅支持 2-8 个有效优先级级别（取决于硬件）。过度使用可能导致优先级反转。

### 常见陷阱

1. **默认 stream 同步**：在默认 stream 上启动 kernel 会创建一个隐式的 `cudaDeviceSynchronize()`，阻塞所有其他 stream。始终使用显式 stream 进行并发工作。

2. **忘记 cudaEventRecord**：必须在同一 stream 上 kernel 启动之前和之后记录 events 才能获得准确的计时。

3. **Pinned memory 分配开销**：`torch.zeros(..., pin_memory=True)` 调用 `cudaHostAlloc`，涉及操作系统页表操作。对于频繁分配，应预先分配 pinned buffer 池。

4. **在错误的内存上使用 non-blocking transfers**：在 pageable host memory 上使用 `copy_(non_blocking=True)` 仍然会阻塞，因为驱动必须先执行到 pinned staging buffer 的内部拷贝。

5. **Stream 优先级误用**：为许多 stream 设置高优先级并不会让它们都快。大多数 GPU 只有 2 个硬件优先级级别。

### CUDA Events 用于计时

CUDA events 是测量 GPU kernel 执行时间的**唯一**准确方式。墙上时钟测量包括：
- Host 端驱动开销（每次 launch 约 5-50 us）
- Python 函数调用开销
- CPU 调度抖动
- `cudaDeviceSynchronize()` 延迟

CUDA events 直接记录在 GPU 时间线上，给出纯 GPU 时间。

## 模块结构

- `stream_basics.py`：单/多 stream 操作、同步模式
- `async_copy.py`：H2D/D2H 异步传输、pinned vs pageable、double-buffering
- `event_timing.py`：基于 CUDA event 的 kernel 计时、重叠检测
- `test_cuda_streams.py`：正确性测试（Python + CUDA C++ kernel）
- `benchmark_cuda_streams.py`：性能基准测试（Python vs CUDA C++ 原生）
- `setup.py`：CUDA C++ 扩展的构建配置
- `csrc/stream_kernels.cu`：CUDA C++ 原生 stream kernel 实现
- `csrc/bindings_stream.cpp`：PyTorch C++ 扩展绑定

### CUDA C++ 原生 Kernel（csrc/）

`csrc/` 目录包含生产环境风格的 CUDA C++ stream 编程示例：

| Kernel / 函数 | 描述 |
|---|---|
| `vector_add_kernel` | 基本向量加法，每个线程处理一个元素 |
| `vector_mul_pow_kernel` | 向量乘法 + pow（4 次），模拟计算密集型工作 |
| `vector_fma_kernel` | 融合乘加，常见于 MLP 层 |
| `multi_stream_concurrent_exec` | 在 N 个独立 CUDA stream 上并发启动 kernel |
| `pinned_async_pipeline` | 带 pinned memory 和异步拷贝的完整 H2D→compute→D2H 管线 |
| `kernel_timing_with_events` | 使用 cudaEventRecord/cudaEventElapsedTime 进行精确 kernel 计时 |
| `stream_wait_event_demo` | 使用 cudaStreamWaitEvent 的跨 stream 同步 |
| `war_sync_correct_vs_wrong` | 对比 `cudaDeviceSynchronize()` vs `cudaStreamSynchronize()` |

构建：
```bash
cd 07_cuda_streams_async && python setup.py build_ext --inplace
```

### Python vs CUDA C++ 原生 Stream 性能

| 方面 | Python torch.cuda.Stream() | CUDA C++ Native |
|---|---|---|
| Stream 管理 | Python 对象开销 | 直接 cudaStream_t |
| Event 计时 | torch.cuda.Event 包装 | 直接 cudaEvent_t |
| 异步 memcpy | copy_(non_blocking=True) | cudaMemcpyAsync |
| Launch 开销 | ~10-50 us（Python+GIL） | ~3-10 us（仅 host 驱动） |
| 控制力 | 适合原型开发 | 生产推理服务器 |

## 参考文献

- NVIDIA CUDA Programming Guide：Asynchronous Concurrent Execution
- CUDA C++ Best Practices Guide：Asynchronous Transfers and Overlapping
- vLLM：Efficient Memory Management for Large Language Model Serving
- Triton Inference Server：Model Concurrency and Dynamic Batching
