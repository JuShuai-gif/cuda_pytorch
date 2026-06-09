# CPU/GPU 混合计算优化

## 1. CPU 与 GPU 工作负载特性对比

### 1.1 CPU 特性

- **延迟导向**：单线程性能强，适合低延迟、分支密集的计算
- **大缓存层次**：L1 32KB ~ 64KB, L2 256KB ~ 512KB, L3 8MB ~ 64MB（共享）
- **乱序执行 + 分支预测 + 推测执行**：擅长控制流复杂的代码
- **核心数少但强**：8 ~ 128 个物理核心
- **频率高**：3GHz ~ 5GHz（桌面级）
- **适用场景**：数据预处理、决策规划、控制算法、多传感器同步、状态估计

### 1.2 GPU 特性

- **吞吐导向**：数千个轻量级线程并发，适合数据并行计算
- **小缓存 / 大寄存器文件**：L1/SMEM 可配置，寄存器压力是主要瓶颈
- **SIMT (Single Instruction, Multiple Threads)**：32 线程一组（warp），所有线程执行相同指令
- **核心数多但弱**：数千 ~ 上万个 CUDA Core
- **频率较低**：1GHz ~ 2GHz
- **适用场景**：深度学习推理、图像处理、点云处理、矩阵运算、FFT

### 1.3 关键决策矩阵

| 操作类型 | 选 CPU | 选 GPU |
|---------|--------|--------|
| 小矩阵乘法 (< 128×128) | ✓ CPU（启动开销大） | ✗ |
| 大矩阵乘法 (> 1024×1024) | ✗ | ✓ GPU |
| 分支密集代码 | ✓ CPU（分支预测） | ✗（warp divergence） |
| 大量并行无分支计算 | ✗ | ✓ GPU |
| 低延迟要求 (< 100μs) | ✓ CPU | ✗（kernel launch ~5μs） |
| 批量处理 | ~ | ✓ GPU |
| 排序、哈希表 | ✓ CPU | ~（较新的 GPU 库） |

## 2. 数据传输开销

### 2.1 PCIe 带宽

- PCIe 3.0 x16：~16 GB/s（单向理论值）
- PCIe 4.0 x16：~32 GB/s（单向理论值）
- PCIe 5.0 x16：~64 GB/s（单向理论值）
- 实际有效带宽通常为理论值的 70% ~ 85%

### 2.2 Host ↔ Device 数据传输

**可分页内存（Pageable Memory）**：

```
host_pageable → staging buffer (pinned) → DMA → GPU
```

- 额外一次内存拷贝（从可分页内存到 pinned 暂存区）
- 带宽利用率低，因为需要先分配 DMA 暂存区

**页锁定内存（Pinned / Page-Locked Memory）**：

```
host_pinned → DMA → GPU  (直接 DMA)
```

- 操作系统保证该内存不会被换出到磁盘
- GPU 可以直接通过 DMA 访问
- 带宽接近 PCIe 理论峰值
- 缺点：分配大块 pinned memory 会大量消耗物理内存，可能导致系统 OOM

### 2.3 统一内存（Unified Memory / Managed Memory）

- `cudaMallocManaged()` 分配的内存在 CPU 和 GPU 间按需迁移（page fault-driven）
- CUDA 6+ 支持，Pascal+ (SM60) 支持硬件页故障
- 优点：编程简单，按需迁移，减少显式传输代码
- 缺点：
  - 页故障有开销（微秒级）
  - 并发访问需要显式同步
  - 迁移粒度是页（通常 64KB），可能浪费带宽
- 适用场景：访问模式不规则的数据结构（树、图）

## 3. CUDA 流（Streams）与异步操作

### 3.1 默认流与多流

- **默认流（Stream 0 / NULL Stream）**：所有操作串行执行
- **非默认流**：不同流中的操作可以并行（前提是无依赖）
- 流是 CUDA 中实现**并发**的核心机制

### 3.2 将计算与数据传输重叠

这是性能优化的核心技巧：

```cpp
// Naive approach (no overlap):
for (int i = 0; i < N; i++) {
    cudaMemcpy(d_in, h_in, size, H2D);    // Transfer
    kernel<<<grid, block>>>(d_in, d_out);  // Compute
    cudaMemcpy(h_out, d_out, size, D2H);   // Transfer back
}
// Total time = N * (transfer + compute)

// Stream-overlapped approach:
for (int i = 0; i < N; i+=2) {
    cudaMemcpyAsync(d_in[i], h_in[i], size, H2D, stream[0]);
    kernel<<<grid, block, 0, stream[0]>>>(d_in[i], d_out[i]);
    cudaMemcpyAsync(h_out[i], d_out[i], size, D2H, stream[0]);

    cudaMemcpyAsync(d_in[i+1], h_in[i+1], size, H2D, stream[1]);
    kernel<<<grid, block, 0, stream[1]>>>(d_in[i+1], d_out[i+1]);
    cudaMemcpyAsync(h_out[i+1], d_out[i+1], size, D2H, stream[1]);
}
// Total time ≈ (N/2) * max(transfer, compute) + transfer + compute
// ~2x improvement when transfer ≈ compute
```

### 3.3 流并发的硬件要求

- **必须使用 pinned memory**：pinned memory 是异步传输的前提
- **需要支持并发 copy 和 kernel 执行的 GPU**：compute capability >= 1.1 支持，但需要确保有独立的 copy engine
- **copy engine 数量**：多数 GPU 有 1~2 个 DMA copy engine，决定了可同时进行的数据传输数

## 4. CPU-GPU 任务调度策略

### 4.1 任务划分原则

1. **粗粒度流水线**：CPU 做预处理（resize、color convert、normalize）→ GPU 做推理
2. **细粒度混合**：推理结果回传 CPU 做后处理（NMS、解码、tracking）
3. **CPU 预取**：在处理当前帧的同时，CPU 提前加载下一帧数据到 pinned memory
4. **批量攒批（Batching）**：积攒多帧数据一起送 GPU，提升 GPU 利用率

### 4.2 负载均衡

- **工作窃取（Work Stealing）**：空闲的 CPU 核心从忙碌核心的任务队列中偷取任务
- **动态划分**：根据历史执行时间动态调整分配给 CPU/GPU 的数据比例
- **优先级队列**：安全关键任务（控制）优先级高于非关键任务（日志）

### 4.3 GPU 的异步回调与 CPU 同步

```cpp
// Polling-based: CPU spins waiting for GPU
while (cudaEventQuery(event) == cudaErrorNotReady) {
    // CPU can do other work here
    do_some_cpu_work();
}

// Callback-based: GPU notifies CPU upon completion
cudaLaunchHostFunc(stream, [](void* data) {
    // This runs on CPU after stream work is done
    reinterpret_cast<MyData*>(data)->cleanup();
});
```

## 5. 内存管理最佳实践

### 5.1 Pinned Memory 池

重复分配/释放 pinned memory 开销很大。建议预分配一个池：

```cpp
class PinnedMemoryPool {
    std::vector<void*> free_list;
public:
    void* alloc(size_t size);      // 从池中获取
    void free(void* ptr);           // 归还给池
};
```

### 5.2 零拷贝（Zero-Copy / Mapped Memory）

- `cudaHostAlloc()` 配合 `cudaHostAllocMapped` 标志
- GPU 可以直接访问主机内存（通过 PCIe），无需显式 copy
- **适用场景**：数据仅被 GPU 访问一次或少量访问
- **注意**：每次 GPU 访问都经过 PCIe，延迟高；适合访问次数少的数据

### 5.3 设备端内存池

- GPU 上的 `cudaMalloc()` / `cudaFree()` 开销较大
- 使用内存池（如 `cub::CachingDeviceAllocator`）减少分配开销
- CUDA 11.2+ 自带 `cudaMemPool_t` API

### 5.4 内存访问模式优化

- **合并访问（Coalesced Access）**：同一 warp 的线程访问连续的全局内存地址
- **避免 bank conflict**：共享内存的 32 个 bank 中不要有多个线程同时访问同一 bank 的不同地址

## 6. 性能调试工具链

| 工具 | 用途 |
|------|------|
| `nvidia-smi` | GPU 利用率、显存、温度、功耗监控 |
| `nsys` (Nsight Systems) | 时间线视图，CPU/GPU 活动关联分析 |
| `ncu` (Nsight Compute) | 单个 kernel 的详细性能分析 |
| `nvprof` | 旧版 profiler（逐步被 nsys/ncu 替代） |
| `perf` | CPU 端性能分析、cache miss、分支预测 |

## 7. 实战：混合计算流水线示例

以自动驾驶感知流水线为例：

```
CPU: 图像解码(JPEG->RGB) → Resize → Normalize → 传输到 GPU (pinned + async)
GPU: NN 推理 → 后处理 (NMS GPU kernel)
CPU: 将检测结果从 GPU 取回 → 匈牙利匹配 → 卡尔曼滤波更新
```

**优化要点**：
- 用 nvJPEG 或 GPU JPEG decoder 替代 CPU 解码（减少 PCIe 上行数据量）
- 双缓冲：交替使用两块 GPU 内存，避免拷贝与计算互相阻塞
- Pipeline 深度：double-buffering 或 triple-buffering（最多隐藏一帧延迟）
