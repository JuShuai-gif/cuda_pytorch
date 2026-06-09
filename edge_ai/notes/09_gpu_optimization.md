# GPU 计算与优化实战

## 1. CUDA 编程模型

### 1.1 Grid / Block / Thread 层次结构

```
Grid (1D/2D/3D)
  ├── Block (0,0)
  │     ├── Thread (0,0), Thread (0,1), ..., Thread (31,0), ...
  │     └── 一组 threads → Warp(32 threads)
  ├── Block (0,1)
  │     └── 更多 threads
  └── Block (N-1, M-1)

Kernel 启动：my_kernel<<<gridDim, blockDim, sharedMem, stream>>>(args);
```

**关键约束**：

| 参数 | 限制（典型） | 说明 |
|------|------------|------|
| Threads/Block | 最大 1024 | 由硬件 SM 决定 |
| Blocks/Grid (1D) | 最大 2³¹-1 | 几乎无限 |
| Shared Memory/Block | 最大 48KB-164KB | 可配置 |
| Registers/Thread | 最大 255 | 寄存器溢出 → 本地内存 |
| Warp Size | 32 | SIMT 执行单元 |
| Active Warps/SM | 最大 64 | 影响占用率 |

### 1.2 线程索引计算

```cuda
// 1D
int tid = blockIdx.x * blockDim.x + threadIdx.x;

// 2D (grid 和 block 都是 2D)
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int index = y * width + x;

// 3D
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
int index = z * width * height + y * width + x;
```

### 1.3 SIMT 执行模型

**Warp** = 32 个线程，执行相同指令。

**Warp Divergence**：

```cuda
// 可能导致 warp 内部分线程空闲
if (threadIdx.x < 16) {
    // 只有前 16 个线程执行
    do_work_A();   // 后 16 个线程闲置
} else {
    do_work_B();   // 前 16 个线程闲置
}

// 好的写法：让分支边界对齐 warp 边界
if (threadIdx.x / 32 == 0) {
    // 整个 warp 执行同一个分支（无 divergence）
}
```

**分支处理策略**：

1. 条件基于 `threadIdx.x` → 如果边界对齐 warp，无性能损失
2. 条件基于数据 → 优化数据结构使 warp 内数据一致
3. 使用 `__any_sync`、`__all_sync` 等 warp 级投票原语

## 2. GPU 内存层次

### 2.1 内存类型对比

```
Thread ──► 寄存器          (~1 cycle, ~8KB total per SM)
  │
Block  ──► 共享内存         (~20-30 cycles, 48-164KB per SM)
  │
Grid   ──► L1 缓存          (~30 cycles, 与共享内存共享芯片面积)
  │        L2 缓存          (~200 cycles, 几MB)
  │        全局内存 (HBM2/3) (~400-800 cycles, GB级别)
  │        常量内存          (缓存优化)
  │        纹理内存          (2D 空间局部性优化)
  │
Host   ──► CPU RAM          (PCIe/NVLink 连接)
```

**带宽示例（A100）**：

| 内存类型 | 带宽 |
|----------|------|
| 寄存器 | ~8TB/s |
| 共享内存 | ~19TB/s |
| L2 Cache | ~4TB/s |
| HBM2e | ~2TB/s (80GB) |
| PCIe 4.0→GPU | ~32GB/s |

### 2.2 合并访问（Coalesced Access）

**合并条件**：同一个 warp 的 32 个线程访问同一个 128 字节对齐段内的内存。

```cuda
// Correct: 合并访问
// Thread 0 → data[0], Thread 1 → data[1], ..., Thread 31 → data[31]
float val = data[threadIdx.x];  // 一个 128B 事务

// Wrong: 跨步访问（stride > 1）
// Thread 0 → data[0], Thread 1 → data[32], Thread 2 → data[64]...
float val = data[threadIdx.x * 32];  // 32 个独立事务！
```

**实际影响**：

```cuda
// Bad (stride = N)
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j += blockDim.x) {
        C[i][j] = A[i][j] + B[i][j];  // 线程访问同一列的不同行
    }
}

// Good (stride = 1)
for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i += blockDim.x) {
        C[i][j] = A[i][j] + B[i][j];  // 相邻线程访问相邻元素
    }
}
```

### 2.3 共享内存和 Bank Conflict

共享内存分为 32 个 bank（每个 4 字节宽）。同一 warp 内的多个线程访问同一 bank 的不同地址 → bank conflict。

```cuda
__shared__ float smem[32][32];

// 无冲突：每个线程访问不同 bank
float val = smem[threadIdx.x][0];  // thread 0→bank0, thread1→bank1...

// 2-way conflict：thread0 和 thread16 都访问 bank0
float val = smem[threadIdx.x * 2];

// 广播（无冲突）：所有线程访问同一 bank 的相同地址
float val = smem[0][threadIdx.x];  // 同一地址 → 广播
```

**解决方案**：

```cuda
// 添加填充使每行错开 bank 对齐
__shared__ float smem[32][32 + 1];  // +1 解决 bank conflict
float val = smem[threadIdx.y][threadIdx.x];  // 无冲突！
```

### 2.4 常量内存（Constant Memory）

```cuda
__constant__ float coeffs[256];  // 最多 64KB

// 优点：当 warp 内所有线程读取相同地址时，等同于寄存器速度
// 缺点：warp 内读取不同地址 → 串行化
// 用例：滤波器系数、物理常数、配置参数
```

## 3. TensorRT 优化概念

### 3.1 层融合（Layer Fusion）

将多个运算合并为一个 kernel，避免多次读写全局内存。

```
融合前：
input → CONV → global mem → BIAS → global mem → RELU → output
                 ↑ 3 次全局内存遍历 ↑

融合后：
input → CONV_BIAS_RELU → output
            ↑ 1 次全局内存遍历 ↑
```

**TensorRT 支持的融合类型**：

| 融合类型 | 示例 |
|---------|------|
| Conv + Bias + ReLU | 最常见的融合 |
| Conv + BatchNorm + ReLU | 推理时 BatchNorm 可吸收到 Conv 权重中 |
| Conv + ElementWise | Conv 后接 Add/Sub/Mul |
| 矩阵乘法 + activation | MatMul + Gelu/SiLU |
| 多个 ElementWise | Add + ReLU、Mul + Add |
| 逐点算子串联 | Sigmoid → Mul → Add |

### 3.2 精度校准（Precision Calibration）

```
FP32 训练 → 校准数据集 → FP16/INT8 推理
```

**FP16 关键点**：

- A100 的 Tensor Core：FP16 吞吐是 FP32 的 8 倍
- 部分层仍需 FP32（如 loss scaling），混合精度解决
- `__half` 和 `__half2`（每 2 个 FP16 打包）类型

**INT8 校准流程**：

1. 收集运行时的激活值分布
2. 确定每层的动态范围（min/max）
3. 计算缩放因子 `scale = (max - min) / 255`
4. 偏置因非对称范围调整：`zero_point = -min / scale`
5. 推理时：`INT8 = round(FP32 / scale) + zero_point`

```python
# TensorRT INT8 calibration (pseudo)
def calibrate_layer(layer_activations):
    # KL divergence minimization
    fp32_hist = histogram(layer_activations, bins=2048)
    best_kl = inf
    best_threshold = 0
    for threshold in candidate_thresholds:
        int8_hist = quantize(fp32_hist, threshold)
        kl = KL_divergence(fp32_hist, int8_hist)
        if kl < best_kl:
            best_kl = kl
            best_threshold = threshold
    return best_threshold
```

### 3.3 Kernel 自动调优

TensorRT 为每个算子预编译多种实现，运行时选择最快的：

```
Conv 3x3 stride=1 pad=1 的实现：
├── implicit_gemm（小通道数）
├── winograd（大通道数，3x3 专用）
├── fft_conv（超大 kernel）
└── cudnn（自动调优的通用路径）
```

Autotuning 选择策略：
1. 第一次运行：对每个候选 kernel 采样少量执行，测量时间
2. 后续运行：直接使用已选最佳 kernel
3. 缓存结果到文件（`timing_cache`）

## 4. CPU-GPU 协调

### 4.1 CUDA Streams

Stream 是一个命令队列。同一 stream 内操作顺序执行，不同 stream 间可并行。

```cuda
cudaStream_t stream1, stream2, stream3;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
cudaStreamCreate(&stream3);

// 每个 stream 独立：预处理 + 传输 + 计算 + 传输 + 后处理
for (int i = 0; i < NUM_IMAGES; i++) {
    cpu_preprocess(host_img[i], i);           // CPU 准备数据
    cudaMemcpyAsync(dev_img[i], host_img[i],  // 异步 H→D
                    size, cudaMemcpyHostToDevice, stream1);
    kernel<<<grid, block, 0, stream1>>>(dev_img[i], dev_out[i]);
    cudaMemcpyAsync(host_out[i], dev_out[i],  // 异步 D→H
                    size, cudaMemcpyDeviceToHost, stream1);
}
cudaStreamSynchronize(stream1);  // 等待所有操作完成
```

**流水线重叠（Pipeline Overlap）**：

```
Stream0: [H→D][Kernel][D→H] [H→D][Kernel][D→H]
Stream1:        [H→D][Kernel][D→H] [H→D][Kernel][D→H]
Stream2:               [H→D][Kernel][D→H] ...
          ↑ 隐藏数据传输延迟 ↑
```

### 4.2 CUDA Events

Events 用于测量时间、同步部分操作。

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(...);
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
printf("Kernel time: %.3f ms\n", ms);
```

### 4.3 固定内存（Pinned Memory）

```cuda
float *host_pinned;
cudaMallocHost(&host_pinned, size);  // 或 cudaHostAlloc

// 固定内存允许 DMA 直接传输
// 带宽比 pageable memory 高 ~2-3x
// 缺点：不能交换出去，占用物理内存

cudaFreeHost(host_pinned);
```

**固定内存类型**：

| 标志 | 用途 |
|------|------|
| `cudaHostAllocDefault` | 标准固定内存 |
| `cudaHostAllocPortable` | 所有 GPU 可访问 |
| `cudaHostAllocWriteCombined` | 仅主机写入（GPU 读取更快） |
| `cudaHostAllocMapped` | 映射到 GPU 地址空间（零拷贝） |

## 5. GPU Profiling

### 5.1 Nsight Compute

```bash
# CLI profiling - 收集 kernel 的详细指标
ncu --set full -o profile_report ./my_cuda_program

# 查看特定 kernel
ncu --kernel-name matmul_kernel ./my_program

# 分析占用率、瓶颈
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --metrics l1tex__throughput.avg.pct_of_peak_sustained_elapsed \
    ./my_program

# 查看 profile 报告
ncu-ui profile_report.ncu-rep  # GUI 模式
```

**关键指标**：

| 指标 | 含义 | 目标 |
|------|------|------|
| `smsp__cycles_active` | SM 活跃周期 | 越高越好 |
| `dram__bytes.sum` | DRAM 读写总量 | 越低越好 |
| `smsp__warps_launched` | 启动的 warp 数 | — |
| `achieved_occupancy` | 实际占用率 | > 50% |
| `l1tex__t_sectors_lookup_hit` | L1 命中率 | > 80% |
| `lts__t_sectors_lookup_hit` | L2 命中率 | > 60% |

### 5.2 Nsight Systems

提供系统级视图（CPU + GPU 时间线），分析整体性能。

```bash
nsys profile --trace=cuda,nvtx,osrt \
             --stats=true \
             --output=timeline \
             ./my_program

nsys-ui timeline.nsys-rep  # GUI 模式
```

**分析重点**：

- CPU-GPU 同步点（`cudaDeviceSynchronize`）
- 空闲时间（GPU 空闲 = 没有 kernel）
- 数据传输与计算的 overlap 程度
- NVTX 标注的自定义区间

### 5.3 NVTX 注释

```cuda
#include <nvtx3/nvToolsExt.h>

void preprocess_images() {
    nvtxRangePushA("CPU:Preprocess");  // 在 Nsight 时间线上显示
    // ... CPU preprocessing ...
    nvtxRangePop();

    nvtxRangePushA("GPU:Inference");
    kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
    nvtxRangePop();
}

// C++ RAII wrapper
class NvtxRange {
public:
    explicit NvtxRange(const char *name) {
        nvtxRangePushA(name);
    }
    ~NvtxRange() { nvtxRangePop(); }
};
```

## 6. 常用优化技巧

### 6.1 Tiling（分块）

将大矩阵分解为小块，在共享内存中批量处理。

```cuda
// 矩阵乘法 C = A × B 的分块实现
#define TILE_SIZE 32
__global__ void matmul_tiled(float *A, float *B, float *C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    float sum = 0.0f;
    for (int tile = 0; tile < N / TILE_SIZE; tile++) {
        // 协作加载一个 tile
        As[ty][tx] = A[(by * TILE_SIZE + ty) * N + tile * TILE_SIZE + tx];
        Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + bx * TILE_SIZE + tx];
        __syncthreads();

        // 在共享内存中计算
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    C[(by * TILE_SIZE + ty) * N + bx * TILE_SIZE + tx] = sum;
}
```

**tiling 的效果**：

- 原始：每个元素从全局内存读取 2N 次
- tiling：每个元素从全局内存读取 N/TILE_SIZE 次
- TILE_SIZE=32, N=1024 → 减少 32 倍全局内存访问

### 6.2 寄存器压力管理

**症状**：寄存器溢出 → local memory（实际上在全局内存中）→ 延迟暴增。

```bash
# 查看寄存器的使用量
ncu --metrics launch__registers_per_thread ./my_program
```

**减少寄存器压力的方法**：

- 减小 block 大小 → 每个线程可用寄存器更多
- 用 `__launch_bounds__` 提示编译器

```cuda
__global__ void __launch_bounds__(256, 2)  // maxThreads=256, minBlocks=2
my_kernel(float *data) { ... }
```

- 手动管理变量生命周期（`{ scope }` 块提前释放寄存器）
- 将大型常量表移到 `__constant__` 内存

### 6.3 Warp 级原语

```cuda
// Shuffle: warp 内线程交换数据，无需共享内存
float val = __shfl_xor_sync(0xffffffff, val, 1);  // 与相邻线程交换

// Warp 内的 reduction（求和）
for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
}

// Vote
int result = __all_sync(0xffffffff, pred);   // 所有线程 pred 为 true？
int result = __any_sync(0xffffffff, pred);   // 任一线程 pred 为 true？
int mask = __ballot_sync(0xffffffff, pred);  // 返回满足条件的线程掩码
```

### 6.4 向量化内存访问

```cuda
// 用 float4 一次读取 4 个 float（128 位/事务）
float4 val = reinterpret_cast<float4*>(global_ptr)[idx];

// 等价于：
// float a = global_ptr[idx*4+0];
// float b = global_ptr[idx*4+1];
// float c = global_ptr[idx*4+2];
// float d = global_ptr[idx*4+3];
// 但只用 1 个内存事务，而非 4 个
```

## 7. 占用率（Occupancy）

### 7.1 定义

```
Occupancy = Active Warps per SM / Max Warps per SM

Max Warps per SM: 通常为 64（取决于硬件架构）
```

高占用率 → 更多 warp 可以在等待内存时切换执行 → 更好隐藏延迟。

### 7.2 占用率计算因素

1. **每个 block 的线程数**：`blockDim.x`
2. **每个线程的寄存器数**：编译器决定（`--ptxas-options=-v` 查看）
3. **每个 block 的共享内存**：`__shared__` + kernel 参数

```
// CUDA Occupancy Calculator 手动计算
Max blocks per SM  = min(
    MaxBlocksPerSM,                  // 通常 32
    MaxRegsPerSM / RegsPerThread,    // 65536 / regs_per_thread
    MaxSMemPerSM / SMemPerBlock      // 共享内存限制
)

Occupancy = (Max blocks per SM * ThreadsPerBlock) / MaxWarpsPerSM / 32
```

### 7.3 占用率优化实践

```cuda
// 用 API 确定最优 block 大小
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                    my_kernel, 0, 0);
// blockSize 现在是理论最优值

my_kernel<<<minGridSize, blockSize>>>(...);
```

**Trade-off**：高占用率不一定意味着最高性能。如果每个线程做了大量工作（指令级并行度高），较低占用率就可能足够隐藏延迟。

## 8. 机器人 GPU 优化实战

### 8.1 点云处理优化

```cuda
// 点云变换 （旋转 + 平移）
// 每个线程处理一个点：天然合并、高占用率
__global__ void transform_points(float3 *points, float3 *out,
                                  float *R, float *T, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float3 p = points[idx];
    out[idx].x = R[0]*p.x + R[1]*p.y + R[2]*p.z + T[0];
    out[idx].y = R[3]*p.x + R[4]*p.y + R[5]*p.z + T[1];
    out[idx].z = R[6]*p.x + R[7]*p.y + R[8]*p.z + T[2];
}
```

### 8.2 视觉预处理流水线

```
Camera frame → [GPU: demosaic + white_balance + resize]
              ↓
              [CPU: extract metadata, allocate tensor]
              ↓
              [GPU: normalize + convert to blob (NCHW)]
              ↓
              [GPU: inference (TensorRT)]
              ↓
              [CPU: post-process bounding boxes]
```

使用 CUDA streams 保持 GPU 不休眠；使用固定内存加速 CPU↔GPU 传输；使用 NVTX 注释方便 Nsight 分析。

### 8.3 GPU 优化后检查清单

```
[ ] 使用 ncu 确认全局内存合并访问 ≥ 80%
[ ] L1 缓存命中率 > 50%，L2 > 40%
[ ] 占用率 > 50%（计算密集型 kernel）
[ ] 使用 Nsight Systems 确认无 GPU 空闲气泡
[ ] 数据传输与计算重叠（启用 streams）
[ ] 已在共享内存中 tiling 矩阵运算
[ ] 注意 bank conflict 影响（在 load 和 store 时检查）
[ ] 已验证 FP16 推理与 FP32 误差 < 0.1%
[ ] 已在 NVTX 中标注关键区间
[ ] 确认无 warp divergence（使用 ncu 分析分支效率）
```
