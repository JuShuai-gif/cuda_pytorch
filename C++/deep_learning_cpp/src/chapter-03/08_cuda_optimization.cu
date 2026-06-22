/*
 * 08_cuda_optimization.cu - 第 3 章：CUDA GPU 加速深度学习
 * CUDA 常见优化技术：合并内存访问与共享内存分块
 *
 * 本章演示 CUDA 编程中最关键的两种优化技术：
 *   优化 1：合并内存访问（Coalesced Access）
 *     - 非合并访问（列优先跨步访问）→ 带宽浪费，多个内存事务
 *     - 合并访问（行优先连续访问）→ 最大化内存总线利用率
 *   优化 2：共享内存分块（Shared Memory Tiling）
 *     - 朴素版本（全部从全局内存读取）→ 重复读取，高延迟
 *     - 共享内存分块版本 → 一次性加载到片上 SRAM，减少全局内存访问
 *
 * 每种优化均提供「优化前 vs 优化后」的带宽对比。
 */

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <chrono>

// ----------------------------------------------------------------
// CUDA 错误检查工具
// ----------------------------------------------------------------
void checkCudaErrors(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA 错误 [" << msg << "]: "
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// ================================================================
// 辅助：使用 cudaEvent 对核函数进行计时
// ================================================================
float timeKernel(void (*kernel)(int, float *, float *, float *),
                 int N, float *a, float *b, float *c,
                 int gridSize, int blockSize) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();
    cudaEventRecord(start);
    kernel<<<gridSize, blockSize>>>(N, a, b, c);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaGetLastError(); // 检查内核错误

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

// 重载：对矩阵加法的计时版本（需要 rows, cols 而非 N）
float timeMatrixKernel(void (*kernel)(int, int, float *, float *, float *),
                       int rows, int cols, float *a, float *b, float *c,
                       int gridSize, int blockSize) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();
    cudaEventRecord(start);
    kernel<<<gridSize, blockSize>>>(rows, cols, a, b, c);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaGetLastError();

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

// ================================================================
// 第 0 部分：基线向量加法核函数
// ================================================================
__global__ void baselineVectorAdd(int n, float *a, float *b, float *c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i]; // 每个线程连续跨步访问——合并访问
    }
}

// ================================================================
// 优化 1：合并内存访问 vs 非合并内存访问
// ================================================================
// 将一维数组 N 个元素视为 rows × cols 的二维矩阵
// N = rows * cols = 2048 * 2048 = 4,194,304 (1<<22)

// --- 非合并访问版本：每个线程处理一行（列优先跨步） ---
// 线程 i 负责第 i 行，遍历该行的所有列
// 同一 warp 中的 32 个线程分别访问 c[0], c[cols], c[2*cols], ...
// 这些地址相隔 cols 个元素，导致每次内存事务只传输少量有效数据
__global__ void addNonCoalesced(int rows, int cols,
                                float *a, float *b, float *c) {
    // 线程 ID 映射到行编号（非合并访问的关键：相邻线程访问非相邻地址）
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        for (int col = 0; col < cols; col++) {
            int idx = row * cols + col; // 行优先的一维索引
            c[idx] = a[idx] + b[idx];
        }
        // 问题：在线程 0 读取 c[0] 时，线程 1 读取 c[2048]
        // 两者相隔 8 KB —— 需要多个独立的内存事务
    }
}

// --- 合并访问版本：每个线程处理一列（行优先连续） ---
// 线程 i 负责第 i 列，遍历该列的所有行
// 同一 warp 中线程 0~31 分别访问 c[0], c[1], ..., c[31]
// 这些地址连续，128 字节对齐，一个内存事务即可覆盖整个 warp
__global__ void addCoalesced(int rows, int cols,
                             float *a, float *b, float *c) {
    // 线程 ID 映射到列编号（合并访问的关键：相邻线程访问相邻地址）
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        for (int row = 0; row < rows; row++) {
            int idx = row * cols + col; // 行优先的一维索引
            c[idx] = a[idx] + b[idx];
        }
        // 优势：在线程 0 读取 c[0] 时，线程 1 读取 c[1]，
        // 32 个连续地址 (<128B) 一次内存事务即可完成
    }
}

// ================================================================
// 优化 2：共享内存分块 —— 1D 块归约（Block Reduction / Sum）
// ================================================================
// 归约操作：将每个 block 内的 N/blockSize 个元素求和为 1 个元素
// 这是深度学习中的常见基础操作：loss 求和、norm 计算、softmax 归一化等

// --- 朴素版本：全局内存原子归约 ---
// 每个线程先用 grid-stride 计算局部和，再用 atomicAdd 跨线程合并
// 问题：大量线程争用同一个原子地址 → 串行化开销极大
__global__ void naiveReduction(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 阶段 1：grid-stride 计算当前线程的局部部分和
    float localSum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        localSum += in[i];
    }

    // 阶段 2：使用原子加法将局部和归并到块级输出
    // ⚠ 性能瓶颈：同一 block 内的 256 个线程争用同一个原子地址
    atomicAdd(&out[blockIdx.x], localSum);
}

// --- 共享内存版本：树状并行归约 ---
// 将块内数据加载到共享内存，用 log₂(N) 步树状归约，避免原子操作
__global__ void sharedReduction(const float *in, float *out, int n) {
    // 动态共享内存，用于块内归约的中间缓冲区
    extern __shared__ float sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 阶段 1：grid-stride 计算每个线程的局部部分和，
    //        存入共享内存（sdata 用作初始缓冲区）
    float localSum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        localSum += in[i];
    }
    sdata[tid] = localSum;
    __syncthreads(); // 确保所有线程写入完成

    // 阶段 2：树状并行归约（log₂(blockSize) 步）
    // 步长从 blockDim.x/2 开始，每步减半
    // 仅前半部分活跃线程参与计算，后半部分线程闲置
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s]; // 两两合并
        }
        __syncthreads(); // 必须！确保每步的读写顺序正确
    }

    // 阶段 3：线程 0 将块内最终归约结果写入全局内存
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
    // 总共仅 blockDim.x 次全局内存读取（块起始偏移）+ 1 次写入
    // 中间归约全部在片上 SRAM 完成
}

// ================================================================
// main() —— 驱动所有优化演示
// ================================================================
int main() {
    // 输出设备信息
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "================================================" << std::endl;
    std::cout << "=== 08_cuda_optimization.cu                     ===" << std::endl;
    std::cout << "=== CUDA 优化技术演示                            ===" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "计算能力: " << prop.major << "." << prop.minor << std::endl;

    const int N = 1 << 22; // 4,194,304 个元素
    const size_t bytes = N * sizeof(float);
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // ---- 分配统一内存 ----
    float *a, *b, *c;
    checkCudaErrors(cudaMallocManaged(&a, bytes), "cudaMallocManaged a");
    checkCudaErrors(cudaMallocManaged(&b, bytes), "cudaMallocManaged b");
    checkCudaErrors(cudaMallocManaged(&c, bytes), "cudaMallocManaged c");

    // ============================================================
    // 第 0 部分：基线向量加法（建立性能参考基线）
    // ============================================================
    std::cout << "\n[第 0 部分] 基线向量加法 —— 建立性能参考" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "N = " << N << " (2^22) 个 float 元素" << std::endl;

    // 预热：先运行一次解决页迁移和上下文初始化
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
        c[i] = 0.0f;
    }
    cudaDeviceSynchronize();
    baselineVectorAdd<<<gridSize, blockSize>>>(N, a, b, c);
    cudaDeviceSynchronize();

    // 正式计时：运行 3 次取最小值（排除干扰）
    float baselineMin = 1e9f;
    for (int r = 0; r < 3; r++) {
        for (int i = 0; i < N; i++) { c[i] = 0.0f; }
        float t = timeKernel(baselineVectorAdd, N, a, b, c, gridSize, blockSize);
        if (t < baselineMin) baselineMin = t;
    }

    // 基线带宽：3 × N × 4 字节 / 时间 = 读 a + 读 b + 写 c
    float baselineBW = (3.0f * bytes) / (baselineMin / 1000.0f) / 1e9f;
    std::cout << "  基线执行时间: " << baselineMin << " ms" << std::endl;
    std::cout << "  基线有效带宽: " << baselineBW << " GB/s" << std::endl;

    // 验证基线结果
    float maxErr = 0.0f;
    for (int i = 0; i < N; i++) maxErr = fmax(maxErr, fabs(c[i] - 3.0f));
    std::cout << "  最大误差: " << maxErr << " (应为 ~0)" << std::endl;

    // ============================================================
    // 优化 1：合并内存访问 vs 非合并内存访问
    // ============================================================
    std::cout << "\n==================================================" << std::endl;
    std::cout << "  [优化 1] 合并内存访问 vs 非合并内存访问" << std::endl;
    std::cout << "==================================================" << std::endl;

    // 将 N 个元素重新解释为 2048 × 2048 的二维矩阵
    const int ROWS = 2048;
    const int COLS = 2048;
    std::cout << "  矩阵维度: " << ROWS << " × " << COLS << " = " << (ROWS * COLS) << " 元素" << std::endl;

    // 非合并访问的网格：每个线程处理一行
    int gridRows = (ROWS + blockSize - 1) / blockSize;
    // 合并访问的网格：每个线程处理一列
    int gridCols = (COLS + blockSize - 1) / blockSize;

    // --- 测试非合并访问版本 ---
    for (int i = 0; i < N; i++) { c[i] = 0.0f; }
    // 预热
    cudaDeviceSynchronize();
    addNonCoalesced<<<gridRows, blockSize>>>(ROWS, COLS, a, b, c);
    cudaDeviceSynchronize();

    float nonCoalescedMin = 1e9f;
    for (int r = 0; r < 3; r++) {
        for (int i = 0; i < N; i++) { c[i] = 0.0f; }
        float t = timeMatrixKernel(addNonCoalesced, ROWS, COLS,
                                   a, b, c, gridRows, blockSize);
        if (t < nonCoalescedMin) nonCoalescedMin = t;
    }
    float nonCoalescedBW = (3.0f * bytes) / (nonCoalescedMin / 1000.0f) / 1e9f;

    std::cout << "\n  --- 非合并访问（线程→行，列优先跨步） ---" << std::endl;
    std::cout << "    访问模式: 线程 i 处理第 i 行" << std::endl;
    std::cout << "    问题: warp 内相邻线程访问地址间隔 " << COLS << " 个元素" << std::endl;
    std::cout << "    [" << COLS << " × 4B = " << (COLS * 4 / 1024) << " KB 跨步]" << std::endl;
    std::cout << "    执行时间: " << nonCoalescedMin << " ms" << std::endl;
    std::cout << "    有效带宽: " << nonCoalescedBW << " GB/s" << std::endl;

    // --- 测试合并访问版本 ---
    for (int i = 0; i < N; i++) { c[i] = 0.0f; }
    cudaDeviceSynchronize();
    addCoalesced<<<gridCols, blockSize>>>(ROWS, COLS, a, b, c);
    cudaDeviceSynchronize();

    float coalescedMin = 1e9f;
    for (int r = 0; r < 3; r++) {
        for (int i = 0; i < N; i++) { c[i] = 0.0f; }
        float t = timeMatrixKernel(addCoalesced, ROWS, COLS,
                                   a, b, c, gridCols, blockSize);
        if (t < coalescedMin) coalescedMin = t;
    }
    float coalescedBW = (3.0f * bytes) / (coalescedMin / 1000.0f) / 1e9f;

    std::cout << "\n  --- 合并访问（线程→列，行优先连续） ---" << std::endl;
    std::cout << "    访问模式: 线程 i 处理第 i 列" << std::endl;
    std::cout << "    优势: warp 内相邻线程访问连续地址（128B 对齐）" << std::endl;
    std::cout << "    执行时间: " << coalescedMin << " ms" << std::endl;
    std::cout << "    有效带宽: " << coalescedBW << " GB/s" << std::endl;

    // --- 合并访问加速比 ---
    float coalescedSpeedup = nonCoalescedMin / coalescedMin;
    std::cout << "\n  >>> 合并访问加速比: " << coalescedSpeedup << "×" << std::endl;
    std::cout << "      非合并带宽: " << nonCoalescedBW << " GB/s" << std::endl;
    std::cout << "      合并带宽:   " << coalescedBW << " GB/s" << std::endl;
    std::cout << "      原因: 合并访问将多次内存事务合并为单次 128B 事务，" << std::endl;
    std::cout << "            消除了跨步访问造成的内存带宽浪费" << std::endl;

    // ============================================================
    // 优化 2：共享内存分块 —— 1D 块归约 (Block Reduction)
    // ============================================================
    std::cout << "\n==================================================" << std::endl;
    std::cout << "  [优化 2] 共享内存分块 —— 1D 块归约 (Reduction)" << std::endl;
    std::cout << "==================================================" << std::endl;

    // 归约参数：使用较少的 block，让每个线程处理更多元素
    // 这样可以分摊 __syncthreads() 的开销，充分体现共享内存优势
    const int redBlockSize = 256;
    const int redGridSize = 512; // 少量 block → 多迭代/线程
    // 每个线程处理 N / (redGridSize * redBlockSize) = 32 个元素
    const int outSize = redGridSize; // 512 个归约结果

    std::cout << "  操作: 每个 Block 独立将 chunk 归约为 1 个和" << std::endl;
    std::cout << "  N = " << N << " 个输入元素，输出 " << outSize << " 个归约结果" << std::endl;
    std::cout << "  每线程迭代次数: " << (N / (redGridSize * redBlockSize)) << std::endl;
    std::cout << "  启动配置: <<<" << redGridSize << ", " << redBlockSize << ">>>" << std::endl;
    std::cout << "  每个输出地址被 " << redBlockSize << " 个线程原子争用" << std::endl;

    // 归约输入数组：全部填 1.0
    float *redIn, *redOut;
    checkCudaErrors(cudaMallocManaged(&redIn, bytes), "cudaMallocManaged redIn");
    checkCudaErrors(cudaMallocManaged(&redOut, outSize * sizeof(float)), "cudaMallocManaged redOut");
    for (int i = 0; i < N; i++) { redIn[i] = 1.0f; }
    for (int i = 0; i < outSize; i++) { redOut[i] = 0.0f; }

    // --- 朴素归约版本（全局内存原子加法） ---
    // 预热
    cudaDeviceSynchronize();
    naiveReduction<<<redGridSize, redBlockSize>>>(redIn, redOut, N);
    cudaDeviceSynchronize();

    float naiveRedMin = 1e9f;
    for (int r = 0; r < 5; r++) {
        for (int i = 0; i < outSize; i++) { redOut[i] = 0.0f; }
        cudaDeviceSynchronize();
        cudaEvent_t s, e;
        cudaEventCreate(&s);
        cudaEventCreate(&e);
        cudaEventRecord(s);
        naiveReduction<<<redGridSize, redBlockSize>>>(redIn, redOut, N);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float t = 0.0f;
        cudaEventElapsedTime(&t, s, e);
        cudaGetLastError();
        cudaEventDestroy(s);
        cudaEventDestroy(e);
        if (t < naiveRedMin) naiveRedMin = t;
    }

    std::cout << "\n  --- 朴素版本（全局内存原子加法） ---" << std::endl;
    std::cout << "    方法: 每线程 grid-stride 计算局部和，" << std::endl;
    std::cout << "          再 atomicAdd 到块级输出" << std::endl;
    std::cout << "    问题: 每块 256 线程争用同一原子地址 → 串行化" << std::endl;
    std::cout << "    每个 Block 全局内存操作: "
              << (N / redGridSize) << " 次读取 + " << redBlockSize << " 次原子写" << std::endl;
    std::cout << "    执行时间: " << naiveRedMin << " ms" << std::endl;

    // --- 共享内存树状归约版本 ---
    for (int i = 0; i < outSize; i++) { redOut[i] = 0.0f; }
    // 共享内存大小：每块 redBlockSize 个 float
    size_t redSmemBytes = redBlockSize * sizeof(float);

    // 预热
    cudaDeviceSynchronize();
    sharedReduction<<<redGridSize, redBlockSize, redSmemBytes>>>(redIn, redOut, N);
    cudaDeviceSynchronize();

    float sharedRedMin = 1e9f;
    for (int r = 0; r < 5; r++) {
        for (int i = 0; i < outSize; i++) { redOut[i] = 0.0f; }
        cudaDeviceSynchronize();
        cudaEvent_t s, e;
        cudaEventCreate(&s);
        cudaEventCreate(&e);
        cudaEventRecord(s);
        sharedReduction<<<redGridSize, redBlockSize, redSmemBytes>>>(redIn, redOut, N);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float t = 0.0f;
        cudaEventElapsedTime(&t, s, e);
        cudaGetLastError();
        cudaEventDestroy(s);
        cudaEventDestroy(e);
        if (t < sharedRedMin) sharedRedMin = t;
    }

    std::cout << "\n  --- 共享内存树状归约版本 ---" << std::endl;
    std::cout << "    每块共享内存: " << redSmemBytes << " 字节 ("
              << redBlockSize << " float × 4B)" << std::endl;
    std::cout << "    方法: 加载到共享内存，log₂(256)=8 步树状归约" << std::endl;
    std::cout << "    优势: 零原子操作，全部在片上 SRAM 完成" << std::endl;
    std::cout << "    每个 Block 全局内存操作: " << (N / redGridSize)
              << " 次读取 + 1 次写入" << std::endl;
    std::cout << "    执行时间: " << sharedRedMin << " ms" << std::endl;

    // --- 共享内存加速比 ---
    float sharedSpeedup = naiveRedMin / sharedRedMin;
    std::cout << "\n  >>> 共享内存归约加速比: " << sharedSpeedup << "×" << std::endl;
    std::cout << "      朴素版本 (原子): " << naiveRedMin << " ms" << std::endl;
    std::cout << "      分块版本 (共享): " << sharedRedMin << " ms" << std::endl;
    std::cout << "      原因: 共享内存树状归约避免了原子争用瓶颈，" << std::endl;
    std::cout << "            将 O(N) 原子操作替换为 O(log₂N) 步并行归约" << std::endl;

    // 验证归约结果
    // 全部输入为 1.0 → 每个块的和 = N / redGridSize = 8192
    const float expectedBlockSum = (float)N / redGridSize;
    float maxRedErr = 0.0f;
    for (int i = 0; i < outSize; i++) {
        maxRedErr = fmax(maxRedErr, fabs(redOut[i] - expectedBlockSum));
    }
    std::cout << "\n  最大误差: " << maxRedErr
              << " (期望每个块和 = " << expectedBlockSum << ")" << std::endl;

    // ============================================================
    // 最终总结
    // ============================================================
    std::cout << "\n==================================================" << std::endl;
    std::cout << "=== 优化总结                                       ===" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "  基础操作      基线带宽:   " << baselineBW << " GB/s" << std::endl;
    std::cout << "               基线时间:   " << baselineMin << " ms" << std::endl;
    std::cout << std::endl;
    std::cout << "  优化 1: 合并内存访问" << std::endl;
    std::cout << "    非合并访问: " << nonCoalescedMin << " ms @ " << nonCoalescedBW << " GB/s" << std::endl;
    std::cout << "    合并访问:   " << coalescedMin << " ms @ " << coalescedBW << " GB/s" << std::endl;
    std::cout << "    加速比:     " << coalescedSpeedup << "×" << std::endl;
    std::cout << "    要点: 确保同一 warp 内相邻线程访问相邻内存地址" << std::endl;
    std::cout << std::endl;
    std::cout << "  优化 2: 共享内存分块（块归约）" << std::endl;
    std::cout << "    朴素版本 (原子): " << naiveRedMin << " ms" << std::endl;
    std::cout << "    分块版本 (共享): " << sharedRedMin << " ms" << std::endl;
    std::cout << "    加速比:           " << sharedSpeedup << "×" << std::endl;
    std::cout << "    要点: 使用共享内存树状归约消除原子争用瓶颈" << std::endl;
    std::cout << "==================================================" << std::endl;

    // 清理资源
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(redIn);
    cudaFree(redOut);

    return 0;
}
