#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call)                                   \
    do {                                                   \
        cudaError_t err = (call);                          \
        if (err != cudaSuccess) {                          \
            std::cerr << "CUDA error: "                    \
                      << cudaGetErrorString(err)           \
                      << " at " << __FILE__                \
                      << ':' << __LINE__ << '\n';          \
            std::exit(EXIT_FAILURE);                       \
        }                                                  \
    } while (0)

constexpr int N = 1 << 26;  // 67,108,864 个 float ≈ 256 MB
constexpr int BLOCKS = 512;
constexpr int THREADS = 256;

// ----------------------------------------------------------------
// 初始化：x = [-1, 1, -1, 1, ...]，期望 sum(abs(x)) == N
// ----------------------------------------------------------------
__global__ void init_kernel(float* x, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        x[i] = (i % 2 == 0) ? -1.0f : 1.0f;
    }
}

// ----------------------------------------------------------------
// 未融合版本：两个独立 Kernel
// ----------------------------------------------------------------

/*
 * Kernel 1: 逐元素取绝对值，写入中间数组 tmp。
 * 数据通路：global x → register(abs) → global tmp
 */
__global__ void abs_kernel(
    const float* x,
    float* tmp,
    int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        tmp[i] = fabsf(x[i]);
    }
}

/*
 * Kernel 2: 对 tmp 求和。用共享内存做 Block 级归约，
 * 每个 Block 通过 atomicAdd 累加一个部分和。
 *
 * 数据通路：global tmp → register → shared memory → atomicAdd → result
 */
template<int BLOCK_SIZE>
__global__ void sum_kernel(
    const float* tmp,
    float* result,
    int n)
{
    __shared__ float shared_sum[BLOCK_SIZE];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float thread_sum = 0.0f;

    for (int i = index; i < n; i += stride) {
        thread_sum += tmp[i];
    }

    shared_sum[tid] = thread_sum;
    __syncthreads();

    for (int offset = BLOCK_SIZE / 2;
         offset > 0;
         offset >>= 1) {

        if (tid < offset) {
            shared_sum[tid] += shared_sum[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, shared_sum[0]);
    }
}

// ----------------------------------------------------------------
// 融合版本：一个 Kernel 同时完成 abs 和 sum
// ----------------------------------------------------------------

/*
 * 融合 Kernel：从 x 读取后，直接在寄存器中计算 fabsf 并累加，
 * 不再需要 tmp 中间数组，也不再有 global ↔ tmp 的额外读写。
 *
 * 数据通路：global x → register(abs+累加) → shared memory → atomicAdd → result
 *
 * 关键优化：
 *   - 省掉了写入 tmp 的一次全局显存写
 *   - 省掉了从 tmp 读取的一次全局显存读
 *   - 中间值 abs(x[i]) 从未离开寄存器，生命周期极短
 */
template<int BLOCK_SIZE>
__global__ void sum_abs_kernel(
    const float* x,
    float* result,
    int n)
{
    __shared__ float shared_sum[BLOCK_SIZE];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float thread_sum = 0.0f;

    for (int i = index; i < n; i += stride) {
        // abs 的结果直接在寄存器中使用，无需经过全局显存。
        thread_sum += fabsf(x[i]);
    }

    shared_sum[tid] = thread_sum;
    __syncthreads();

    for (int offset = BLOCK_SIZE / 2;
         offset > 0;
         offset >>= 1) {

        if (tid < offset) {
            shared_sum[tid] += shared_sum[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, shared_sum[0]);
    }
}

// ----------------------------------------------------------------
// 工具函数：使用 CUDA Event 计时的单次测试
// ----------------------------------------------------------------
float run_unfused(
    float* d_x,
    float* d_tmp,
    float* d_result,
    cudaEvent_t start,
    cudaEvent_t stop)
{
    CUDA_CHECK(cudaMemset(
        d_result,
        0,
        sizeof(float)));

    CUDA_CHECK(cudaEventRecord(start));

    abs_kernel<<<BLOCKS, THREADS>>>(
        d_x,
        d_tmp,
        N);

    sum_kernel<THREADS><<<BLOCKS, THREADS>>>(
        d_tmp,
        d_result,
        N);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &ms,
        start,
        stop));

    return ms;
}

float run_fused(
    float* d_x,
    float* d_result,
    cudaEvent_t start,
    cudaEvent_t stop)
{
    CUDA_CHECK(cudaMemset(
        d_result,
        0,
        sizeof(float)));

    CUDA_CHECK(cudaEventRecord(start));

    sum_abs_kernel<THREADS><<<BLOCKS, THREADS>>>(
        d_x,
        d_result,
        N);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &ms,
        start,
        stop));

    return ms;
}

// ----------------------------------------------------------------
// main
// ----------------------------------------------------------------
int main()
{
    try {
        CUDA_CHECK(cudaSetDevice(0));

        cudaDeviceProp deviceProperties{};
        CUDA_CHECK(cudaGetDeviceProperties(
            &deviceProperties,
            0));

        std::cout << "GPU: "
                  << deviceProperties.name << '\n';

        // 理论显存带宽（GB/s），用于计算分析
        const double bwGBs =
            deviceProperties.memoryClockRate * 1e3 *
            (deviceProperties.memoryBusWidth / 8) * 2.0 /
            1e9;

        std::cout << "Theoretical BW:  "
                  << bwGBs << " GB/s\n";

        // --------------------------------------------------------
        // 分配显存
        // --------------------------------------------------------
        float* d_x = nullptr;
        float* d_tmp = nullptr;
        float* d_result = nullptr;

        CUDA_CHECK(cudaMalloc(
            &d_x,
            N * sizeof(float)));

        CUDA_CHECK(cudaMalloc(
            &d_tmp,
            N * sizeof(float)));

        CUDA_CHECK(cudaMalloc(
            &d_result,
            sizeof(float)));

        // 初始化 x 数组
        init_kernel<<<BLOCKS, THREADS>>>(d_x, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        // --------------------------------------------------------
        // 显存流量分析（理论值）
        // --------------------------------------------------------
        const double dataSizeMB =
            static_cast<double>(N * sizeof(float)) /
            (1024.0 * 1024.0);

        std::cout << "\n[Memory traffic analysis]\n";
        std::cout << "  Data size:            "
                  << dataSizeMB << " MB\n";

        std::cout << "  Unfused:\n";
        std::cout << "    Read x        "
                  << dataSizeMB << " MB\n";

        std::cout << "    Write tmp     "
                  << dataSizeMB << " MB\n";

        std::cout << "    Read tmp      "
                  << dataSizeMB << " MB\n";

        std::cout << "    Total         "
                  << dataSizeMB * 3 << " MB\n";

        std::cout << "  Fused:\n";
        std::cout << "    Read x        "
                  << dataSizeMB << " MB\n";

        std::cout << "    Total         "
                  << dataSizeMB << " MB\n";

        // --------------------------------------------------------
        // 创建 CUDA Event 用于计时
        // --------------------------------------------------------
        cudaEvent_t start;
        cudaEvent_t stop;

        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // --------------------------------------------------------
        // 预热：避免首次 CUDA 调用影响计时
        // --------------------------------------------------------
        std::cout << "\n[Warmup]\n";

        run_unfused(d_x, d_tmp, d_result, start, stop);
        run_fused(d_x, d_result, start, stop);

        // --------------------------------------------------------
        // 正式测量
        // --------------------------------------------------------
        constexpr int REPEAT = 10;

        std::cout << "\n[Benchmark (" << REPEAT
                  << " runs)]\n";

        float best_unfused = 1e9f;
        float best_fused = 1e9f;

        for (int i = 0; i < REPEAT; ++i) {
            float t = run_unfused(
                d_x, d_tmp, d_result, start, stop);

            if (t < best_unfused) {
                best_unfused = t;
            }

            t = run_fused(d_x, d_result, start, stop);

            if (t < best_fused) {
                best_fused = t;
            }
        }

        // --------------------------------------------------------
        // 最终验证：运行一次完整版本并读取结果
        // --------------------------------------------------------
        CUDA_CHECK(cudaMemset(
            d_result,
            0,
            sizeof(float)));

        abs_kernel<<<BLOCKS, THREADS>>>(
            d_x,
            d_tmp,
            N);

        sum_kernel<THREADS><<<BLOCKS, THREADS>>>(
            d_tmp,
            d_result,
            N);

        CUDA_CHECK(cudaDeviceSynchronize());

        float naive_result = 0.0f;

        CUDA_CHECK(cudaMemcpy(
            &naive_result,
            d_result,
            sizeof(float),
            cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemset(
            d_result,
            0,
            sizeof(float)));

        sum_abs_kernel<THREADS><<<BLOCKS, THREADS>>>(
            d_x,
            d_result,
            N);

        CUDA_CHECK(cudaDeviceSynchronize());

        float fused_result = 0.0f;

        CUDA_CHECK(cudaMemcpy(
            &fused_result,
            d_result,
            sizeof(float),
            cudaMemcpyDeviceToHost));

        // --------------------------------------------------------
        // 输出结果
        // --------------------------------------------------------
        std::cout << "\n[Results]\n";
        std::cout << "  Expected:        "
                  << N << '\n';

        std::cout << "  Unfused result:  "
                  << naive_result << '\n';

        std::cout << "  Fused result:    "
                  << fused_result << '\n';

        std::cout << "  Unfused time:    "
                  << best_unfused << " ms (best)\n";

        std::cout << "  Fused time:      "
                  << best_fused << " ms (best)\n";

        std::cout << "  Speedup:         "
                  << best_unfused / best_fused << "x\n";

        // 实际显存带宽利用率（粗略计算）
        std::cout << "\n[Observed BW (approx)]\n";

        // 未融合：3 倍数据量的读写 / 时间
        std::cout << "  Unfused:         "
                  << dataSizeMB * 3.0 / best_unfused * 1000.0 / 1024.0
                  << " GB/s\n";

        // 融合：1 倍数据量的读写 / 时间
        std::cout << "  Fused:           "
                  << dataSizeMB * 1.0 / best_fused * 1000.0 / 1024.0
                  << " GB/s\n";

        // 清理
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        CUDA_CHECK(cudaFree(d_result));
        CUDA_CHECK(cudaFree(d_tmp));
        CUDA_CHECK(cudaFree(d_x));

        return EXIT_SUCCESS;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
