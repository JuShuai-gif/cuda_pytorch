#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "timer.h"
#include "matmul_naive.cuh"
#include "matmul_tiled.cuh"
#include "coalesced_demo.cuh"
#include "stream_pipeline.cuh"
#include "kernel_fusion.cuh"

// 辅助函数：打印节标题
static void print_header(const std::string &title) {
    std::cout << "\n"
              << std::string(70, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(70, '=') << "\n";
}

// ============================================================================
// 演示 1: 矩阵乘法 - 朴素 vs 分块 vs 优化
// ============================================================================
void demo_matmul() {
    print_header("演示 1: 矩阵乘法优化");

    constexpr int N = 1024;
    constexpr size_t BYTES = N * N * sizeof(float);

    // 分配主机内存
    std::vector<float> h_A(N * N), h_B(N * N), h_C_naive(N * N),
        h_C_tiled(N * N), h_C_opt(N * N);

    // 使用随机值初始化
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, BYTES));
    CUDA_CHECK(cudaMalloc(&d_B, BYTES));
    CUDA_CHECK(cudaMalloc(&d_C, BYTES));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), BYTES, cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (N + TILE_SIZE - 1) / TILE_SIZE);

    std::vector<std::pair<std::string, float>> results;

    // 朴素内核
    {
        NVTX_RANGE_START("matmul_naive");
        GPUTimer timer;
        timer.start();
        matmul_naive<<<grid, block>>>(d_A, d_B, d_C, N);
        timer.stop();
        float ms = timer.elapsed_ms();
        results.push_back({"朴素（全局内存）", ms});
        CUDA_CHECK(cudaMemcpy(h_C_naive.data(), d_C, BYTES,
                              cudaMemcpyDeviceToHost));
        NVTX_RANGE_END();
    }

    // 分块内核
    {
        NVTX_RANGE_START("matmul_tiled");
        GPUTimer timer;
        timer.start();
        matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
        timer.stop();
        float ms = timer.elapsed_ms();
        results.push_back({"分块（共享内存）", ms});
        CUDA_CHECK(cudaMemcpy(h_C_tiled.data(), d_C, BYTES,
                              cudaMemcpyDeviceToHost));
        NVTX_RANGE_END();
    }

    // 优化的分块内核
    {
        NVTX_RANGE_START("matmul_optimized");
        GPUTimer timer;
        timer.start();
        matmul_optimized<<<grid, block>>>(d_A, d_B, d_C, N);
        timer.stop();
        float ms = timer.elapsed_ms();
        results.push_back({"优化（填充 + 展开）", ms});
        CUDA_CHECK(cudaMemcpy(h_C_opt.data(), d_C, BYTES,
                              cudaMemcpyDeviceToHost));
        NVTX_RANGE_END();
    }

    // 验证正确性
    double max_err_tiled = 0.0, max_err_opt = 0.0;
    for (int i = 0; i < N * N; ++i) {
        double err_t = std::abs(h_C_naive[i] - h_C_tiled[i]);
        double err_o = std::abs(h_C_naive[i] - h_C_opt[i]);
        if (err_t > max_err_tiled) max_err_tiled = err_t;
        if (err_o > max_err_opt) max_err_opt = err_o;
    }

    // 打印结果
    std::cout << "  矩阵大小: " << N << "x" << N
              << " (" << (N * N / 1000) << "K 元素)\n\n";
    std::cout << "  " << std::left << std::setw(35) << "内核"
              << std::right << std::setw(12) << "时间(ms)"
              << std::setw(12) << "加速比" << "\n";
    std::cout << "  " << std::string(59, '-') << "\n";

    float baseline = results[0].second;
    for (const auto &r : results) {
        std::cout << "  " << std::left << std::setw(35) << r.first
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << r.second
                  << std::fixed << std::setprecision(2)
                  << std::setw(11) << (baseline / r.second) << "x\n";
    }
    std::cout << "\n  最大误差（分块 vs 朴素）:   " << std::scientific
              << max_err_tiled << "\n";
    std::cout << "  最大误差（优化 vs 朴素）: " << std::scientific
              << max_err_opt << "\n";

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

// ============================================================================
// 演示 5: 占用率与启动配置
// ============================================================================

__global__ void register_heavy_kernel(float *data, int N) {
    // 声明多个变量以增加寄存器压力
    float r0 = 1.0f, r1 = 2.0f, r2 = 3.0f, r3 = 4.0f;
    float r4 = 5.0f, r5 = 6.0f, r6 = 7.0f, r7 = 8.0f;
    float r8 = 9.0f, r9 = 10.0f, r10 = 11.0f, r11 = 12.0f;
    float r12 = 13.0f, r13 = 14.0f, r14 = 15.0f, r15 = 16.0f;
    float r16 = 17.0f, r17 = 18.0f, r18 = 19.0f, r19 = 20.0f;
    float r20 = 21.0f, r21 = 22.0f, r22 = 23.0f, r23 = 24.0f;
    float r24 = 25.0f, r25 = 26.0f, r26 = 27.0f, r27 = 28.0f;
    float r28 = 29.0f, r29 = 30.0f, r30 = 31.0f, r31 = 32.0f;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float val = data[idx];
    val += r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
    val += r8 + r9 + r10 + r11 + r12 + r13 + r14 + r15;
    val += r16 + r17 + r18 + r19 + r20 + r21 + r22 + r23;
    val += r24 + r25 + r26 + r27 + r28 + r29 + r30 + r31;
    data[idx] = val;
}

__global__ void register_light_kernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float val = data[idx];
    val = val * 2.0f + 1.0f;
    data[idx] = val;
}

void demo_occupancy() {
    print_header("演示 5: 占用率与启动配置");

    constexpr int N = 16 * 1024 * 1024;
    constexpr size_t BYTES = N * sizeof(float);

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, BYTES));

    // 演示 cudaOccupancyMaxPotentialBlockSize
    std::cout << "  register_heavy_kernel 的占用率分析:\n\n";

    int min_grid, block_size;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &min_grid, &block_size, register_heavy_kernel, 0, 0));
    std::cout << "  高寄存器压力内核 - 推荐线程块大小: " << block_size
              << " (最小网格数: " << min_grid << ")\n";

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &min_grid, &block_size, register_light_kernel, 0, 0));
    std::cout << "  低寄存器压力内核 - 推荐线程块大小: " << block_size
              << " (最小网格数: " << min_grid << ")\n\n";

    // 对高寄存器压力内核测试不同线程块大小
    std::cout << "  " << std::left << std::setw(16) << "线程块大小"
              << std::right << std::setw(14) << "活跃线程块数"
              << std::setw(12) << "时间(ms)" << "\n";
    std::cout << "  " << std::string(42, '-') << "\n";

    for (int bs : {32, 64, 128, 256, 512, 1024}) {
        int grid_size = (N + bs - 1) / bs;
        int active_blocks;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks, register_heavy_kernel, bs, 0));

        GPUTimer timer;
        timer.start();
        register_heavy_kernel<<<grid_size, bs>>>(d_data, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop();
        float ms = timer.elapsed_ms();

        std::cout << "  " << std::left << std::setw(16) << bs
                  << std::right << std::setw(14) << active_blocks
                  << std::fixed << std::setprecision(3)
                  << std::setw(12) << ms << "\n";
    }

    std::cout << "\n  => 较大的线程块大小会减少每个 SM 上的最大活跃线程块数\n"
              << "  原因是寄存器压力。最佳点在占用率与\n"
              << "  指令级并行之间取得平衡。\n";

    CUDA_CHECK(cudaFree(d_data));
}

// ============================================================================
// 性能总结表格
// ============================================================================
void print_performance_summary(
    const std::vector<std::pair<std::string, std::string>> &entries) {
    std::cout << "\n  " << std::left << std::setw(40) << "技术"
              << std::setw(30) << "预期收益" << "\n";
    std::cout << "  " << std::string(70, '-') << "\n";
    for (const auto &e : entries) {
        std::cout << "  " << std::left << std::setw(40) << e.first
                  << std::setw(30) << e.second << "\n";
    }
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    // 打印 CUDA 设备信息
    int dev_count;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));

    std::cout << "=====================================================\n";
    std::cout << "  GPU 优化演示\n";
    std::cout << "  发现的设备数: " << dev_count << "\n";

    for (int d = 0; d < dev_count; ++d) {
        cudaDeviceProp props;
        CUDA_CHECK(cudaGetDeviceProperties(&props, d));
        std::cout << "  设备 " << d << ": " << props.name << "\n";
        std::cout << "    SM 数: " << props.multiProcessorCount
                  << ", 最大线程/块: " << props.maxThreadsPerBlock << "\n";
        std::cout << "    共享内存/块: " << (props.sharedMemPerBlock / 1024)
                  << " KB, 寄存器/块: " << props.regsPerBlock << "\n";
        std::cout << "    全局内存: " << (props.totalGlobalMem / (1024 * 1024 * 1024))
                  << " GB\n";
    }
    std::cout << "=====================================================\n";

    NVTX_RANGE_START("main_all_demos");

    demo_matmul();
    demo_memory_coalescing();
    demo_cuda_streams();
    demo_kernel_fusion();
    demo_occupancy();

    NVTX_RANGE_END();

    // 性能总结
    print_header("性能优化总结");
    std::vector<std::pair<std::string, std::string>> summary = {
        {"共享内存分块", "将全局内存读取次数降低 TILE_SIZEx 倍"},
        {"内存合并访问", "带宽提升可达 10 倍"},
        {"CUDA 流", "计算与数据传输重叠"},
        {"内核融合", "消除启动开销 + 内存往返"},
        {"占用率调优", "最佳线程块大小更好地隐藏延迟"},
        {"避免 Bank 冲突", "防止共享内存访问串行化"},
        {"寄存器压力管理", "更多活跃 warp = 更好的延迟隐藏"},
        {"NVTX 注解", "在 Nsight 时间线中识别瓶颈"},
    };
    print_performance_summary(summary);

    std::cout << "\n所有演示已完成。\n";
    return 0;
}
