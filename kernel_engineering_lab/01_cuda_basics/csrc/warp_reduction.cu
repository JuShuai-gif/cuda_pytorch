#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdio>

// ============================================================================
// warp-level reduction：使用 __shfl_down_sync 的 warp shuffle 实现
// 工业背景：warp shuffle 是生产级 reduction 的核心技术
// 用于 LayerNorm / Softmax 内部的 reduction，避免 shared memory bank conflict
// 相比 naive reduction（shared memory），延迟更低、吞吐更高
// ============================================================================

// ---------------------------------------------------------------------------
// 1. warp_reduce_sum：在单个 warp 内完成 reduction，无需 shared memory
//    使用 __shfl_down_sync 进行 warp 内数据交换
//    每个 warp 内的 32 个线程共同完成一次 reduction
// ---------------------------------------------------------------------------

// 活跃线程掩码：0xffffffff 表示 warp 内所有 32 个线程都参与
// 在生产代码中，针对 Volta+ 架构，必须使用 __shfl_down_sync 而非老的 __shfl_down

__inline__ __device__ float warp_reduce_sum(float val) {
    // 在 warp 内进行 butterfly reduction：log2(32) = 5 步
    // 每一步使用 __shfl_down_sync 将数据向下传递并累加
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ---------------------------------------------------------------------------
// 2. block_reduce_with_warp_shuffle：block 级 reduction
//    流程：
//      a) 每个 warp 内先用 warp_reduce_sum 获得 warp 的部分和
//      b) warp 0 的 thread 0 收集各 warp 的部分和到 shared memory
//      c) warp 0 再在 shared memory 上完成最终 reduction
//
//    这样只需要 少量 shared memory，大部分工作在寄存器中完成
// ---------------------------------------------------------------------------

__global__ void warp_reduction_kernel(const float* input, float* output, int64_t n) {
    // shared memory 用于存储每个 warp 的部分和
    // warp 数量 = blockDim.x / 32，每个 warp 一个 float
    __shared__ float warp_sums[32];  // 最多 32 个 warp（1024 线程 / 32 = 32）

    int64_t tid = threadIdx.x;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;

    // 每个线程从 global memory 加载一个元素
    float val = (idx < n) ? input[idx] : 0.0f;

    // 第一步：warp 内 reduction
    val = warp_reduce_sum(val);

    // 每个 warp 的第一个线程将部分和写入 shared memory
    int warp_id = tid / 32;       // 当前线程所在的 warp 编号
    int lane_id = tid % 32;       // 当前线程在 warp 内的 lane 编号

    if (lane_id == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    // 第二步：warp 0 的线程从 shared memory 读取各 warp 部分和，
    // 完成最终的 block 内 reduction
    int num_warps = (blockDim.x + 31) / 32;
    val = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;

    if (warp_id == 0) {
        val = warp_reduce_sum(val);
        // 只有 warp 0 的 lane 0 写入最终结果
        if (lane_id == 0) {
            atomicAdd(output, val);
        }
    }
}

// ---------------------------------------------------------------------------
// 3. 纯 global memory 到 global memory 的 warp reduction
//    一次性完成整个 tensor 的 reduction，返回部分和
//    这个版本用于性能对比：展示 warp shuffle vs shared memory 的差异
// ---------------------------------------------------------------------------

__global__ void full_warp_reduction_kernel(const float* input, float* output, int64_t n) {
    // 每个 block 的 reduction 完全在 warp 内完成（使用 warp shuffle），
    // 然后每个 block 只写入一个值到 output，
    // 最后由 CPU 或另一个小 kernel 完成最终 reduction
    __shared__ float warp_partials[32];

    int64_t tid = threadIdx.x;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;

    float val = (idx < n) ? input[idx] : 0.0f;

    val = warp_reduce_sum(val);

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        warp_partials[warp_id] = val;
    }
    __syncthreads();

    int num_warps = (blockDim.x + 31) / 32;
    if (tid == 0) {
        float block_sum = 0.0f;
        for (int i = 0; i < num_warps; ++i) {
            block_sum += warp_partials[i];
        }
        output[blockIdx.x] = block_sum;
    }
}

// ---------------------------------------------------------------------------
// 4. 对比 kernel：使用 shared memory 的 naive reduction
//    （和 01_cuda_basics 中的实现相同，但增加错误检查）
// ---------------------------------------------------------------------------

__global__ void naive_reduce_kernel_warp(const float* input, float* output, int64_t n) {
    extern __shared__ float sdata[];

    int64_t tid = threadIdx.x;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for (int64_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// ============================================================================
// Host 端 launch 函数
// ============================================================================

torch::Tensor launch_warp_reduce_sum(const torch::Tensor& input) {
    TORCH_CHECK(input.device().is_cuda(), "输入必须在 CUDA 上");
    TORCH_CHECK(input.is_contiguous(), "输入必须是 contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "输入必须是 float32");

    int64_t n = input.numel();
    auto output = torch::zeros({1}, input.options());

    const int threads_per_block = 256;
    const int blocks_per_grid = std::min(
        static_cast<int>((n + threads_per_block - 1) / threads_per_block),
        1024
    );

    warp_reduction_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "warp_reduction_kernel 启动失败: %s\n", cudaGetErrorString(err));
    }

    return output;
}

torch::Tensor launch_full_warp_reduction(const torch::Tensor& input) {
    TORCH_CHECK(input.device().is_cuda(), "输入必须在 CUDA 上");
    TORCH_CHECK(input.is_contiguous(), "输入必须是 contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "输入必须是 float32");

    int64_t n = input.numel();
    const int threads_per_block = 256;
    const int blocks_per_grid = std::min(
        static_cast<int>((n + threads_per_block - 1) / threads_per_block),
        1024
    );

    auto partial_sums = torch::empty({blocks_per_grid}, input.options());

    full_warp_reduction_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        partial_sums.data_ptr<float>(),
        n
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "full_warp_reduction_kernel 启动失败: %s\n", cudaGetErrorString(err));
    }

    // 对 partial sums 再做一次 CPU 端求和（partical_sums 很小，CPU 更快）
    auto cpu_partial = partial_sums.to(torch::kCPU);
    float total = cpu_partial.sum().item<float>();

    auto final_output = torch::empty({1}, input.options());
    final_output[0] = total;
    return final_output;
}

torch::Tensor launch_naive_reduce_sum(const torch::Tensor& input) {
    TORCH_CHECK(input.device().is_cuda(), "输入必须在 CUDA 上");
    TORCH_CHECK(input.is_contiguous(), "输入必须是 contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "输入必须是 float32");

    int64_t n = input.numel();
    auto output = torch::zeros({1}, input.options());

    const int threads_per_block = 256;
    const int blocks_per_grid = std::min(
        static_cast<int>((n + threads_per_block - 1) / threads_per_block),
        1024
    );

    size_t shared_mem_size = threads_per_block * sizeof(float);

    naive_reduce_kernel_warp<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "naive_reduce_kernel 启动失败: %s\n", cudaGetErrorString(err));
    }

    return output;
}
