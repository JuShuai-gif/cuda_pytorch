#include <cuda_runtime.h>
#include <torch/extension.h>

/*
 * Naive sum reduction kernel.
 *
 * Phase 1 (block-level): Each block loads elements into shared memory,
 * performs sequential reduction, and atomically adds its partial sum
 * to the global output.
 *
 * Limitation: atomicAdd on the same global address can cause contention
 * when many blocks are launched (e.g. with very large arrays).
 * This is the "naive" approach -- later modules will show better patterns.
 */

__global__ void naive_reduce_kernel(const float* input, float* output, int64_t n) {
    extern __shared__ float sdata[];

    int64_t tid = threadIdx.x;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;

    // Load into shared memory (coalesced)
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // Block-level reduction via sequential addressing
    for (int64_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 of each block writes partial sum to global output
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}


torch::Tensor launch_reduce_sum_cuda(const torch::Tensor& input) {
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    int64_t n = input.numel();
    auto output = torch::zeros({1}, input.options());

    const int threads_per_block = 256;
    const int blocks_per_grid = std::min(
        static_cast<int>((n + threads_per_block - 1) / threads_per_block),
        1024  // Cap blocks to limit atomicAdd contention
    );

    size_t shared_mem_size = threads_per_block * sizeof(float);

    naive_reduce_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output;
}
