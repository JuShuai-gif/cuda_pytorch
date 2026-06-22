#include <cuda_runtime.h>
#include <torch/extension.h>

/*
 * Element-wise vector addition kernel.
 * Each thread handles one element. Handles arbitrary vector sizes
 * (not just multiples of block size) via bounds checking.
 */

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}


torch::Tensor launch_vector_add_cuda(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "b must be float32");

    auto c = torch::empty_like(a);
    int64_t n = a.numel();

    const int threads_per_block = 256;
    const int blocks_per_grid = (static_cast<int>(n) + threads_per_block - 1) / threads_per_block;

    vector_add_kernel<<<blocks_per_grid, threads_per_block>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n
    );

    return c;
}
