#include "gemm_kernels.cuh"
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <iostream>



/*
32, 32, 32
128, 128, 128
512, 512, 512
1024, 1024, 1024
2048, 2048, 2048
4096, 4096, 4096
// 非方形矩阵
1024, 512, 2048
2048, 1024, 512
512, 2048, 1024
*/

int M = 1024,N = 1024,K=1024;

int main() {
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(torch::kCUDA);

    auto A = torch::randn({M, K}, options);
    auto B = torch::randn({K, N}, options);
    auto C_naive = torch::zeros({M, N}, options);
    auto C_ref   = torch::zeros({M, N}, options);

    // 先做一次 reference
    // C_ref = torch::matmul(A, B);

    // warmup，避免初始化污染 profiling
    for (int i = 0; i < 2; ++i) {
        sgemm_global_mem_coalesce(A, B, C_naive, 1.0f, 0.0f);
    }
    cudaDeviceSynchronize();

    // 正式跑几次，方便 ncu 抓稳定 kernel
    for (int i = 0; i < 10; ++i) {
        sgemm_global_mem_coalesce(A, B, C_naive, 1.0f, 0.0f);
    }
    cudaDeviceSynchronize();

    // 简单正确性检查
    auto max_diff = (C_ref - C_naive).abs().max().item<float>();
    std::cout << "max diff = " << max_diff << std::endl;

    return 0;
}