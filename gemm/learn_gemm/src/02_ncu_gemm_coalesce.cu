#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include "gemm_kernels.cuh"
#include "utils.cuh"


#define LOAD_FLOAT4(value)  (reinterpret_cast<const float4*>(&(value))[0])
#define STORE_FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<const float4*>(&(value))[0])


template <const uint block_size>
__global__ void sgemm_global_mem_coalesce_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                                                 float alpha, const float *matrix_a,
                                                 const float *matrix_b, float beta, float *matrix_c)
{
    // 每个 thread 负责 1 个 row, 4 个连续 col
    const int threads_per_row = block_size / 4;  
    const int linear_tid = threadIdx.x;

    const int output_row = blockIdx.x * block_size + (linear_tid / threads_per_row);
    const int output_col = blockIdx.y * block_size + (linear_tid % threads_per_row) * 4;

    if (output_row >= num_rows_a || output_col >= num_cols_b) {
        return;
    }

    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);

    for (int k_idx = 0; k_idx < num_cols_a; ++k_idx) {
        float a_val = matrix_a[output_row * num_cols_a + k_idx];

        // B[k_idx, output_col:output_col+3]
        if (output_col + 3 < num_cols_b) {
            const float4 b_vec = LOAD_FLOAT4(matrix_b[k_idx * num_cols_b + output_col]);

            acc.x += a_val * b_vec.x;
            acc.y += a_val * b_vec.y;
            acc.z += a_val * b_vec.z;
            acc.w += a_val * b_vec.w;
        } else {
            // 处理尾部非4对齐
            for (int j = 0; j < 4; ++j) {
                if (output_col + j < num_cols_b) {
                    float b_val = matrix_b[k_idx * num_cols_b + output_col + j];
                    reinterpret_cast<float*>(&acc)[j] += a_val * b_val;
                }
            }
        }
    }

    const int c_idx = output_row * num_cols_b + output_col;

    if (output_col + 3 < num_cols_b) {
        float4 c_vec = STORE_FLOAT4(matrix_c[c_idx]);

        c_vec.x = alpha * acc.x + beta * c_vec.x;
        c_vec.y = alpha * acc.y + beta * c_vec.y;
        c_vec.z = alpha * acc.z + beta * c_vec.z;
        c_vec.w = alpha * acc.w + beta * c_vec.w;

        STORE_FLOAT4(matrix_c[c_idx]) = c_vec;
    } else {
        for (int j = 0; j < 4; ++j) {
            if (output_col + j < num_cols_b) {
                matrix_c[c_idx + j] =
                    alpha * reinterpret_cast<float*>(&acc)[j] +
                    beta * matrix_c[c_idx + j];
            }
        }
    }
}

void sgemm_global_mem_coalesce(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                               torch::Tensor &output_matrix, float alpha, float beta)
{
    // Validate inputs
    TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
    TORCH_CHECK(matrix_b.device().is_cuda(), "Matrix B must be on CUDA device");
    TORCH_CHECK(matrix_a.dtype() == torch::kFloat32, "Matrix A must be float32");
    TORCH_CHECK(matrix_b.dtype() == torch::kFloat32, "Matrix B must be float32");
    TORCH_CHECK(matrix_a.dim() == 2, "Matrix A must be 2D");
    TORCH_CHECK(matrix_b.dim() == 2, "Matrix B must be 2D");

    const int num_rows_a = static_cast<int>(matrix_a.size(0));
    const int num_cols_a = static_cast<int>(matrix_a.size(1));
    const int num_cols_b = static_cast<int>(matrix_b.size(1));

    TORCH_CHECK(matrix_b.size(0) == num_cols_a, "Matrix dimensions must match: A is MxK, B must be KxN");

    TORCH_CHECK(output_matrix.device().is_cuda(), "Matrix C must be on CUDA device");
    TORCH_CHECK(output_matrix.dtype() == torch::kFloat32, "Matrix C must be float32");
    TORCH_CHECK(output_matrix.size(0) == num_rows_a && output_matrix.size(1) == num_cols_b, "Matrix C must be MxN");

    // Get raw device pointers
    const float *d_matrix_a = matrix_a.data_ptr<float>();
    const float *d_matrix_b = matrix_b.data_ptr<float>();
    float *d_output_matrix = output_matrix.data_ptr<float>();

    // Configure kernel launch: 1D blocks with block_size^2 threads (32x32 = 1024 threads per block)
    constexpr uint block_size = 32;
    dim3 block_dim(block_size * block_size / 4);
    dim3 grid_dim(ceil_div(num_rows_a, block_size),
                  ceil_div(num_cols_b, block_size));
    
    sgemm_global_mem_coalesce_kernel<block_size><<<grid_dim, block_dim>>>(
        num_rows_a, num_cols_b, num_cols_a,
        alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);
}
