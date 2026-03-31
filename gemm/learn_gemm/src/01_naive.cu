#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include "gemm_kernels.cuh"
#include "utils.cuh"

// 朴素实现方式
template <const uint block_size>
__global__ void sgemm_naive_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                                   float alpha, const float *matrix_a,
                                   const float *matrix_b, float beta, float *output_matrix)
{
    // Map 1D thread ID to 2D output position
    // threadIdx.x % block_size 行号
    // threadIdx.x / block_size 列号
    const int output_row = blockIdx.x * block_size + (threadIdx.x % block_size);// 行
    const int output_col = blockIdx.y * block_size + (threadIdx.x / block_size);// 列

    // Boundary check for non-multiple of block size
    if (output_row < num_rows_a && output_col < num_cols_b)
    {
        float accumulator = 0.0f;
        for (int k_idx = 0; k_idx < num_cols_a; ++k_idx)
        {
            accumulator += matrix_a[output_row * num_cols_a + k_idx] *
                           matrix_b[k_idx * num_cols_b + output_col];
        }
        // C = α*(A@B)+β*C
        const int output_idx = output_row * num_cols_b + output_col;
        output_matrix[output_idx] = alpha * accumulator + beta * output_matrix[output_idx];
    }
}

__global__ void warmupKernel() {
  __shared__ int s[100];
  s[0] += s[1];
}

void warmup_gpu()
{
    // Launch a simple kernel to warm up the GPU
    warmupKernel<<<1, 32>>>();
    cudaDeviceSynchronize();
}

// F16的朴素版本
__global__ void hgemm_naive_f16_kernel(half* a,half* b,half* c,int M,int N,int K){
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;

    if (n < N && m < M)
    {
        half psum = 0.0;
        #pragma unroll
        for (size_t i = 0; i < K; ++i)
        {
            psum += a[m*K+k] * b[k * N + n];
        }
        c[m*N + n] = psum; // c[m,n]
    }
}

// F16 沿K方向切分版本
/*
// HGEMM: Block Tile + K Tile, with smem
// Block Tile (BM, BN) + K Tile (BK=32)
// grid((N + BN - 1) / BN, (M + BM - 1) / BM), block(BN, BM)
// a: MxK, b: KxN, c: MxN, compute: c = a * b, all row major  
*/
template<const int BM = 32,const int BN = 32,const int BK = 32>
__global__ void hgemm_sliced_k_f16_kernel(half* a,half*b,half* c,int M,int N,int K){
    __shared__ half s_a[BM][BK],s_b[BK][BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // 计算的是块内索引
    int load_smem_a_m = tid / 32;// 行
    int load_smem_a_k = tid % 32;// 列

    int load_smem_b_k = tid / 32;// 行
    int load_smem_b_n = tid % 32;// 列

    // 根据块索引计算全局索引
    int load_gmem_a_m = by * BM + load_gmem_a_m;    // 全局行
    int load_gmem_b_n = bx * BN + load_smem_b_n;    // 全局列

    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    half sum = __float2half(0.f);

    // 在 k 的方向上进行循环
    for (int bk = 0; bk < (K + BK - 1)/BK; bk++)
    {
        int load_gmem_a_k = bk * BK + load_gmem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;        

        // 加载全局内存到共享内存
        s_a[load_smem_a_m][load_smem_a_k] = a[load_gmem_a_addr];    
        
        // 和上面的一样
        int load_gmem_b_k = bk * BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        // 加载全局内存到共享内存
        s_b[load_smem_b_k][load_smem_b_n] = b[load_gmem_b_addr];   
        
        __syncthreads();
        
        // 相当于计算朴素方式
        #pragma unroll
        for (int k = 0; k < BK; k++){
            int comp_smem_a_m = load_smem_a_m;
            int comp_smem_b_n = load_smem_b_n;
            sum += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
        }
        __syncthreads();
    }

    int store_gmem_c_m = load_gmem_a_m;
    int store_gmem_c_n = load_gmem_b_n;
    int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
    c[store_gmem_c_addr] = sum;    
}



void sgemm_naive(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
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

    const float *d_matrix_a = matrix_a.data_ptr<float>();
    const float *d_matrix_b = matrix_b.data_ptr<float>();
    float *d_output_matrix = output_matrix.data_ptr<float>();

    // Configure kernel launch: 1D blocks with block_size^2 threads (32x32 = 1024 threads per block)
    constexpr uint block_size = 32;
    dim3 block_dim(block_size * block_size);
    dim3 grid_dim(ceil_div(num_rows_a, block_size),
                  ceil_div(num_cols_b, block_size));

    // Launch kernel
    sgemm_naive_kernel<block_size><<<grid_dim, block_dim>>>(
        num_rows_a, num_cols_b, num_cols_a,
        alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);
}