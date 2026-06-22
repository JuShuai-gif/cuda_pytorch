#include "matmul_tiled.cuh"

__global__ void matmul_tiled(const float *__restrict__ A,
                              const float *__restrict__ B,
                              float *__restrict__ C,
                              int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;

        As[ty][tx] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void matmul_optimized(const float *__restrict__ A,
                                  const float *__restrict__ B,
                                  float *__restrict__ C,
                                  int N) {
    __shared__ float As[TILE_SIZE_OPT][TILE_SIZE_OPT + SMEM_PAD];
    __shared__ float Bs[TILE_SIZE_OPT][TILE_SIZE_OPT + SMEM_PAD];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE_OPT + ty;
    int col = bx * TILE_SIZE_OPT + tx;

    float sum = 0.0f;
    int num_tiles = (N + TILE_SIZE_OPT - 1) / TILE_SIZE_OPT;

    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE_SIZE_OPT + tx;
        int b_row = t * TILE_SIZE_OPT + ty;

        As[ty][tx] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        // 手动展开内层循环以获得更好的指令级并行
        #pragma unroll
        for (int k = 0; k < TILE_SIZE_OPT; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
