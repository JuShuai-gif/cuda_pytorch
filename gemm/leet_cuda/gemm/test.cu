/**
 * @file test.cu
 * @brief 朴素SGEMM正确性验证测试
 * 
 * 本文件是sgemm_naive.cu的简化版本，仅用于验证朴素实现的正确性。
 * 不包含性能测试部分，测试矩阵尺寸固定为512×512×512。
 */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>

/**
 * @brief 矩阵索引宏
 */
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

/**
 * @brief 向量化访存宏
 */
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

float testError(void);
float testPerformance(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat);

/**
 * @brief CPU参考实现
 */
void cpuSgemm(
    float *a, float *b, float *c, const int M, const int N, const int K) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}


/**
 * @brief 朴素SGEMM Kernel - 简化版本
 * 
 * 每个线程计算C矩阵的一个元素
 * 
 * @param a 输入矩阵A (M×K)
 * @param b 输入矩阵B (K×N)
 * @param c 输出矩阵C (M×N)
 * @param M 矩阵A行数，矩阵C行数
 * @param N 矩阵B列数，矩阵C列数
 * @param K 矩阵A列数，矩阵B行数
 */
__global__ void naiveSgemm(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && n < N) {
        float psum = 0.0;
        #pragma unroll
        for (int k = 0; k < K; k++) {
            psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
        }
        c[OFFSET(m, n, N)] = psum;
    }
}

int main(void) {
    float max_error = testError();
    printf("Max Error = %f\n", max_error);
    return 0;
}


/**
 * @brief 测试朴素实现的正确性
 * 
 * 测试参数:
 * - Block尺寸: 32×32 = 1024 threads/block
 * - 矩阵尺寸: M=512, N=512, K=512
 * - Grid尺寸: (512/32)×(512/32) = 16×16 blocks
 */
float testError(void) {
    const int BM = 32, BN = 32;
    const int M = 512, N = 512, K = 512;
    dim3 blockDim(BN, BM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_d_c;
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (float *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = rand() / float(RAND_MAX);
    for (int i = 0; i < K * N; i++)
        h_b[i] = rand() / float(RAND_MAX);
    cudaMemset(d_c, 15, size_c);

    cpuSgemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    naiveSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs(h_d_c[i] - h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_d_c);

    return max_error;
}

