/**
 * @file sgemm_naive.cu
 * @brief 朴素SGEMM实现 - 最基本的CUDA矩阵乘法实现
 * 
 * 本文件展示最简单的GPU矩阵乘法实现：
 * - 每个线程负责计算输出矩阵C的一个元素
 * - 直接从全局内存读取数据，无任何优化
 * - 作为性能对比的基准（最原始的实现）
 * 
 * 性能特点：通常只有理论峰值的5-10%，因为：
 * 1. 大量重复访问全局内存
 * 2. 内存访问延迟高
 * 3. 无法利用共享内存和寄存器
 */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>

/**
 * @brief 矩阵索引宏
 */
#define OFFSET(row,col,ld) ((row) * (ld) + (col))

/**
 * @brief 向量化访存宏 - 一次操作4个float
 */
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

float testError(void);

float testPerformance(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat);

/**
 * @brief CPU参考实现 - 用于验证结果正确性
 */
void cpuSgemm(float*a,float*b,float*c,const int M,const int N,const int K){
    for (int m = 0; m<M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0;
            for (int k = 0; k<K; k++) {
                psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}


/**
 * @brief 朴素SGEMM kernel - 每个线程计算C矩阵的一个元素
 * 
 * 工作原理：
 * 1. 通过blockIdx和threadIdx计算全局线程坐标(n, m)
 * 2. 检查边界确保不越界
 * 3. 对K维度进行循环累加计算点积
 * 4. 将结果写入C矩阵
 * 
 * 特点：
 * - 简单直接，易于理解
 * - 性能较差，存在大量全局内存访问
 * - 每个线程独立计算，无协作
 * 
 * @param a 输入矩阵A (M x K，行主序)
 * @param b 输入矩阵B (K x N，行主序)
 * @param c 输出矩阵C (M x N，行主序)
 * @param M 矩阵A行数
 * @param N 矩阵B列数
 * @param K 矩阵A列数/矩阵B行数
 */
__global__ void naiveSgemm(float* __restrict__ a,float* __restrict__ b,float* __restrict__ c,const int M,const int N,const int K){
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    int m = blockDim.y * blockIdx.y + threadIdx.y;
    if (m < M && n < N) {
        float psum = 0.0;
        #pragma unroll
        for (int k = 0; k < K; k++) {
            psum += a[OFFSET(m, k, K)] + b[OFFSET(k, n, N)];
        }
        c[OFFSET(m, n, N)] = psum;
    }
}

int main(void) {
    float max_error = testError();
    printf("Max Error = %f\n", max_error);

    printf("\nKernal = naiveSgemm\n");
    const int M_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int N_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[15] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};
    
    const int outer_repeat = 10, inner_repeat = 1;
    const int BM = 32, BN = 32;
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int) = naiveSgemm;
    const int TESTNUM = 15;

    for (int i = 0; i < TESTNUM; i++) {
        const int M = M_list[i], N = N_list[i], K = K_list[i];

        dim3 blockDim(BN, BM);
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int j = 0; j < outer_repeat; j++) {
            double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }

        double avg_sec = total_sec / outer_repeat;
        double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

        printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
    }
    return 0;
}


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


float testPerformance(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
        gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return sec;
}



















