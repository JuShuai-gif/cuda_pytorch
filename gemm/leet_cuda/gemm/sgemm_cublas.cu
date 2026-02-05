
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
// 一次读 4 个float
#define FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

float testCublasError(const int M, const int N, const int K);
float testCublasPerformance(const int M, const int N, const int K, const int repeat);

// 朴素矩阵乘法
void cpuSgemm(float *a, float *b, float *c, const int M, const int N, const int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0f;
            for (int k = 0; k < K; k++) {
                psum += a[OFFSET(m, k, k)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}


int main(void) {
    printf("\nKernal = cublas\n");
    const int outer_repeat = 10, inner_repeat = 1;
    {
        const int M = 512, N = 512, K = 512;
        float max_error = testCublasError(M, N, K);
        printf("Max Error = %f\n", max_error);
    }

    // 矩阵长度序列
    const int M_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int N_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[15] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};

    // 
    const int TESTNUM = 15;
    for (int i = 0; i < TESTNUM; i++) {
        const int M = M_list[i], N = N_list[i], K = K_list[i];

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int j = 0; j < outer_repeat; j++) {
            double this_sec = testCublasPerformance(M, N, K, inner_repeat);
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

float testCublasError(const int M, const int N, const int K) {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_d_c;

    // 申请空间
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);
    h_d_c = (float *)malloc(size_c);

    // 随机
    srand(time(0));

    for (int i = 0; i < M * K; i++) {
        h_a[i] = rand() / float(RAND_MAX);
    }

    for (int i = 0; i < K * N; i++) {
        h_b[i] = rand() / float(RAND_MAX);
    }

    cpuSgemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    float cublas_alpha = 1.0;
    float cublas_beta = 0;

    cublasSgemm(cublas_handle,  // cuBLAS库的句柄，表示当前使用的cuBLAS上下文 
        CUBLAS_OP_N, // 表示是否对矩阵 A 和 B 进行转置(Transpose),N表示无转置
        CUBLAS_OP_N, // 无转置
        N, // 矩阵 C 的行数
        M, // 矩阵 C 的列数
        K, // 矩阵 A 的列数和矩阵 B 的行数，必须匹配
        &cublas_alpha, // 乘法运算中的标量系数
        d_b,    // 指向矩阵 B 数据的指针
        N,      // 矩阵 A 的列数
        d_a,    // 指向矩阵 A 数据的指针
        K,      // 矩阵 A 的列数
        &cublas_beta, // 另一个标量系数
        d_c,    // 指向矩阵 C 的数据指针
        N);     // 矩阵 C 的列数

    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs(h_d_c[i] - h_c[i]);
        if (max_error != max_error || this_error != this_error) {
            max_error = -NAN;
        } else {
            max_error = max(max_error, this_error);
        }
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

float testCublasPerformance(const int M, const int N, const int K, const int repeat) {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    float cublas_alpha = 1.0;
    float cublas_beta = 0;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    for (int i = 0; i < repeat; i++) {
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &cublas_alpha, d_b, N, d_a, K, &cublas_beta, d_c, N);
    }

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
