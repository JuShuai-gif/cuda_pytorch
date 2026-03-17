#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024
#define BLOCK_SIZE 16

void gemm_cpu(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

__global__ void gemm_naive_gpu(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

void init_matrix(float* mat, int n) {
    for (int i = 0; i < n * n; i++) {
        mat[i] = (float)(rand() % 100) / 10.0f;
    }
}

void print_matrix(float* mat, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", mat[i * n + j]);
        }
        printf("\n");
    }
}

int main() {
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C_cpu = (float*)malloc(size);
    h_C_gpu = (float*)malloc(size);

    srand(42);
    init_matrix(h_A, N);
    init_matrix(h_B, N);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    clock_t start_cpu = clock();
    gemm_cpu(h_A, h_B, h_C_cpu, N);
    clock_t end_cpu = clock();
    double cpu_time = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC;

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, end;
    float gpu_time = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    gemm_naive_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&gpu_time, start, end);

    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    printf("CPU GEMM Time: %.4f seconds\n", cpu_time);
    printf("GPU GEMM Time: %.4f ms\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time * 1000 / gpu_time);

    float max_diff = 0.0f;
    for (int i = 0; i < N * N; i++) {
        float diff = fabsf(h_C_cpu[i] - h_C_gpu[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Max difference between CPU and GPU: %.6f\n", max_diff);

    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return 0;
}
