#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Global cuBLAS handle
cublasHandle_t handle;

// Initialize cuBLAS
int init_cublas() {
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS initialization failed\n");
        return -1;
    }
    printf("cuBLAS initialized successfully\n");
    return 0;
}

// Cleanup cuBLAS
int cleanup_cublas() {
    cublasStatus_t status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS cleanup failed\n");
        return -1;
    }
    printf("cuBLAS cleaned up successfully\n");
    return 0;
}

// Matrix multiplication: C = A * B (single precision)
// m, n, k: dimensions (C is m x n, A is m x k, B is k x n)
int sgemm_wrapper(float *A, float *B, float *C, int m, int n, int k, int transa, int transb) {
    float alpha = 1.0f;
    float beta = 0.0f;

    // Move data to device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, m * k * sizeof(float));
    cudaMalloc((void **)&d_B, k * n * sizeof(float));
    cudaMalloc((void **)&d_C, m * n * sizeof(float));

    cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice);

    // Transpose options
    cublasOperation_t opA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;

    // Row-major to column-major conversion
    // In cuBLAS: C = α·op(B)·op(A) + β·C (note order is reversed)
    // We need to compute C = A * B

    // Adjust leading dimensions based on operations
    int lda = transa ? m : k;
    int ldb = transb ? k : n;
    int ldc = n;

    // Call cuBLAS
    cublasStatus_t status = cublasSgemm(handle, opB, opA, // Operations on B and A (note the order)
                                        n, m, k,          // Dimensions (n, m, k)
                                        &alpha,           // Alpha
                                        d_B, ldb,         // B matrix and leading dimension
                                        d_A, lda,         // A matrix and leading dimension
                                        &beta,            // Beta
                                        d_C, ldc          // C matrix and leading dimension
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS SGEMM failed with error code %d\n", status);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -1;
    }

    // Copy result back to host
    cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

// Create and initialize GPU memory
float *gpu_alloc(float *host_data, int size) {
    float *dev_ptr;
    cudaMalloc((void **)&dev_ptr, size * sizeof(float));
    if (host_data != NULL) {
        cudaMemcpy(dev_ptr, host_data, size * sizeof(float), cudaMemcpyHostToDevice);
    }
    return dev_ptr;
}

// Copy data from GPU to host
int gpu_to_host(float *dev_ptr, float *host_ptr, int size) {
    return cudaMemcpy(host_ptr, dev_ptr, size * sizeof(float), cudaMemcpyDeviceToHost);
}

// Free GPU memory
void gpu_free(float *dev_ptr) {
    cudaFree(dev_ptr);
}
