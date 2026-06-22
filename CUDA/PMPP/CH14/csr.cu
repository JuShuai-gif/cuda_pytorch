#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>

struct CSRMatrix {
    int numRows;
    int* rowPtrs;
    int* colIdx;
    float* value;
};

__global__ void spmv_csr_kernel(CSRMatrix csrMatrix, float* x, float* y) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < csrMatrix.numRows) {
        float sum = 0.0f;
        for (unsigned int i = csrMatrix.rowPtrs[row]; i < csrMatrix.rowPtrs[row + 1]; ++i) {
            unsigned int col = csrMatrix.colIdx[i];
            float value = csrMatrix.value[i];
            sum += x[col] * value;
        }
        y[row] = sum;
    }
}

int main() {
    int numRows = 4;

    int h_rowPtrs[] = {0, 2, 5, 7, 8};
    int h_colIdx[] = {0, 1, 0, 2, 3, 1, 2, 3};
    float h_values[] = {1, 7, 5, 3, 9, 2, 8, 6};
    float h_x[] = {1, 2, 3, 4};
    float h_y[4] = {0};  // init as zeros

    int *d_rowPtrs, *d_colIdx;
    float *d_values, *d_x, *d_y;

    cudaMalloc((void**)&d_rowPtrs, (numRows + 1) * sizeof(int));
    cudaMalloc((void**)&d_colIdx, 8 * sizeof(int));
    cudaMalloc((void**)&d_values, 8 * sizeof(float));
    cudaMalloc((void**)&d_x, 4 * sizeof(float));
    cudaMalloc((void**)&d_y, 4 * sizeof(float));

    cudaMemcpy(d_rowPtrs, h_rowPtrs, (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, h_colIdx, 8 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, 8 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, 4 * sizeof(float), cudaMemcpyHostToDevice);

    CSRMatrix d_csrMatrix;
    d_csrMatrix.numRows = numRows;
    d_csrMatrix.rowPtrs = d_rowPtrs;
    d_csrMatrix.colIdx = d_colIdx;
    d_csrMatrix.value = d_values;

    spmv_csr_kernel<<<1, numRows>>>(d_csrMatrix, d_x, d_y);
    cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Result vector y:\n");
    for (int i = 0; i < numRows; i++) {
        printf("%.2f ", h_y[i]);
    }
    printf("\n");

    cudaFree(d_rowPtrs);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
