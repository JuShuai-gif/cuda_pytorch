#include <cuda_runtime.h>
#include <stdio.h>

__global__ void computeHistogram(int nnz, int* rowIdx, int* rowPtrs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) {
        atomicAdd(&rowPtrs[rowIdx[i] + 1], 1);
    }
}

__global__ void exclusiveScan(int* rowPtrs, int numRows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int offset = 1; offset < numRows + 1; offset *= 2) {
        int temp = 0;
        if (i >= offset) {
            temp = rowPtrs[i - offset];
        }
        __syncthreads();

        if (i < numRows + 1) {
            rowPtrs[i] += temp;
        }
        __syncthreads();
    }
}

void cooToCsr(int nnz, int numRows, int* h_rowIdx, int* h_colIdx, float* h_values, int** d_csrRowPtrs,
              int** d_csrColIdx, float** d_csrValues) {
    int *d_rowIdx, *d_colIdx;
    float* d_values;

    cudaMalloc(&d_rowIdx, nnz * sizeof(int));
    cudaMalloc(&d_colIdx, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(float));

    cudaMalloc(d_csrRowPtrs, (numRows + 1) * sizeof(int));
    cudaMalloc(d_csrColIdx, nnz * sizeof(int));
    cudaMalloc(d_csrValues, nnz * sizeof(float));

    cudaMemcpy(d_rowIdx, h_rowIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, h_colIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(*d_csrRowPtrs, 0, (numRows + 1) * sizeof(int));

    int blockSize = 256;
    int gridSize = (nnz + blockSize - 1) / blockSize;

    computeHistogram<<<gridSize, blockSize>>>(nnz, d_rowIdx, *d_csrRowPtrs);
    cudaDeviceSynchronize();

    exclusiveScan<<<1, numRows + 1>>>(*d_csrRowPtrs, numRows);
    cudaDeviceSynchronize();

    cudaMemcpy(*d_csrColIdx, d_colIdx, nnz * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(*d_csrValues, d_values, nnz * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaFree(d_rowIdx);
    cudaFree(d_colIdx);
    cudaFree(d_values);
}

int main() {
    int numRows = 4;
    int nnz = 8;

    int h_rowIdx[] = {0, 0, 1, 1, 1, 2, 2, 3};
    int h_colIdx[] = {0, 1, 0, 2, 3, 1, 2, 3};
    float h_values[] = {1, 7, 5, 3, 9, 2, 8, 6};

    int *d_csrRowPtrs, *d_csrColIdx;
    float* d_csrValues;

    cooToCsr(nnz, numRows, h_rowIdx, h_colIdx, h_values, &d_csrRowPtrs, &d_csrColIdx, &d_csrValues);

    int* h_csrRowPtrs = (int*)malloc((numRows + 1) * sizeof(int));
    int* h_csrColIdx = (int*)malloc(nnz * sizeof(int));
    float* h_csrValues = (float*)malloc(nnz * sizeof(float));

    cudaMemcpy(h_csrRowPtrs, d_csrRowPtrs, (numRows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_csrColIdx, d_csrColIdx, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_csrValues, d_csrValues, nnz * sizeof(float), cudaMemcpyDeviceToHost);

    printf("COO h_rowIdx:\n");
    for (int i = 0; i < nnz; i++) {
        printf("%d ", h_rowIdx[i]);
    }

    printf("\nCOO h_colIdx:\n");
    for (int i = 0; i < nnz; i++) {
        printf("%d ", h_colIdx[i]);
    }

    printf("\nCOO h_values:\n");
    for (int i = 0; i < nnz; i++) {
        printf("%.2f ", h_values[i]);
    }

    printf("\n \n");

    printf("CSR row pointers:\n");
    for (int i = 0; i <= numRows; i++) {
        printf("%d ", h_csrRowPtrs[i]);
    }
    printf("\n");

    printf("CSR column indices:\n");
    for (int i = 0; i < nnz; i++) {
        printf("%d ", h_csrColIdx[i]);
    }
    printf("\n");

    printf("CSR values:\n");
    for (int i = 0; i < nnz; i++) {
        printf("%.2f ", h_csrValues[i]);
    }
    printf("\n");

    cudaFree(d_csrRowPtrs);
    cudaFree(d_csrColIdx);
    cudaFree(d_csrValues);
    free(h_csrRowPtrs);
    free(h_csrColIdx);
    free(h_csrValues);

    return 0;
}
