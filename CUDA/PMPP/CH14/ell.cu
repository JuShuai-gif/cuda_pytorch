#include <cuda_runtime.h>

#include <iostream>
#include <vector>

struct ELLMatrix {
    int numRows;
    int numCols;
    int maxNonZerosPerRow;
    int* colIdx;
    float* values;
};

__global__ void spmv_ell_kernel(ELLMatrix ellMatrix, float* x, float* y) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < ellMatrix.numRows) {
        float sum = 0.0f;
        for (unsigned int t = 0; t < ellMatrix.maxNonZerosPerRow; ++t) {
            unsigned int i = t * ellMatrix.numRows + row;
            int col = ellMatrix.colIdx[i];
            float value = ellMatrix.values[i];
            if (col >= 0) {  // Handle padding
                sum += x[col] * value;
            }
        }
        y[row] = sum;
    }
}

void spmv_ell(const ELLMatrix& ellMatrix, float* d_x, float* d_y) {
    int blockSize = 256;
    int gridSize = (ellMatrix.numRows + blockSize - 1) / blockSize;
    spmv_ell_kernel<<<gridSize, blockSize>>>(ellMatrix, d_x, d_y);
}

int main() {
    const int numRows = 4;
    const int numCols = 4;
    const int maxNonZerosPerRow = 3;

    // -1 is padding
    std::vector<int> h_colIdx = {0, 1, -1, 0, 2, 3, 1, 2, -1, 3, -1, -1};
    std::vector<float> h_values = {1, 7, 0, 5, 3, 9, 2, 8, 0, 6, 0, 0};
    std::vector<float> h_x = {1, 2, 3, 4};
    std::vector<float> h_y(numRows, 0);

    int* d_colIdx;
    float* d_values;
    float* d_x;
    float* d_y;

    cudaMalloc(&d_colIdx, h_colIdx.size() * sizeof(int));
    cudaMalloc(&d_values, h_values.size() * sizeof(float));
    cudaMalloc(&d_x, h_x.size() * sizeof(float));
    cudaMalloc(&d_y, h_y.size() * sizeof(float));

    cudaMemcpy(d_colIdx, h_colIdx.data(), h_colIdx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values.data(), h_values.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice);

    ELLMatrix d_ellMatrix = {numRows, numCols, maxNonZerosPerRow, d_colIdx, d_values};

    // Call the kernel function
    spmv_ell(d_ellMatrix, d_x, d_y);

    // Copy result back to host
    cudaMemcpy(h_y.data(), d_y, h_y.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Result y: ";
    for (float val : h_y) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
