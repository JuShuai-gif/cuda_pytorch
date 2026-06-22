#include <cuda_runtime.h>

#include <iostream>
#include <vector>

struct JDSMatrix {
    int numRows;
    int numCols;
    int numTiles;
    int* colIdx;
    float* values;
    int* rowPerm;
    int* iterPtr;
};

__global__ void spmv_jds_kernel(JDSMatrix jdsMatrix, float* x, float* y) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= jdsMatrix.numRows) {
        return;
    }

    float sum = 0.0f;
    for (int t = 0; t < jdsMatrix.numTiles; ++t) {
        int i = jdsMatrix.iterPtr[t] + tid;
        if (i < jdsMatrix.iterPtr[t + 1]) {
            int col = jdsMatrix.colIdx[i];
            float value = jdsMatrix.values[i];
            sum += x[col] * value;
        }
    }
    y[jdsMatrix.rowPerm[tid]] = sum;
}

void spmv_jds(const JDSMatrix& jdsMatrix, float* d_x, float* d_y) {
    int blockSize = 256;
    int gridSize = (jdsMatrix.numRows + blockSize - 1) / blockSize;
    spmv_jds_kernel<<<gridSize, blockSize>>>(jdsMatrix, d_x, d_y);
}

int main() {
    const int numRows = 6;
    const int numCols = 6;

    std::vector<int> h_colIdx = {0, 0, 2, 0, 1, 3, 2, 4, 3, 4, 5};
    std::vector<float> h_values = {1, 5, 3, 7, 2, 8, 6, 4, 9, 10, 11};
    std::vector<int> h_rowPerm = {1, 3, 5, 2, 4, 0};
    std::vector<int> h_iterPtr = {0, 6, 11, 14, 15};
    std::vector<float> h_x = {1, 2, 3, 4, 5, 6};
    std::vector<float> h_y(numRows, 0);

    int* d_colIdx;
    float* d_values;
    int* d_rowPerm;
    int* d_iterPtr;
    float* d_x;
    float* d_y;

    cudaMalloc(&d_colIdx, h_colIdx.size() * sizeof(int));
    cudaMalloc(&d_values, h_values.size() * sizeof(float));
    cudaMalloc(&d_rowPerm, h_rowPerm.size() * sizeof(int));
    cudaMalloc(&d_iterPtr, h_iterPtr.size() * sizeof(int));
    cudaMalloc(&d_x, h_x.size() * sizeof(float));
    cudaMalloc(&d_y, h_y.size() * sizeof(float));

    cudaMemcpy(d_colIdx, h_colIdx.data(), h_colIdx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values.data(), h_values.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowPerm, h_rowPerm.data(), h_rowPerm.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_iterPtr, h_iterPtr.data(), h_iterPtr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), h_y.size() * sizeof(float), cudaMemcpyHostToDevice);

    int numTiles = h_iterPtr.size() - 1;
    JDSMatrix d_jdsMatrix = {numRows, numCols, numTiles, d_colIdx, d_values, d_rowPerm, d_iterPtr};

    spmv_jds(d_jdsMatrix, d_x, d_y);

    cudaMemcpy(h_y.data(), d_y, h_y.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Result y: ";
    for (float val : h_y) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_rowPerm);
    cudaFree(d_iterPtr);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
