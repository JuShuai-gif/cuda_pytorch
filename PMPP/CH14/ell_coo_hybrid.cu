#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <vector>

struct ELLMatrix {
    int numRows;
    int numCols;
    int maxNonZerosPerRow;
    int* colIdx;
    float* values;
};

struct COOMatrix {
    int numRows;
    int numCols;
    int numNonZeros;
    // we can use vector cause we do this on CPU
    std::vector<int> rowIdx;
    std::vector<int> colIdx;
    std::vector<float> values;
};

struct HybridMatrix {
    ELLMatrix ellPart;
    COOMatrix cooPart;
};

__global__ void spmv_ell_kernel(ELLMatrix ellMatrix, float* x, float* y) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < ellMatrix.numRows) {
        float sum = 0.0f;
        for (unsigned int t = 0; t < ellMatrix.maxNonZerosPerRow; t++) {
            unsigned int i = t * ellMatrix.numRows + row;
            int col = ellMatrix.colIdx[i];
            float value = ellMatrix.values[i];
            if (col >= 0) {
                sum += x[col] * value;
            }
        }
        y[row] = sum;
    }
}

// host spmv coo
void spmv_coo(const COOMatrix& cooMatrix, const std::vector<float>& x, std::vector<float>* y) {
    for (int i = 0; i < cooMatrix.numNonZeros; i++) {
        (*y)[cooMatrix.rowIdx[i]] += cooMatrix.values[i] * x[cooMatrix.colIdx[i]];
    }
}

HybridMatrix convertToHybrid(const std::vector<std::vector<std::pair<int, float>>>& matrix, int maxEllNonZeros) {
    int numRows = matrix.size();
    int numCols = 0;
    for (const auto& row : matrix) {
        for (const auto& elem : row) {
            numCols = std::max(numCols, elem.first + 1);
        }
    }

    // init matrices
    HybridMatrix hybridMatrix;
    hybridMatrix.ellPart.numRows = numRows;
    hybridMatrix.ellPart.numCols = numCols;
    hybridMatrix.ellPart.maxNonZerosPerRow = maxEllNonZeros;

    hybridMatrix.cooPart.numRows = numRows;
    hybridMatrix.cooPart.numCols = numCols;
    hybridMatrix.cooPart.numNonZeros = 0;

    std::vector<int> ellColIdx(numRows * maxEllNonZeros, -1);
    std::vector<float> ellValues(numRows * maxEllNonZeros, 0.0f);

    for (int i = 0; i < numRows; ++i) {
        int ellCount = 0;
        for (const auto& elem : matrix[i]) {
            if (ellCount < maxEllNonZeros) {  // add to Ell part
                ellColIdx[ellCount * numRows + i] = elem.first;
                ellValues[ellCount * numRows + i] = elem.second;
                ellCount++;
            } else {  // add to COO part
                hybridMatrix.cooPart.rowIdx.push_back(i);
                hybridMatrix.cooPart.colIdx.push_back(elem.first);
                hybridMatrix.cooPart.values.push_back(elem.second);
                hybridMatrix.cooPart.numNonZeros++;
            }
        }
    }

    // put ell part on device
    cudaMalloc(&hybridMatrix.ellPart.colIdx, ellColIdx.size() * sizeof(int));
    cudaMalloc(&hybridMatrix.ellPart.values, ellValues.size() * sizeof(float));
    cudaMemcpy(hybridMatrix.ellPart.colIdx, ellColIdx.data(), ellColIdx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(hybridMatrix.ellPart.values, ellValues.data(), ellValues.size() * sizeof(float), cudaMemcpyHostToDevice);

    return hybridMatrix;
}

void spmv_hybrid(const HybridMatrix& hybridMatrix, float* d_x, std::vector<float>* h_y) {
    int blockSize = 256;
    int gridSize = (hybridMatrix.ellPart.numRows + blockSize - 1) / blockSize;

    // allocate memory for y
    float* d_y;
    cudaMalloc(&d_y, h_y->size() * sizeof(float));
    cudaMemset(d_y, 0, h_y->size() * sizeof(float));

    spmv_ell_kernel<<<gridSize, blockSize>>>(hybridMatrix.ellPart, d_x, d_y);

    // copy partial results back to host
    cudaMemcpy(h_y->data(), d_y, h_y->size() * sizeof(float), cudaMemcpyDeviceToHost);

    // put x in the host memory
    std::vector<float> h_x(hybridMatrix.cooPart.numCols);
    cudaMemcpy(h_x.data(), d_x, h_x.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // and run the coo matmul on the remaining part
    spmv_coo(hybridMatrix.cooPart, h_x, h_y);

    cudaFree(d_y);
}

int main() {
    std::vector<std::vector<std::pair<int, float>>> matrix = {
        {{0, 1.0f}, {1, 7.0f}},             // Row 0
        {{0, 5.0f}, {2, 3.0f}, {3, 9.0f}},  // Row 1
        {{1, 2.0f}, {2, 8.0f}},             // Row 2
        {{3, 6.0f}}                         // Row 3
    };

    HybridMatrix hybrid = convertToHybrid(matrix, 2);

    std::vector<float> h_x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> h_y(matrix.size(), 0.0f);

    float* d_x;
    cudaMalloc(&d_x, h_x.size() * sizeof(float));
    cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice);

    spmv_hybrid(hybrid, d_x, &h_y);

    std::cout << "Result y: ";
    for (float val : h_y) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    cudaFree(hybrid.ellPart.colIdx);
    cudaFree(hybrid.ellPart.values);
    cudaFree(d_x);

    return 0;
}
