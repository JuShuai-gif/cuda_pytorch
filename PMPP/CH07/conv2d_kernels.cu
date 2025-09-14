#define DEFINE_CONSTANT_MEMORY
#include <stdio.h>

#include "conv2d_kernels.cuh"

__global__ void conv2d_kernel(float* M, float* F, float* P, int r, int height, int width) {
    int outRow = blockDim.y * blockIdx.y + threadIdx.y;
    int outCol = blockDim.x * blockIdx.x + threadIdx.x;

    float Pvalue = 0.0f;
    for (int Frow = 0; Frow < r * 2 + 1; Frow++) {
        for (int Fcol = 0; Fcol < r * 2 + 1; Fcol++) {
            int inRow = outRow - r + Frow;
            int inCol = outCol - r + Fcol;
            if ((inRow >= 0) && (inRow < height) && (inCol >= 0) && (inCol < width)) {
                Pvalue += M[inRow * width + inCol] * F[Frow * (2 * r + 1) + Fcol];
            }
        }
    }

    if ((outRow < height) && (outCol < width)) {
        P[outRow * width + outCol] = Pvalue;
    }
}

cudaError_t initConstFilter(const float* filter, int r) {
    return cudaMemcpyToSymbol(constFilter, filter, (2 * r + 1) * (2 * r + 1) * sizeof(float));
}

__global__ void conv2d_kernel_with_constant_memory(float* M, float* P, int r, int height, int width) {
    int outRow = blockDim.y * blockIdx.y + threadIdx.y;
    int outCol = blockDim.x * blockIdx.x + threadIdx.x;

    if (outRow < height && outCol < width) {
        float sum = 0.0f;
        for (int i = -r; i <= r; i++) {
            for (int j = -r; j <= r; j++) {
                int inRow = outRow + i;
                int inCol = outCol + j;
                if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                    int filterIndex = (i + r) * (2 * r + 1) + (j + r);
                    sum += M[inRow * width + inCol] * constFilter[filterIndex];
                }
            }
        }
        P[outRow * width + outCol] = sum;
    }
}

__global__ void tiled_convolution_kernel(float* M, float* P, int r, int height, int width) {
    int row = blockIdx.y * OUT_TILE_SIZE + threadIdx.y - FILTER_RADIUS;
    int col = blockIdx.x * OUT_TILE_SIZE + threadIdx.x - FILTER_RADIUS;

    __shared__ float M_s[IN_TILE_SIZE][IN_TILE_SIZE];

    if (row >= 0 && row < height && col >= 0 && col < width) {
        M_s[threadIdx.y][threadIdx.x] = M[row * width + col];
    } else {
        M_s[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();

    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;
    int filterIndex;

    if (row >= 0 && row < height && col >= 0 && col < width) {
        if (tileRow >= 0 && tileRow < OUT_TILE_SIZE && tileCol >= 0 && tileCol < OUT_TILE_SIZE) {
            float Pvalue = 0.0f;

            for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
                for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
                    filterIndex = fRow * (2 * FILTER_RADIUS + 1) + fCol;
                    Pvalue += M_s[threadIdx.y + fRow - FILTER_RADIUS][threadIdx.x + fCol - FILTER_RADIUS] *
                              constFilter[filterIndex];
                }
            }
            P[row * width + col] = Pvalue;
        }
    }
}

__global__ void tiled_convolution_kernel_with_l2_caching(float* M, float* P, int r, int height, int width) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ float M_s[TILE_SIZE][TILE_SIZE];

    if (row < height && col < width) {
        M_s[threadIdx.y][threadIdx.x] = M[row * width + col];
    } else {
        M_s[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();

    int filterIndex;

    if (row < height && col < width) {
        float Pvalue = 0.0f;
        for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
            for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
                // Calculate shared memory indices
                filterIndex = fRow * (2 * FILTER_RADIUS + 1) + fCol;
                int sy = threadIdx.y - FILTER_RADIUS + fRow;
                int sx = threadIdx.x - FILTER_RADIUS + fCol;

                // first the ones that we already put into the shared memory
                // check if it is within the tile that we are interested about
                if (sy >= 0 && sy < TILE_SIZE && sx >= 0 && sx < TILE_SIZE) {
                    Pvalue += M_s[sy][sx] * constFilter[filterIndex];
                } else {
                    // than the ones that (we hope) already are in the l2 cache cause the the access of neighbouring
                    // threads but at first check if the values actually are within the matrix (the edge cases)

                    // Global memory access for elements outside shared memory tile
                    int gy = row - FILTER_RADIUS + fRow;
                    int gx = col - FILTER_RADIUS + fCol;

                    if (gy >= 0 && gy < height && gx >= 0 && gx < width) {
                        Pvalue += M[gy * width + gx] * constFilter[filterIndex];
                    }
                }
            }
        }
        P[row * width + col] = Pvalue;
    }
}
