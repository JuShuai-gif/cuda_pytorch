#include <iomanip>
#include <iostream>

#include "conv2d_functions.cuh"
#include "conv2d_kernels.cuh"

void conv2d_with_constant_memory(float* M, float* F, float* P, int r, int height, int width) {
    float *d_M, *d_P;
    cudaError_t error;

    // Initialize constant memory using the function from kernels.cu
    error = initConstFilter(F, r);
    if (error != cudaSuccess) {
        std::cout << "Failed to initialize constant memory: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMalloc((void**)&d_M, width * height * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_M failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMalloc((void**)&d_P, width * height * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_P failed: " << cudaGetErrorString(error) << std::endl;
        cudaFree(d_M);
        return;
    }

    cudaMemcpy(d_M, M, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_P, 0, width * height * sizeof(float));

    dim3 dimBloc(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(cdiv(width, dimBloc.x), cdiv(height, dimBloc.y));
    conv2d_kernel_with_constant_memory<<<dimGrid, dimBloc>>>(d_M, d_P, r, height, width);

    cudaMemcpy(P, d_P, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_M);
    cudaFree(d_P);
}

void conv2d_with_tiled_convolution(float* M, float* F, float* P, int r, int height, int width) {
    float *d_M, *d_P;
    cudaError_t error;

    // Initialize constant memory using the function from kernels.cu
    error = initConstFilter(F, r);
    if (error != cudaSuccess) {
        std::cout << "Failed to initialize constant memory: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMalloc((void**)&d_M, width * height * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_M failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMalloc((void**)&d_P, width * height * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_P failed: " << cudaGetErrorString(error) << std::endl;
        cudaFree(d_M);
        return;
    }

    cudaMemcpy(d_M, M, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_P, 0, width * height * sizeof(float));

    dim3 dimBlock(IN_TILE_SIZE, IN_TILE_SIZE);
    dim3 dimGrid(cdiv(width + 2 * FILTER_RADIUS, OUT_TILE_SIZE), cdiv(height + 2 * FILTER_RADIUS, OUT_TILE_SIZE));

    tiled_convolution_kernel<<<dimGrid, dimBlock>>>(d_M, d_P, r, height, width);

    cudaMemcpy(P, d_P, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_M);
    cudaFree(d_P);
}

void conv2d_with_tiled_convolution_with_l2_caching(float* M, float* F, float* P, int r, int height, int width) {
    float *d_M, *d_P;
    cudaError_t error;

    // Initialize constant memory using the function from kernels.cu
    error = initConstFilter(F, r);
    if (error != cudaSuccess) {
        std::cout << "Failed to initialize constant memory: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMalloc((void**)&d_M, width * height * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_M failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMalloc((void**)&d_P, width * height * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_P failed: " << cudaGetErrorString(error) << std::endl;
        cudaFree(d_M);
        return;
    }

    cudaMemcpy(d_M, M, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_P, 0, width * height * sizeof(float));

    dim3 dimBlock(IN_TILE_SIZE, IN_TILE_SIZE);
    dim3 dimGrid(cdiv(width + 2 * FILTER_RADIUS, OUT_TILE_SIZE), cdiv(height + 2 * FILTER_RADIUS, OUT_TILE_SIZE));

    tiled_convolution_kernel_with_l2_caching<<<dimGrid, dimBlock>>>(d_M, d_P, r, height, width);

    cudaMemcpy(P, d_P, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_M);
    cudaFree(d_P);
}
