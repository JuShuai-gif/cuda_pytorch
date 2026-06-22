// nvcc reduction_max.cu -o reduction_max
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_DIM 16
#define COARSE_FACTOR 2

__global__ void CoarsenedMaxReductionKernel(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    float maximum_value = input[i];
    for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; ++tile) {
        maximum_value = fmax(maximum_value, input[i + tile * BLOCK_DIM]);
    }
    input_s[t] = maximum_value;

    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            input_s[t] = fmax(input_s[t], input_s[t + stride]);
        }
    }
    if (t == 0) {
        atomicExch(output, fmax(*output, input_s[0]));
    }
}

void checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(error), file, line);
        exit(EXIT_FAILURE);
    }
}
#define CHECK_CUDA_ERROR(error) checkCudaError(error, __FILE__, __LINE__)

float cpuMaxReduction(float* input, int size) {
    float maxVal = input[0];
    for (int i = 1; i < size; i++) {
        maxVal = fmax(maxVal, input[i]);
    }
    return maxVal;
}

int main() {
    const int numBlocks = 2;
    const int totalElements = numBlocks * BLOCK_DIM * COARSE_FACTOR * 2;
    const size_t size = totalElements * sizeof(float);

    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(sizeof(float));

    for (int i = 0; i < totalElements; i++) {
        h_input[i] = (float)(rand() % 100);  // Random values between 0 and 99
        printf("%.2f, ", h_input[i]);
    }
    printf("\n");
    *h_output = 0.0f;

    float *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_output, h_output, sizeof(float), cudaMemcpyHostToDevice));

    CoarsenedMaxReductionKernel<<<numBlocks, BLOCK_DIM>>>(d_input, d_output);

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    float cpuMax = cpuMaxReduction(h_input, totalElements);
    float gpuMax = *h_output;

    printf("CPU Max: %f\n", cpuMax);
    printf("GPU Max: %f\n", gpuMax);

    const float epsilon = 1e-5;
    if (fabs(cpuMax - gpuMax) < epsilon) {
        printf("Test PASSED!\n");
    } else {
        printf("Test FAILED!\n");
        printf("Difference: %f\n", fabs(cpuMax - gpuMax));
    }

    free(h_input);
    free(h_output);
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));

    return 0;
}
