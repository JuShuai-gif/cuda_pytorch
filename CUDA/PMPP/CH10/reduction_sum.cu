// nvcc reduction_sum.cu reduction_common.cu -o reduction_sum
#include "reduction_common.cuh"

#define BLOCK_DIM 1024
#define COARSE_FACTOR 2

__global__ void segmented_sum_reduction_kernel(float* input, float* output, int length) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int segment = 2 * blockDim.x * blockIdx.x;

    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;

    if (i >= length || i + BLOCK_DIM >= length) {
        return;
    }

    input_s[t] = input[i] + input[i + BLOCK_DIM];

    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }

    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}

__global__ void coarsed_sum_reduction_kernel(float* input, float* output, int length) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;

    float sum = 0.0f;
    if (i < length) {
        sum = input[i];

        for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; ++tile) {
            if (i + tile * BLOCK_DIM < length) {
                sum += input[i + tile * BLOCK_DIM];
            }
        }
    }

    input_s[t] = sum;

    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }
    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}

float segmented_sum_reduction(float* data, int length) {
    float total;
    float* d_total;
    float* d_data;

    dim3 dimBlock(BLOCK_DIM);  // we always run this with as much threads in block as possible
    dim3 dimGrid((length + BLOCK_DIM - 1) /
                 BLOCK_DIM);  // since the blocks can't communicate we are stuck for now with a single block

    CUDA_CHECK(cudaMalloc((void**)&d_data, length * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_total, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_data, data, length * sizeof(float), cudaMemcpyHostToDevice));

    segmented_sum_reduction_kernel<<<dimGrid, dimBlock>>>(d_data, d_total, length);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_total));

    return total;
}

float coarsed_sum_reduction(float* data, int length) {
    float total;
    float* d_total;
    float* d_data;

    int elementsPerBlock = 2 * COARSE_FACTOR * BLOCK_DIM;

    dim3 dimBlock(BLOCK_DIM);  // we always run this with as much threads in block as possible
    dim3 dimGrid((length + elementsPerBlock - 1) /
                 elementsPerBlock);  // since the blocks can't communicate we are stuck for now with a single block

    CUDA_CHECK(cudaMalloc((void**)&d_data, length * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_total, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_data, data, length * sizeof(float), cudaMemcpyHostToDevice));

    coarsed_sum_reduction_kernel<<<dimGrid, dimBlock>>>(d_data, d_total, length);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_total));

    return total;
}

int main(int argc, char const* argv[]) {
    unsigned int length = 20480000;
    float* numbers = (float*)malloc(length * sizeof(float));

    for (unsigned int i = 0; i < length; i++) {
        numbers[i] = 1.0f;
    }

    printf("Benchmarking segmented sum reduction...\n");
    float segmented_time = benchmark_sum_reduction(segmented_sum_reduction, numbers, length);
    float segmented_sum = segmented_sum_reduction(numbers, length);

    printf("Benchmarking coarsed sum reduction...\n");
    float coarsed_time = benchmark_sum_reduction(coarsed_sum_reduction, numbers, length);
    float coarsed_sum = coarsed_sum_reduction(numbers, length);

    printf("Benchmarking sequential sum reduction...\n");
    float sequential_time = benchmark_sum_reduction(sequential_sum_reduction, numbers, length, 10, 10);
    float sequential_sum = sequential_sum_reduction(numbers, length);

    printf("\nResults:\n");

    printf("\nSegmented Sum Implementation:\n");
    printf("Sum: %.2f\n", segmented_sum);
    printf("Average time: %.3f ms\n", segmented_time);

    printf("\nCoarsed Sum Implementation:\n");
    printf("Sum: %.2f\n", coarsed_sum);
    printf("Average time: %.3f ms\n", coarsed_time);

    printf("\nSequential Implementation:\n");
    printf("Sum: %.2f\n", sequential_sum);
    printf("Average time: %.3f ms\n", sequential_time);

    printf("\nSpeedup:\n");
    printf("Segmented Sum vs Sequential: %.2fx\n", sequential_time / segmented_time);
    printf("Coarsed Sum vs Sequential: %.2fx\n", sequential_time / coarsed_time);

    bool results_match = is_close(sequential_sum, segmented_sum) && is_close(segmented_sum, coarsed_sum);
    printf("\nAll results match: %s\n", results_match ? "Yes" : "No");

    free(numbers);
    return 0;
}
