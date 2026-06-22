// nvcc reduction_sum_2048.cu reduction_common.cu -o reduction_sum_2048
// limited to kernels that work only within a single block - hence are limited to 2048 elements
#include "reduction_common.cuh"

#define BLOCK_DIM 1024

__global__ void simple_sum_reduction_kernel(float* input, float* output) {
    unsigned int i = 2 * threadIdx.x;

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *output = input[0];
    }
}

__global__ void covergent_sum_reduction_kernel(float* input, float* output) {
    unsigned int i = threadIdx.x;

    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *output = input[0];
    }
}

__global__ void covergent_sum_reduction_kernel_reversed(float* input, float* output) {
    unsigned int i = threadIdx.x + blockDim.x;
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        // stride iterations remains the same, but we just use it to index the previous input to be taken
        if (blockDim.x - threadIdx.x <= stride) {
            input[i] += input[i - stride];
        }
        __syncthreads();
    }
    // take it from the last input
    if (threadIdx.x == blockDim.x - 1) {
        *output = input[i];
    }
}

__global__ void shared_memory_sum_reduction_kernel(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int t = threadIdx.x;

    input_s[t] = input[t] + input[t + BLOCK_DIM];

    // we already did the first iteration compared to covergent_sum_reduction_kernel so we start with the 2nd one
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }

    if (t == 0) {
        *output = input_s[0];
    }
}

float simple_parallel_sum_reduction(float* data, int length) {
    assert(length == 2 * BLOCK_DIM && "Length must be equal to 2 * BLOCK_DIM");

    float total;
    float* d_total;
    float* d_data;

    dim3 dimBlock(BLOCK_DIM);  // we always run this with as much threads in block as possible
    dim3 dimGrid(1);           // since the blocks can't communicate we are stuck for now with a single block

    CUDA_CHECK(cudaMalloc((void**)&d_data, length * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_total, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_data, data, length * sizeof(float), cudaMemcpyHostToDevice));

    simple_sum_reduction_kernel<<<dimGrid, dimBlock>>>(d_data, d_total);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_total));

    return total;
}

float covergent_parallel_sum_reduction(float* data, int length) {
    assert(length == 2 * BLOCK_DIM && "Length must be equal to 2 * BLOCK_DIM");

    float total;
    float* d_total;
    float* d_data;

    dim3 dimBlock(BLOCK_DIM);  // we always run this with as much threads in block as possible
    dim3 dimGrid(1);           // since the blocks can't communicate we are stuck for now with a single block

    CUDA_CHECK(cudaMalloc((void**)&d_data, length * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_total, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_data, data, length * sizeof(float), cudaMemcpyHostToDevice));

    covergent_sum_reduction_kernel<<<dimGrid, dimBlock>>>(d_data, d_total);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_total));

    return total;
}

float covergent_parallel_sum_reduction_reversed(float* data, int length) {
    assert(length == 2 * BLOCK_DIM && "Length must be equal to 2 * BLOCK_DIM");

    float total;
    float* d_total;
    float* d_data;

    dim3 dimBlock(BLOCK_DIM);  // we always run this with as much threads in block as possible
    dim3 dimGrid(1);           // since the blocks can't communicate we are stuck for now with a single block

    CUDA_CHECK(cudaMalloc((void**)&d_data, length * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_total, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_data, data, length * sizeof(float), cudaMemcpyHostToDevice));

    covergent_sum_reduction_kernel_reversed<<<dimGrid, dimBlock>>>(d_data, d_total);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_total));

    return total;
}

float shared_memory_sum_reduction(float* data, int length) {
    assert(length == 2 * BLOCK_DIM && "Length must be equal to 2 * BLOCK_DIM");

    float total;
    float* d_total;
    float* d_data;

    dim3 dimBlock(BLOCK_DIM);  // we always run this with as much threads in block as possible
    dim3 dimGrid(1);           // since the blocks can't communicate we are stuck for now with a single block

    CUDA_CHECK(cudaMalloc((void**)&d_data, length * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_total, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_data, data, length * sizeof(float), cudaMemcpyHostToDevice));

    shared_memory_sum_reduction_kernel<<<dimGrid, dimBlock>>>(d_data, d_total);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_total));

    return total;
}

int main(int argc, char const* argv[]) {
    unsigned int length = 2048;
    float* numbers = (float*)malloc(length * sizeof(float));

    for (unsigned int i = 0; i < length; i++) {
        numbers[i] = 1.0f;
    }

    printf("Benchmarking simple parallel sum reduction...\n");
    float simple_parallel_time = benchmark_sum_reduction(simple_parallel_sum_reduction, numbers, length);
    float simple_parallel_sum = simple_parallel_sum_reduction(numbers, length);

    printf("Benchmarking convergent parallel sum reduction...\n");
    float convergent_parallel_time = benchmark_sum_reduction(covergent_parallel_sum_reduction, numbers, length);
    float convergent_parallel_sum = covergent_parallel_sum_reduction(numbers, length);

    printf("Benchmarking shared memory parallel sum reduction...\n");
    float shared_memory_time = benchmark_sum_reduction(shared_memory_sum_reduction, numbers, length);
    float shared_memory_sum = shared_memory_sum_reduction(numbers, length);

    printf("Benchmarking reversed convergent parallel sum reduction...\n");
    float convergent_parallel_reversed_time =
        benchmark_sum_reduction(covergent_parallel_sum_reduction_reversed, numbers, length);
    float convergent_parallel_reversed_sum = covergent_parallel_sum_reduction_reversed(numbers, length);

    printf("Benchmarking sequential sum reduction...\n");
    float sequential_time = benchmark_sum_reduction(sequential_sum_reduction, numbers, length, 10, 10);
    float sequential_sum = sequential_sum_reduction(numbers, length);

    printf("\nResults:\n");
    printf("Simple Parallel Implementation:\n");
    printf("Sum: %.2f\n", simple_parallel_sum);
    printf("Average time: %.3f ms\n", simple_parallel_time);

    printf("\nConvergent Parallel Implementation:\n");
    printf("Sum: %.2f\n", convergent_parallel_sum);
    printf("Average time: %.3f ms\n", convergent_parallel_time);

    printf("\nReversed Convergent Parallel Implementation:\n");
    printf("Sum: %.2f\n", convergent_parallel_reversed_sum);
    printf("Average time: %.3f ms\n", convergent_parallel_reversed_time);

    printf("\nShared Memory Parallel Implementation:\n");
    printf("Sum: %.2f\n", shared_memory_sum);
    printf("Average time: %.3f ms\n", shared_memory_time);

    printf("\nSequential Implementation:\n");
    printf("Sum: %.2f\n", sequential_sum);
    printf("Average time: %.3f ms\n", sequential_time);

    printf("\nSpeedup:\n");
    printf("Simple Parallel vs Sequential: %.2fx\n", sequential_time / simple_parallel_time);
    printf("Convergent Parallel vs Sequential: %.2fx\n", sequential_time / convergent_parallel_time);
    printf("Reversed Convergent Parallel vs Sequential: %.2fx\n", sequential_time / convergent_parallel_reversed_time);
    printf("Shared Memory Parallel vs Sequential: %.2fx\n", sequential_time / shared_memory_time);

    bool results_match = is_close(sequential_sum, simple_parallel_sum) &&
                         is_close(simple_parallel_sum, convergent_parallel_sum) &&
                         is_close(convergent_parallel_sum, shared_memory_sum);

    printf("\nAll results match: %s\n", results_match ? "Yes" : "No");

    free(numbers);
    return 0;
}
