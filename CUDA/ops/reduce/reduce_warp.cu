#include "common.h"

#include <stdio.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

const int NUM_REPEATS = 100;
const int N = 100000000;
const int M = sizeof(float) * N;
const int BLOCK_SIZE = 128;
const unsigned FULL_MASK = 0xffffffff;

void timing(const float *d_x, const int method);

int main(void) {
    float *h_x = (float *)malloc(M);
    for (int n = 0; n < N; ++n) {
        h_x[n] = 1.23;
    }
    float *d_x;
    CHECK(cudaMalloc(&d_x, M));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

    printf("\nusing syncwarp:\n");
    timing(d_x, 0);
    printf("\nusing shfl:\n");
    timing(d_x, 1);
    printf("\nusing cooperative group:\n");
    timing(d_x, 2);

    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}

void __global__ reduce_syncwarp(const float *d_x, float *d_y, const int N) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int n = bid * blockDim.x + tid;
    extern __shared__ float s_y[];

    s_y[tid] = (n < N) ? d_x[n] : 0.0;

    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_y, s_y[0]);
    }
}

void __global__ reduce_shfl(const float *d_x, float *d_y, const int N) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ float s_y[];

    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    float y = s_y[tid];

    for (int offset = 16; offset > 0; offset >>= 1) {
        y += __shfl_down_sync(FULL_MASK, y, offset);
    }

    if (tid == 0) {
        atomicAdd(d_y, y);
    }
}

void __global__ reduce_cp(const float *d_x, float *d_y, const int N) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ float s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    float y = s_y[tid];

    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
    for (int i = g.size() >> 1; i > 0; i >>= 1) {
        y += g.shfl_down(y, i);
    }

    if (tid == 0) {
        atomicAdd(d_y, y);
    }
}

float reduce(const float *d_x, const int method) {
    const int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int smem = sizeof(float) * BLOCK_SIZE;

    float h_y[1] = {0};
    float *d_y;
    CHECK(cudaMalloc(&d_y, sizeof(float)));
    CHECK(cudaMemcpy(d_y, h_y, sizeof(float), cudaMemcpyHostToDevice));

    switch (method) {
    case 0:
        reduce_syncwarp<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y, N);
        break;
    case 1:
        reduce_shfl<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y, N);
        break;
    case 2:
        reduce_cp<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y, N);
        break;
    default:
        printf("Wrong method.\n");
        exit(1);
    }

    CHECK(cudaMemcpy(h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_y));

    return h_y[0];
}

void timing(const float *d_x, const int method) {
    float sum = 0;

    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat) {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        sum = reduce(d_x, method);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    printf("sum = %f.\n", sum);
}
