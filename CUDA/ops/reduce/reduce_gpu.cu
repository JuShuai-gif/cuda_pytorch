#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <stdio.h>

const int NUM_REPEATS = 100;
const int N = 100000000;

const int M = sizeof(float) * N;
const int BLOCK_SIZE = 128;

void timing(float *h_x, float *d_x, const int method);

int main(void) {
    float *h_x = (float *)malloc(M);

    for (int n = 0; n < N; ++n) {
        h_x[n] = 1.23f;
    }

    float *d_x;
    CHECK(cudaMalloc(&d_x, M));

    printf("\nUsing global memory only:\n");
    timing(h_x, d_x, 0);
    printf("\nUsing static shared memory:\n");
    timing(h_x, d_x, 1);
    printf("\nUsing dynamic shared memory:\n");
    timing(h_x, d_x, 2);

    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}

void __global__ reduce_global(float *d_x, float *d_y) {
    const int tid = threadIdx.x;

    float *x = d_x + blockDim.x * blockIdx.x;

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            x[tid] += x[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_y[blockIdx.x] = x[0];
    }
}

void __global__ reduce_shared(float *d_x, float *d_y) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_y[128];
    s_y[tid] = (idx < N) ? d_x[idx] : 0.0;

    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_y[blockIdx.x] = s_y[0];
    }
}

void __global__ reduce_dynamic(float *d_x, float *d_y) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ float s_y[];

    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_y[bid] = s_y[0];
    }
}


float reduce(float* d_x,const int method){
    int grid_size = (N + BLOCK_SIZE - 1)/BLOCK_SIZE;
    const int ymem = sizeof(float)* grid_size;
    const int smem = sizeof(float)*BLOCK_SIZE;

    float* d_y;
    CHECK(cudaMalloc(&d_y,ymem));
    float* h_y = (float*)malloc(ymem);

    switch (method) {
        case 0:
            reduce_global<<<grid_size,BLOCK_SIZE>>>(d_x, d_y);
            break;
        case 1:
            reduce_shared<<<grid_size,BLOCK_SIZE>>>(d_x, d_y);
            break;
        case 2:
            reduce_dynamic<<<grid_size,BLOCK_SIZE>>>(d_x, d_y);
            break;
        default:
            printf("Error: wrong method\n");
            exit(1);
            break;
    }

    CHECK(cudaMemcpy(h_y, d_y, ymem, cudaMemcpyDeviceToHost));

    float result = 0.0;
    for (int n = 0; n < grid_size; ++n) {
        result += h_y[n];
    }

    free(h_y);
    CHECK(cudaFree(d_y));
    return result;
}

void timing(float *h_x, float *d_x, const int method)
{
    float sum = 0;

    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

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













