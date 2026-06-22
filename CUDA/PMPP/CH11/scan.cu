// nvcc scan.cu -o scan

#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define cdiv(x, y) (((x) + (y)-1) / (y))

#define COARSE_FACTOR 4

#define CUDA_CHECK(call)                                                                                 \
    do {                                                                                                 \
        cudaError_t error = call;                                                                        \
        if (error != cudaSuccess) {                                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE);                                                                          \
        }                                                                                                \
    } while (0)

__global__ void kogge_stone_scan_kernel(float* X, float* Y, unsigned int N) {
    extern __shared__ float buffer[];
    unsigned int tid = threadIdx.x;

    if (tid < N) {
        buffer[tid] = X[tid];
    } else {
        buffer[tid] = 0.0;
    }
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        float temp;
        __syncthreads();
        if (tid >= stride) {
            // read
            temp = buffer[tid] + buffer[tid - stride];
        }
        // make sure reading is done
        __syncthreads();
        if (tid >= stride) {
            // write the updated version
            buffer[tid] = temp;
        }
    }
    if (tid < N) {
        Y[tid] = buffer[tid];
    }
}

__global__ void kogge_stone_scan_kernel_with_double_buffering(float* X, float* Y, unsigned int N) {
    // we do this trick to use one shared_mem as cuda doesn't allow to declare two extern shared memory fields
    extern __shared__ float shared_mem[];
    float* buffer1 = shared_mem;
    float* buffer2 = &shared_mem[N];

    unsigned int tid = threadIdx.x;

    float* src_buffer = buffer1;
    float* trg_buffer = buffer2;

    if (tid < N) {
        src_buffer[tid] = X[tid];
    } else {
        src_buffer[tid] = 0.0;
    }
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        if (tid >= stride) {
            trg_buffer[tid] = src_buffer[tid] + src_buffer[tid - stride];
        } else {
            trg_buffer[tid] = src_buffer[tid];
        }

        float* temp;
        temp = src_buffer;
        src_buffer = trg_buffer;
        trg_buffer = temp;
    }

    if (tid < N) {
        Y[tid] = src_buffer[tid];
    }
}

__global__ void brent_kung_scan_kernel(float* X, float* Y, unsigned int N) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;

    // Load input into shared memory
    if (tid < N) {
        sdata[tid] = X[tid];
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();

    // Up-sweep phase (reduction)
    for (unsigned int offset = 1; offset < blockDim.x; offset *= 2) {
        int i = (tid + 1) * 2 * offset - 1;
        if (i < blockDim.x) {
            int j = i - offset;
            sdata[i] += sdata[j];
        }
        __syncthreads();
    }

    // Store total sum and clear last element
    if (tid == 0) {
        sdata[blockDim.x - 1] = 0;
    }
    __syncthreads();

    // Down-sweep phase
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        int i = (tid + 1) * 2 * offset - 1;
        if (i < blockDim.x) {
            int j = i - offset;
            float t = sdata[i];
            sdata[i] = sdata[j] + t;
            sdata[j] = t;
        }
        __syncthreads();
    }

    // Write results - convert to inclusive scan
    if (tid < N) {
        Y[tid] = sdata[tid] + X[tid];
    }
}

__global__ void three_phases_scan_kernel(float* X, float* Y, unsigned int N) {
    extern __shared__ float shared_mem[];
    float* buffer1 = shared_mem;
    float* sections_ends = &shared_mem[N];

    // Phase 1: Load data
    for (unsigned int i = 0; i < COARSE_FACTOR; i++) {
        unsigned int idx = threadIdx.x * COARSE_FACTOR + i;
        if (idx < N) {
            buffer1[idx] = X[idx];
        }
    }
    __syncthreads();

    // Local scan within each section
    for (unsigned int i = 1; i < COARSE_FACTOR; i++) {
        unsigned int idx = threadIdx.x * COARSE_FACTOR + i;
        if (idx < N) {
            buffer1[idx] += buffer1[idx - 1];
        }
    }
    __syncthreads();

    // Store section ends
    if (threadIdx.x < N / COARSE_FACTOR) {
        unsigned int section_end_idx = (threadIdx.x + 1) * COARSE_FACTOR - 1;
        if (section_end_idx < N) {
            sections_ends[threadIdx.x] = buffer1[section_end_idx];
        }
    }
    __syncthreads();

    // Phase 2: Kogge-Stone scan on section ends
    unsigned int tid = threadIdx.x;
    unsigned int num_sections = (N + COARSE_FACTOR - 1) / COARSE_FACTOR;

    for (unsigned int stride = 1; stride < num_sections; stride *= 2) {
        float temp;
        __syncthreads();
        if (tid >= stride && tid < num_sections) {
            temp = sections_ends[tid] + sections_ends[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && tid < num_sections) {
            sections_ends[tid] = temp;
        }
    }
    __syncthreads();

    // Phase 3: Distribute section sums
    for (unsigned int i = 0; i < COARSE_FACTOR; i++) {
        unsigned int idx = threadIdx.x * COARSE_FACTOR + i;
        if (idx < N) {
            unsigned int section = idx / COARSE_FACTOR;
            if (section > 0) {
                buffer1[idx] += sections_ends[section - 1];
            }
            Y[idx] = buffer1[idx];
        }
    }
}

bool allclose(float* a, float* b, int N, float rtol = 1e-5, float atol = 1e-8) {
    for (int i = 0; i < N; i++) {
        float allowed_error = atol + rtol * fabs(b[i]);
        if (fabs(a[i] - b[i]) > allowed_error) {
            printf("Arrays differ at index %d: %f != %f (allowed error: %f)\n", i, a[i], b[i], allowed_error);
            return false;
        }
    }
    return true;
}

void scan_via_kogge_stone(float* X, float* Y, unsigned int N) {
    assert(N <= 1024 && "Length must be less than or equal to 1024");

    float* d_X;
    float* d_Y;

    dim3 dimBlock(N);  // for now we stick to a single section executed within a single block
    dim3 dimGrid(1);

    CUDA_CHECK(cudaMalloc((void**)&d_X, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Y, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_X, X, N * sizeof(float), cudaMemcpyHostToDevice));

    kogge_stone_scan_kernel<<<dimGrid, dimBlock, N * sizeof(float)>>>(d_X, d_Y, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));
}

void scan_via_kogge_stone_with_double_buffering(float* X, float* Y, unsigned int N) {
    assert(N <= 1024 && "Length must be less than or equal to 1024");

    float* d_X;
    float* d_Y;

    dim3 dimBlock(N);  // for now we stick to a single section executed within a single block
    dim3 dimGrid(1);

    CUDA_CHECK(cudaMalloc((void**)&d_X, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Y, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_X, X, N * sizeof(float), cudaMemcpyHostToDevice));

    kogge_stone_scan_kernel_with_double_buffering<<<dimGrid, dimBlock, 2 * N * sizeof(float)>>>(d_X, d_Y, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));
}

void scan_via_brent_kung(float* X, float* Y, unsigned int N) {
    assert(N <= 1024 && "Length must be less than or equal to 1024");

    float* d_X;
    float* d_Y;

    // We'll use power of 2 for block size
    unsigned int blockSize = 1;
    while (blockSize < N && blockSize < 1024) {
        blockSize *= 2;
    }

    dim3 dimBlock(blockSize);
    dim3 dimGrid(1);

    CUDA_CHECK(cudaMalloc((void**)&d_X, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Y, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_X, X, N * sizeof(float), cudaMemcpyHostToDevice));

    brent_kung_scan_kernel<<<dimGrid, dimBlock, blockSize * sizeof(float)>>>(d_X, d_Y, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));
}

void scan_via_three_phase_kernel(float* X, float* Y, unsigned int N) {
    assert(N <= 1024 && "Length must be less than or equal to 1024");

    float* d_X;
    float* d_Y;

    dim3 dimBlock(cdiv(N, COARSE_FACTOR));
    dim3 dimGrid(1);

    CUDA_CHECK(cudaMalloc((void**)&d_X, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Y, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_X, X, N * sizeof(float), cudaMemcpyHostToDevice));

    three_phases_scan_kernel<<<dimGrid, dimBlock, (N + N / COARSE_FACTOR) * sizeof(float)>>>(d_X, d_Y, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));
}

void sequential_inclusive_scan(float* X, float* Y, unsigned int N) {
    Y[0] = X[0];
    for (unsigned int i = 1; i < N; i++) {
        Y[i] = X[i] + Y[i - 1];
    }
}

int main() {
    unsigned int length = 999;  // works for arbitrary numbers not just the powers of 2
    float* X = (float*)malloc(length * sizeof(float));
    float* Y_kogge_stone = (float*)malloc(length * sizeof(float));
    float* Y_kogge_stone_double = (float*)malloc(length * sizeof(float));
    float* Y_brent_kung = (float*)malloc(length * sizeof(float));
    float* Y_three_phases = (float*)malloc(length * sizeof(float));
    float* Y_sequential = (float*)malloc(length * sizeof(float));

    for (unsigned int i = 0; i < length; i++) {
        // X[i] = 9.0f * ((float)rand() / RAND_MAX);
        X[i] = 1.0 * (i + 1);
    }

    sequential_inclusive_scan(X, Y_sequential, length);
    scan_via_kogge_stone(X, Y_kogge_stone, length);
    scan_via_kogge_stone_with_double_buffering(X, Y_kogge_stone_double, length);
    scan_via_brent_kung(X, Y_brent_kung, length);
    scan_via_three_phase_kernel(X, Y_three_phases, length);

    // printf("Kogge Stone Scan results:           [");
    // for (unsigned int i = 0; i < length; i++) {
    //     printf("%.2f%s", Y_kogge_stone[i], (i < length-1) ? ", " : "");
    // }
    // printf("]\n");

    // printf("Kogge Stone Double Buffer results:  [");
    // for (unsigned int i = 0; i < length; i++) {
    //     printf("%.2f%s", Y_kogge_stone_double[i], (i < length-1) ? ", " : "");
    // }
    // printf("]\n");

    // printf("Brent-Kung Scan results:           [");
    // for (unsigned int i = 0; i < length; i++) {
    //     printf("%.2f%s", Y_brent_kung[i], (i < length-1) ? ", " : "");
    // }
    // printf("]\n");

    // printf("Three Phases Scan results:         [");
    // for (unsigned int i = 0; i < length; i++) {
    //     printf("%.2f%s", Y_three_phases[i], (i < length-1) ? ", " : "");
    // }
    // printf("]\n");

    // printf("Sequential Scan results:            [");
    // for (unsigned int i = 0; i < length; i++) {
    //     printf("%.2f%s", Y_sequential[i], (i < length-1) ? ", " : "");
    // }
    // printf("]\n");

    printf("\nComparing results...\n");
    printf("Comparing regular Kogge Stone with sequential:\n");
    if (allclose(Y_kogge_stone, Y_sequential, length)) {
        printf("Arrays are close enough!\n");
    } else {
        printf("Arrays differ significantly!\n");
    }

    printf("\nComparing double buffered Kogge Stone with sequential:\n");
    if (allclose(Y_kogge_stone_double, Y_sequential, length)) {
        printf("Arrays are close enough!\n");
    } else {
        printf("Arrays differ significantly!\n");
    }

    printf("\nComparing Brent-Kung with sequential:\n");
    if (allclose(Y_brent_kung, Y_sequential, length)) {
        printf("Arrays are close enough!\n");
    } else {
        printf("Arrays differ significantly!\n");
    }

    printf("\nComparing Three Phases with sequential:\n");
    if (allclose(Y_three_phases, Y_sequential, length)) {
        printf("Arrays are close enough!\n");
    } else {
        printf("Arrays differ significantly!\n");
    }

    free(X);
    free(Y_kogge_stone);
    free(Y_kogge_stone_double);
    free(Y_brent_kung);
    free(Y_three_phases);
    free(Y_sequential);
    return 0;
}
