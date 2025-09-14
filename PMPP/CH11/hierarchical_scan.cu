// nvcc hierarchical_scan.cu -o hierarchical_scan

#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SECTION_SIZE 1024  // Maximum threads per block
#define cdiv(x, y) (((x) + (y)-1) / (y))

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}
#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }

void clear_l2() {
    static int l2_clear_size = 0;
    static unsigned char* gpu_scratch_l2_clear = NULL;
    if (!gpu_scratch_l2_clear) {
        cudaDeviceGetAttribute(&l2_clear_size, cudaDevAttrL2CacheSize, 0);
        l2_clear_size *= 2;
        gpuErrchk(cudaMalloc(&gpu_scratch_l2_clear, l2_clear_size));
    }
    gpuErrchk(cudaMemset(gpu_scratch_l2_clear, 0, l2_clear_size));
}

// Phase 1: Block-level scan and collect block sums
__global__ void hierarchical_kogge_stone_phase1(float* X, float* Y, float* S, unsigned int N) {
    extern __shared__ float buffer[];

    unsigned int tid = threadIdx.x;
    unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx < N) {
        buffer[tid] = X[global_idx];
    } else {
        buffer[tid] = 0.0f;
    }

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        float temp;
        __syncthreads();
        if (tid >= stride) {
            temp = buffer[tid] + buffer[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            buffer[tid] = temp;
        }
    }

    if (global_idx < N) {
        Y[global_idx] = buffer[tid];
    }

    if (tid == blockDim.x - 1) {
        S[blockIdx.x] = buffer[tid];
    }
}

// Phase 2: Scan block sums
__global__ void hierarchical_kogge_stone_phase2(float* S, unsigned int num_blocks) {
    extern __shared__ float buffer[];
    unsigned int tid = threadIdx.x;
    unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only process if within valid range
    if (global_idx < num_blocks) {
        buffer[tid] = S[global_idx];
    } else {
        buffer[tid] = 0.0f;
    }
    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        float temp = buffer[tid];
        __syncthreads();

        if (tid >= stride) {
            temp += buffer[tid - stride];
        }
        __syncthreads();

        buffer[tid] = temp;
    }

    if (global_idx < num_blocks) {
        S[global_idx] = buffer[tid];
    }
}

// Phase 3: Distribute block sums
__global__ void hierarchical_kogge_stone_phase3(float* Y, float* S, unsigned int N) {
    unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx < N && blockIdx.x > 0) {
        Y[global_idx] += S[blockIdx.x - 1];
    }
}

void hierarchical_scan(float* X, float* Y, unsigned int N) {
    float *d_X, *d_Y, *d_S;
    unsigned int block_size = SECTION_SIZE;
    unsigned int num_blocks = cdiv(N, block_size);

    gpuErrchk(cudaMalloc((void**)&d_X, N * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&d_Y, N * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&d_S, num_blocks * sizeof(float)));

    gpuErrchk(cudaMemcpy(d_X, X, N * sizeof(float), cudaMemcpyHostToDevice));

    // Phase 1: Block-level scan and collect block sums
    hierarchical_kogge_stone_phase1<<<num_blocks, block_size, block_size * sizeof(float)>>>(d_X, d_Y, d_S, N);
    gpuErrchk(cudaDeviceSynchronize());

    // Phase 2: Process block sums with multiple blocks if needed
    unsigned int block_size_phase2 = SECTION_SIZE;
    unsigned int num_blocks_phase2 = cdiv(num_blocks, block_size_phase2);

    hierarchical_kogge_stone_phase2<<<num_blocks_phase2, block_size_phase2, block_size_phase2 * sizeof(float)>>>(
        d_S, num_blocks);
    gpuErrchk(cudaDeviceSynchronize());

    // Phase 3: Distribute block sums
    hierarchical_kogge_stone_phase3<<<num_blocks, block_size>>>(d_Y, d_S, N);
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_X));
    gpuErrchk(cudaFree(d_Y));
    gpuErrchk(cudaFree(d_S));
}

// Single-kernel domino-style scan implementation
__global__ void hierarchical_kogge_stone_domino(float* X, float* Y, float* scan_value, int* flags, int* blockCounter,
                                                unsigned int N) {
    extern __shared__ float buffer[];
    __shared__ unsigned int bid_s;
    __shared__ float previous_sum;

    const unsigned int tid = threadIdx.x;

    // DEADLOCK PREVENTION: Dynamic block index assignment
    if (tid == 0) {
        bid_s = atomicAdd(blockCounter, 1);
    }
    __syncthreads();

    const unsigned int bid = bid_s;
    const unsigned int gid = bid * blockDim.x + tid;

    // Phase 1: Local block scan using Kogge-Stone
    if (gid < N) {
        buffer[tid] = X[gid];
    } else {
        buffer[tid] = 0.0f;
    }

    // Kogge-Stone scan within block
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float temp = buffer[tid];
        if (tid >= stride) {
            temp += buffer[tid - stride];
        }
        __syncthreads();
        buffer[tid] = temp;
    }

    // Store local result
    if (gid < N) {
        Y[gid] = buffer[tid];
    }

    // Get local sum for this block
    const float local_sum = buffer[blockDim.x - 1];

    // Phase 2: Inter-block sum propagation
    if (tid == 0) {
        if (bid > 0) {
            // Wait for previous block's flag
            while (atomicAdd(&flags[bid], 0) == 0) {
            }

            // Get sum from previous block
            previous_sum = scan_value[bid];

            // Add local sum and propagate
            const float total_sum = previous_sum + local_sum;
            scan_value[bid + 1] = total_sum;

            // Ensure scan_value is visible
            __threadfence();

            // Signal next block
            atomicAdd(&flags[bid + 1], 1);
        } else {
            // First block just propagates its sum
            scan_value[1] = local_sum;
            __threadfence();
            atomicAdd(&flags[1], 1);
        }
    }
    __syncthreads();

    // Phase 3: Add previous block's sum to local results
    if (bid > 0 && gid < N) {
        Y[gid] += previous_sum;
    }
}

void hierarchical_scan_with_domino_like_sync(float* X, float* Y, unsigned int N) {
    float *d_X, *d_Y, *d_scan_value;
    int *d_flags, *d_blockCounter;

    const unsigned int block_size = SECTION_SIZE;
    const unsigned int num_blocks = cdiv(N, block_size);

    gpuErrchk(cudaMalloc((void**)&d_X, N * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&d_Y, N * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&d_scan_value, (num_blocks + 1) * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&d_flags, (num_blocks + 1) * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&d_blockCounter, sizeof(int)));

    gpuErrchk(cudaMemset(d_flags, 0, (num_blocks + 1) * sizeof(int)));
    gpuErrchk(cudaMemset(d_blockCounter, 0, sizeof(int)));

    gpuErrchk(cudaMemcpy(d_X, X, N * sizeof(float), cudaMemcpyHostToDevice));

    hierarchical_kogge_stone_domino<<<num_blocks, block_size, block_size * sizeof(float)>>>(d_X, d_Y, d_scan_value,
                                                                                            d_flags, d_blockCounter, N);
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_X));
    gpuErrchk(cudaFree(d_Y));
    gpuErrchk(cudaFree(d_scan_value));
    gpuErrchk(cudaFree(d_flags));
    gpuErrchk(cudaFree(d_blockCounter));
}

void sequential_inclusive_scan(float* X, float* Y, unsigned int N) {
    Y[0] = X[0];
    for (unsigned int i = 1; i < N; i++) {
        Y[i] = X[i] + Y[i - 1];
    }
}

float benchmark_scan(void (*scan_func)(float*, float*, unsigned int), float* X, float* Y, unsigned int N, int warmup,
                     int reps) {
    // Warmup runs
    for (int i = 0; i < warmup; ++i) {
        scan_func(X, Y, N);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float totalTime_ms = 0.0f;

    // Benchmark runs
    for (int i = 0; i < reps; ++i) {
        clear_l2();  // Clear L2 cache before each run

        cudaEventRecord(start);
        scan_func(X, Y, N);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        float iterTime = 0.0f;
        cudaEventElapsedTime(&iterTime, start, stop);
        totalTime_ms += iterTime;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return totalTime_ms / reps;
}

bool allclose(float* a, float* b, int N, float rtol = 1e-5, float atol = 1e-8) {
    for (int i = 0; i < N; i++) {
        float allowed_error = atol + rtol * fabs(b[i]);
        if (fabs(a[i] - b[i]) > allowed_error) {
            printf("Arrays differ at index %d: %f != %f (allowed error: %f)\n", i, a[i], b[i], allowed_error);
            printf("Values around error point:\n");
            int start = (i > 5) ? i - 5 : 0;
            int end = (i + 5 < N) ? i + 5 : N - 1;
            printf("Index\tSequential\tHierarchical\n");
            for (int j = start; j <= end; j++) {
                printf("%d\t%.1f\t\t%.1f\n", j, a[j], b[j]);
            }
            return false;
        }
    }
    return true;
}

int main() {
    unsigned int sizes[] = {16384, 65536, 262144, 1048576};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    // Benchmark parameters
    const int warmup = 5;
    const int reps = 10;

    printf("\nBenchmarking Scan Operations\n");
    printf("----------------------------\n");
    printf("%-10s %-15s %-15s %-15s %-10s %-10s %-8s\n", "Size", "Sequential(ms)", "Hierarchical(ms)", "Domino(ms)",
           "Speedup-H", "Speedup-D", "Match");

    for (int i = 0; i < num_sizes; i++) {
        unsigned int N = sizes[i];

        float* X = (float*)malloc(N * sizeof(float));
        float* Y_sequential = (float*)malloc(N * sizeof(float));
        float* Y_hierarchical = (float*)malloc(N * sizeof(float));
        float* Y_domino = (float*)malloc(N * sizeof(float));

        for (unsigned int j = 0; j < N; j++) {
            X[j] = 1.0f;
        }

        float sequential_time = benchmark_scan(sequential_inclusive_scan, X, Y_sequential, N, warmup, reps);
        float hierarchical_time = benchmark_scan(hierarchical_scan, X, Y_hierarchical, N, warmup, reps);
        float domino_time = benchmark_scan(hierarchical_scan_with_domino_like_sync, X, Y_domino, N, warmup, reps);

        bool hier_match = allclose(Y_sequential, Y_hierarchical, N);
        bool domino_match = allclose(Y_sequential, Y_domino, N);

        printf("%-10u %-15.3f %-15.3f %-15.3f %-10.2f %-10.2f %s%s\n", N, sequential_time, hierarchical_time,
               domino_time, sequential_time / hierarchical_time, sequential_time / domino_time, hier_match ? "✓" : "✗",
               domino_match ? "✓" : "✗");

        free(X);
        free(Y_sequential);
        free(Y_hierarchical);
        free(Y_domino);
    }
    return 0;
}
