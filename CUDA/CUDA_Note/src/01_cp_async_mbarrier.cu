#include <cuda_runtime.h>
#include <cuda/barrier>
#include <stdio.h>

using barrier = cuda::barrier<cuda::thread_scope_block>;

constexpr int N = 1024;
constexpr int BLOCK_SIZE = 256;

// Producer-consumer with cp.async using mbarrier (SM80+)
// Demonstrates: mbarrier for collective wait on async copy completion
__global__ void cp_async_mbarrier_demo(float *dst, const float *src, int n) {
    extern __shared__ float smem[];

    barrier *b = reinterpret_cast<barrier *>(&smem[0]);
    float *smem_data = &smem[BLOCK_SIZE]; // after barrier object

    int tid = threadIdx.x;

    if (tid == 0) {
        init(b, blockDim.x);
    }
    __syncthreads();

    if (tid == 0) {
        b->arrive_and_wait();
        __threadfence_block();
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + tid;
    float local_compute = 0.0f;

    if (idx < n) {
        // Issue cp.async: dst, src, size, barrier
        cuda::memcpy_async(&smem_data[tid], &src[idx], sizeof(float), *b);
        local_compute = src[idx] * 0.5f;
    } else {
        // Even threads with no data must arrive to avoid deadlock
        cuda::memcpy_async(&smem_data[tid], &src[0], sizeof(float), *b);
    }

    barrier::arrival_token token = b->arrive();
    b->wait(std::move(token));

    if (idx < n) {
        dst[idx] = smem_data[tid] + local_compute;
    }

    __syncthreads();
}

// Classic flag-based approach for comparison
// Uses __threadfence + volatile flag for producer-consumer
__global__ void flag_based_copy(float *dst, const float *src, int n) {
    extern __shared__ float smem[];
    float *smem_data = smem;
    int *flag = reinterpret_cast<int *>(&smem[BLOCK_SIZE]);

    int tid = threadIdx.x;

    if (tid == 0) {
        *flag = 0;
    }
    __syncthreads();

    // Producer: copy data and set flag
    int idx = blockIdx.x * blockDim.x + tid;
    if (idx < n) {
        smem_data[tid] = src[idx];
    }
    __threadfence_block();

    if (tid == 0) {
        atomicExch(flag, 1);
    }

    // Consumer: read flag then consume data
    // Problem: cannot express "which round" without additional state
    if (tid == 0) {
        while (atomicAdd(flag, 0) != 1) {
            // spin-wait polling
        }
    }
    __syncthreads();

    // Now consume
    if (idx < n) {
        dst[idx] = smem_data[tid] * 1.5f;
    }
}

// Two-phase barrier: demonstrate phase reuse
// Each phase uses a different part of shared memory
__global__ void barrier_phase_reuse(float *dst, const float *a, const float *b, int n) {
    extern __shared__ float smem[];

    barrier *bar = reinterpret_cast<barrier *>(&smem[0]);
    float *buf_a = &smem[BLOCK_SIZE];
    float *buf_b = &smem[BLOCK_SIZE * 2];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (tid == 0) {
        init(bar, blockDim.x);
    }
    __syncthreads();

    // Phase 0: load vector A via barrier-controlled async copy
    if (idx < n) {
        cuda::memcpy_async(&buf_a[tid], &a[idx], sizeof(float), *bar);
    } else {
        cuda::memcpy_async(&buf_a[tid], &a[0], sizeof(float), *bar);
    }
    auto token_a = bar->arrive();
    bar->wait(std::move(token_a));

    // Phase 1: load vector B, reuse same barrier (phase flips automatically)
    if (idx < n) {
        cuda::memcpy_async(&buf_b[tid], &b[idx], sizeof(float), *bar);
    } else {
        cuda::memcpy_async(&buf_b[tid], &b[0], sizeof(float), *bar);
    }
    auto token_b = bar->arrive();
    bar->wait(std::move(token_b));

    // Compute: dst = buf_a + buf_b
    if (idx < n) {
        dst[idx] = buf_a[tid] + buf_b[tid];
    }
}

void run_barrier_test() {
    float *h_src, *h_dst_barrier, *h_dst_flag;
    float *d_src, *d_dst_barrier, *d_dst_flag, *d_dst_phase;

    h_src = (float *)malloc(N * sizeof(float));
    h_dst_barrier = (float *)malloc(N * sizeof(float));
    h_dst_flag = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_src[i] = static_cast<float>(i);
    }

    cudaMalloc(&d_src, N * sizeof(float));
    cudaMalloc(&d_dst_barrier, N * sizeof(float));
    cudaMalloc(&d_dst_flag, N * sizeof(float));
    cudaMalloc(&d_dst_phase, N * sizeof(float));

    cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice);

    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Shared memory: barrier object + data buffer + flag
    size_t smem_barrier = sizeof(barrier) + BLOCK_SIZE * sizeof(float);
    size_t smem_flag = BLOCK_SIZE * sizeof(float) + sizeof(int);

    printf("=== cp.async + mbarrier Demo (Barrier Approach) ===\n");
    cp_async_mbarrier_demo<<<grid_size, BLOCK_SIZE, smem_barrier>>>(d_dst_barrier, d_src, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_dst_barrier, d_dst_barrier, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("=== Flag-based Approach (for comparison) ===\n");
    flag_based_copy<<<grid_size, BLOCK_SIZE, smem_flag>>>(d_dst_flag, d_src, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_dst_flag, d_dst_flag, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare results
    bool match = true;
    for (int i = 0; i < N; i++) {
        if (h_dst_barrier[i] != h_dst_flag[i]) {
            match = false;
            printf("Mismatch at %d: barrier=%f flag=%f\n", i, h_dst_barrier[i], h_dst_flag[i]);
            break;
        }
    }
    printf("Results %s\n\n", match ? "MATCH" : "MISMATCH");

    // Phase reuse test: use two input arrays
    float *h_a = (float *)malloc(N * sizeof(float));
    float *h_b = (float *)malloc(N * sizeof(float));
    float *h_phase_dst = (float *)malloc(N * sizeof(float));
    float *h_expected = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
        h_expected[i] = h_a[i] + h_b[i];
    }

    cudaMemcpy(d_src, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    // We need a second source array on device
    float *d_b;
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    size_t smem_phase = sizeof(barrier) + BLOCK_SIZE * 2 * sizeof(float);
    printf("=== Phase Reuse Test (two-phase barrier) ===\n");
    barrier_phase_reuse<<<grid_size, BLOCK_SIZE, smem_phase>>>(d_dst_phase, d_src, d_b, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_phase_dst, d_dst_phase, N * sizeof(float), cudaMemcpyDeviceToHost);

    match = true;
    for (int i = 0; i < N; i++) {
        if (h_phase_dst[i] != h_expected[i]) {
            match = false;
            printf("Phase mismatch at %d: got=%f expected=%f\n", i, h_phase_dst[i], h_expected[i]);
            break;
        }
    }
    printf("Phase reuse results %s\n", match ? "MATCH" : "MISMATCH");

    free(h_src); free(h_dst_barrier); free(h_dst_flag);
    free(h_a); free(h_b); free(h_phase_dst); free(h_expected);
    cudaFree(d_src); cudaFree(d_dst_barrier); cudaFree(d_dst_flag);
    cudaFree(d_dst_phase); cudaFree(d_b);
}

int main() {
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    if (prop.major < 8) {
        printf("This demo requires SM80+ (Ampere or newer) for cuda::barrier\n");
        printf("Current GPU: %s (compute %d.%d)\n", prop.name, prop.major, prop.minor);
        return 1;
    }

    printf("Running on: %s (compute %d.%d)\n\n", prop.name, prop.major, prop.minor);

    run_barrier_test();

    return 0;
}
