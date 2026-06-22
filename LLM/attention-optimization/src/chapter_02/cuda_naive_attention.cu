/**
 * CUDA Naive Attention - Direct GPU port of the O(N^2) attention pattern.
 *
 * This kernel demonstrates WHY naive GPU attention is slow:
 * - Each thread block writes to global memory (HBM) for the S matrix
 * - S and P are read back from HBM, causing bandwidth bottleneck
 * - No use of shared memory to reduce global memory traffic
 *
 * Key learning: The bottleneck is NOT computation, it's MEMORY BANDWIDTH.
 * This is exactly what FlashAttention addresses.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ----------------------------------------------------------------------
// GPU Kernel: Naive Attention
//
// Grid:  (N, d_v)  threads  → each thread computes one output element
// Block: (TILE, TILE)
//
// Layout:
//   Thread (row, col) computes O[row][col]
//   It iterates through the entire sequence to accumulate:
//     O[row][col] = sum_j P[row][j] * V[j][col]
//   where P[row] = softmax(Q[row] @ K^T / sqrt(d_k))
//
//   This means EVERY thread has to:
//     1. Compute Q[row] @ K^T  (full N-dim dot product)
//     2. Compute softmax (needs max + sum across all K positions)
//     3. Compute P[row] @ V[:, col]
//
//   Steps 1 and 3 alone are O(N) per output element → O(N^2) total.
// ----------------------------------------------------------------------
__global__ void naive_attention_kernel(
    const float* __restrict__ Q,   // [N, d_k] row-major
    const float* __restrict__ K,   // [N, d_k] row-major
    const float* __restrict__ V,   // [N, d_v] row-major
    float* __restrict__ O,         // [N, d_v] row-major
    int N, int d_k, int d_v)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // which query row
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // which value column

    if (row >= N || col >= d_v) return;

    float scale = rsqrtf(static_cast<float>(d_k));

    // ---------- Step 1: Compute attention scores for this query ----------
    // S[row][j] = (1/sqrt(d_k)) * Q[row] @ K[j]  for all j
    // We need to compute all N scores to do softmax.
    // → This requires reading ALL of K (N * d_k floats) for EACH row.

    // Phase A: max for numerical stability
    float max_score = -INFINITY;
    for (int j = 0; j < N; ++j) {
        float dot = 0.0f;
        for (int m = 0; m < d_k; ++m) {
            dot += Q[row * d_k + m] * K[j * d_k + m];
        }
        float score = dot * scale;
        if (score > max_score) max_score = score;
    }

    // Phase B: exp and sum
    float sum_exp = 0.0f;
    for (int j = 0; j < N; ++j) {
        float dot = 0.0f;
        for (int m = 0; m < d_k; ++m) {
            dot += Q[row * d_k + m] * K[j * d_k + m];
        }
        float weight = expf(dot * scale - max_score);
        sum_exp += weight;
    }

    // ---------- Step 2: Weighted sum ----------
    float inv_sum = 1.0f / sum_exp;
    float accum = 0.0f;

    for (int j = 0; j < N; ++j) {
        float dot = 0.0f;
        for (int m = 0; m < d_k; ++m) {
            dot += Q[row * d_k + m] * K[j * d_k + m];
        }
        float score = dot * scale;
        float weight = expf(score - max_score) * inv_sum;
        accum += weight * V[j * d_v + col];
    }

    O[row * d_v + col] = accum;
}

// ----------------------------------------------------------------------
// Improved Kernel: Naive Attention with Shared Memory
//
// Same algorithm, but caches Q[row] in registers and loads K/V tiles
// through shared memory. This still computes QK^T 3 times per output
// element (phase A, B, C) but reduces global memory traffic for K and V.
//
// This is STILL O(N^2) memory traffic because each thread processes
// the full sequence independently. The REAL optimization requires
// decomposing the attention matrix computation across thread blocks
// and fusing operations (FlashAttention).
// ----------------------------------------------------------------------
#define SMEM_TILE_K 64

__global__ void naive_attention_smem_kernel(
    const float* __restrict__ Q,   // [N, d_k]
    const float* __restrict__ K,   // [N, d_k]
    const float* __restrict__ V,   // [N, d_v]
    float* __restrict__ O,         // [N, d_v]
    int N, int d_k, int d_v)
{
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N) return;
    const bool valid_col = col < d_v;

    // Cache Q[row] into registers
    extern __shared__ float smem[];
    float* K_tile = smem;
    // V_tile is indexed by [tile_row, output-column lane]. Each thread in
    // the block owns a different col, so a single V_tile[j] slot would race.
    float* V_tile = smem + SMEM_TILE_K * d_k;

    float scale = rsqrtf(static_cast<float>(d_k));

    // Phase A: Online max
    float max_score = -INFINITY;

    for (int j_start = 0; j_start < N; j_start += SMEM_TILE_K) {
        int tile_size = min(SMEM_TILE_K, N - j_start);

        // Cooperative load K[j_start : j_start+tile_size] into shared memory
        for (int t = 0; t < d_k; t += blockDim.x) {
            int k_idx = threadIdx.x + t;
            if (k_idx < d_k) {
                for (int jj = 0; jj < tile_size; ++jj) {
                    K_tile[jj * d_k + k_idx] = K[(j_start + jj) * d_k + k_idx];
                }
            }
        }
        __syncthreads();

        // Compute dot products for this tile
        for (int jj = 0; jj < tile_size; ++jj) {
            float dot = 0.0f;
            for (int m = 0; m < d_k; ++m) {
                dot += Q[row * d_k + m] * K_tile[jj * d_k + m];
            }
            float score = dot * scale;
            if (score > max_score) max_score = score;
        }
        __syncthreads();
    }

    // Phase B: sum of exp
    float sum_exp = 0.0f;
    for (int j_start = 0; j_start < N; j_start += SMEM_TILE_K) {
        int tile_size = min(SMEM_TILE_K, N - j_start);

        for (int t = 0; t < d_k; t += blockDim.x) {
            int k_idx = threadIdx.x + t;
            if (k_idx < d_k) {
                for (int jj = 0; jj < tile_size; ++jj) {
                    K_tile[jj * d_k + k_idx] = K[(j_start + jj) * d_k + k_idx];
                }
            }
        }
        __syncthreads();

        for (int jj = 0; jj < tile_size; ++jj) {
            float dot = 0.0f;
            for (int m = 0; m < d_k; ++m) {
                dot += Q[row * d_k + m] * K_tile[jj * d_k + m];
            }
            sum_exp += expf(dot * scale - max_score);
        }
        __syncthreads();
    }

    // Phase C: weighted sum (with V tile load)
    float inv_sum = 1.0f / sum_exp;
    float accum = 0.0f;

    for (int j_start = 0; j_start < N; j_start += SMEM_TILE_K) {
        int tile_size = min(SMEM_TILE_K, N - j_start);

        // Load K tile
        for (int t = 0; t < d_k; t += blockDim.x) {
            int k_idx = threadIdx.x + t;
            if (k_idx < d_k) {
                for (int jj = 0; jj < tile_size; ++jj) {
                    K_tile[jj * d_k + k_idx] = K[(j_start + jj) * d_k + k_idx];
                }
            }
        }

        // Load the full V tile for this output-column lane. V_tile is laid
        // out as [tile_row, lane] because each lane owns a different col.
        if (valid_col) {
            for (int jj = 0; jj < tile_size; ++jj) {
                V_tile[jj * blockDim.x + threadIdx.x] = V[(j_start + jj) * d_v + col];
            }
        }
        __syncthreads();

        for (int jj = 0; jj < tile_size; ++jj) {
            float dot = 0.0f;
            for (int m = 0; m < d_k; ++m) {
                dot += Q[row * d_k + m] * K_tile[jj * d_k + m];
            }
            float weight = expf(dot * scale - max_score) * inv_sum;
            accum += weight * V_tile[jj * blockDim.x + threadIdx.x];
        }
        __syncthreads();
    }

    if (valid_col) {
        O[row * d_v + col] = accum;
    }
}

// ----------------------------------------------------------------------
// Host benchmarking
// ----------------------------------------------------------------------
#define CUDA_CHECK(err)                                                    \
    do {                                                                   \
        cudaError_t e_ = (err);                                            \
        if (e_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(e_));                                \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

struct BenchResult {
    float time_ms;
    float bandwidth_gbs;
    float tflops;
};

static BenchResult run_kernel(void (*kernel)(const float*, const float*,
                                              const float*, float*,
                                              int, int, int),
                               const dim3& grid, const dim3& block,
                               size_t smem_bytes,
                               const float* d_Q, const float* d_K,
                               const float* d_V, float* d_O,
                               int N, int d_k, int d_v,
                               int warmup, int iters,
                               float total_bytes, float total_flops)
{
    // Warmup
    for (int w = 0; w < warmup; ++w) {
        kernel<<<grid, block, smem_bytes>>>(d_Q, d_K, d_V, d_O, N, d_k, d_v);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        kernel<<<grid, block, smem_bytes>>>(d_Q, d_K, d_V, d_O, N, d_k, d_v);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    elapsed_ms /= iters;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float bw = (total_bytes / 1e9f) / (elapsed_ms / 1000.0f);   // GB/s
    float tf = (total_flops / 1e12f) / (elapsed_ms / 1000.0f);  // TFLOPS

    return {elapsed_ms, bw, tf};
}

#ifndef ATTENTION_CH02_NO_MAIN
int main() {
    // Test configurations
    struct Config {
        int N;
        int d_k;
        int d_v;
    };

    std::vector<Config> configs = {
        {128,  64, 64},
        {256,  64, 64},
        {512,  64, 64},
        {1024, 64, 64},
        {2048, 64, 64},
        {4096, 64, 64},
    };

    printf("CUDA Naive Attention Benchmark\n");
    printf("%s\n", std::string(80, '=').c_str());
    printf("%-8s %-10s %-10s %-10s %-10s\n",
           "N", "Time(ms)", "BW(GB/s)", "TFLOPS", "Kernel");
    printf("%s\n", std::string(80, '-').c_str());

    for (const auto& cfg : configs) {
        int N = cfg.N, d_k = cfg.d_k, d_v = cfg.d_v;

        // Allocate host memory
        std::vector<float> h_Q(N * d_k), h_K(N * d_k), h_V(N * d_v), h_O(N * d_v);
        for (auto& x : h_Q) x = static_cast<float>(rand()) / RAND_MAX;
        for (auto& x : h_K) x = static_cast<float>(rand()) / RAND_MAX;
        for (auto& x : h_V) x = static_cast<float>(rand()) / RAND_MAX;

        // Allocate device memory
        float *d_Q, *d_K, *d_V, *d_O;
        CUDA_CHECK(cudaMalloc(&d_Q, N * d_k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K, N * d_k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_V, N * d_v * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_O, N * d_v * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), N * d_k * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), N * d_k * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), N * d_v * sizeof(float), cudaMemcpyHostToDevice));

        // FLOPs and bytes moved
        // QK^T: 2*N*N*d_k FLOPs, reads Q(N*d_k)+K(N*d_k), writes S(N*N)
        // Softmax: ~4*N*N FLOPs, reads S(N*N), writes P(N*N)
        // PV: 2*N*N*d_v FLOPs, reads P(N*N)+V(N*d_v), writes O(N*d_v)
        float total_flops = 2.0f * N * N * d_k + 4.0f * N * N + 2.0f * N * N * d_v;
        // Bytes: Q+K+V read once, S/P read/write, O written
        float total_bytes = (N * d_k + N * d_k + N * d_v + 2 * N * N + N * d_v) * sizeof(float);

        dim3 block_global(16, 16);        // 256 threads
        dim3 grid_global((d_v + 15) / 16, (N + 15) / 16);

        dim3 block_smem(16, 1);           // 1D block along d_v
        dim3 grid_smem((d_v + 15) / 16, N);
        size_t smem_bytes = SMEM_TILE_K * (d_k + block_smem.x) * sizeof(float);

        // Run both kernels
        auto res_global = run_kernel(naive_attention_kernel,
                                     grid_global, block_global, 0,
                                     d_Q, d_K, d_V, d_O, N, d_k, d_v,
                                     2, 10, total_bytes, total_flops);

        // Only run SMEM kernel for smaller N (shared memory limited)
        if (smem_bytes < 48 * 1024) {
            auto res_smem = run_kernel(naive_attention_smem_kernel,
                                       grid_smem, block_smem, smem_bytes,
                                       d_Q, d_K, d_V, d_O, N, d_k, d_v,
                                       2, 10, total_bytes, total_flops);

            printf("%-8d %-10.3f %-10.1f %-10.3f %-10s\n",
                   N, res_global.time_ms, res_global.bandwidth_gbs,
                   res_global.tflops, "Global");
            printf("%-8d %-10.3f %-10.1f %-10.3f %-10s\n",
                   N, res_smem.time_ms, res_smem.bandwidth_gbs,
                   res_smem.tflops, "SharedMem");
        } else {
            printf("%-8d %-10.3f %-10.1f %-10.3f %-10s\n",
                   N, res_global.time_ms, res_global.bandwidth_gbs,
                   res_global.tflops, "Global");
        }

        CUDA_CHECK(cudaFree(d_Q));
        CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_V));
        CUDA_CHECK(cudaFree(d_O));
    }

    printf("%s\n", std::string(80, '=').c_str());
    printf("\nAnalysis:\n");
    printf("  - The O(N^2) attention matrix is the bandwidth bottleneck\n");
    printf("  - Even with shared memory, we recompute QK^T 3 times per output\n");
    printf("  - A100 HBM: ~2TB/s → S matrix (N=4096) read/write takes ~0.07ms\n");
    printf("  - Real bottleneck: repeated reads of K matrix by every thread\n");
    printf("  - Solution: Tiling + Kernel Fusion (FlashAttention, Ch04)\n");

    return 0;
}
#endif  // ATTENTION_CH02_NO_MAIN
