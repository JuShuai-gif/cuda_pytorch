/**
 * FlashAttention V1 - Step-by-step CUDA implementation.
 *
 * This file implements the full FlashAttention V1 forward pass.
 *
 * Algorithm:
 *   for each Q block:
 *     for each KV block:
 *       S = Q_i @ K_j^T        (on shared memory)
 *       P = exp(S - m_new)     (online softmax)
 *       O_i = O_i * rescale + P @ V_j
 *
 * Key optimizations:
 *   - No O(N^2) intermediate matrices written to HBM
 *   - Online softmax avoids two passes over the attention scores
 *   - Shared memory tiling amortizes HBM reads of K and V
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ----------------------------------------------------------------------
// Configuration
// ----------------------------------------------------------------------
// Block sizes for tiling. Tuned for A100 (108 SMs, 164KB shared memory per SM).
constexpr int Br = 64;   // Q tile size (rows)
constexpr int Bc = 64;   // KV tile size (columns)
constexpr int Bd = 64;   // Head dimension tile (must match d)

// ----------------------------------------------------------------------
// Online Softmax Helpers (device functions)
// ----------------------------------------------------------------------
__device__ __forceinline__ float safe_exp(float x) {
    return __expf(fminf(x, 80.0f));  // clamp to avoid overflow
}

// ----------------------------------------------------------------------
// Step 1: Tiled S = Q_i @ K_j^T  (computed in shared memory)
//
// Each thread block computes one Br x Bc tile of the attention matrix.
// Thread layout: Br rows, Bc/warp cols per warp.
//
// Grid: (ceil(N/Br), ceil(N/Bc)) blocks
// Block: Br x (Bc/32) threads
// ----------------------------------------------------------------------
__global__ void step1_tiled_matmul(
    const float* __restrict__ Q,   // [N, d]
    const float* __restrict__ K,   // [N, d]
    float* __restrict__ S,         // [N, N] output (for verification)
    int N, int d)
{
    // Block indices: which tiles of Q and K we process
    int block_row = blockIdx.x;  // Q tile index
    int block_col = blockIdx.y;  // K tile index

    // Thread indices
    int tx = threadIdx.x;

    // Shared memory tiles
    extern __shared__ float smem[];
    float* Q_tile = smem;                    // [Br, d]
    float* K_tile = smem + Br * Bd;          // [d, Bc]

    // Load Q_tile cooperatively
    int q_row = block_row * Br;
    for (int i = tx; i < Br * Bd; i += blockDim.x) {
        int row = i / Bd;
        int col = i % Bd;
        int global_row = q_row + row;
        if (global_row < N && col < d) {
            Q_tile[i] = Q[global_row * d + col];
        } else {
            Q_tile[i] = 0.0f;
        }
    }

    // Load K_tile cooperatively (as K^T, so rows=d, cols=Bc)
    int k_col_base = block_col * Bc;
    for (int i = tx; i < Bd * Bc; i += blockDim.x) {
        int row = i / Bc;  // d dimension
        int col = i % Bc;  // K sequence position
        int global_k = k_col_base + col;
        if (global_k < N && row < d) {
            K_tile[i] = K[global_k * d + row];  // K[col, row] for K^T
        } else {
            K_tile[i] = 0.0f;
        }
    }
    __syncthreads();

    // Compute S[block_row*Br + local_row, block_col*Bc + local_col]
    for (int local_row = 0; local_row < Br; ++local_row) {
        int global_s_row = q_row + local_row;
        if (global_s_row >= N) break;

        for (int local_col = 0; local_col < Bc; ++local_col) {
            int global_s_col = k_col_base + local_col;
            if (global_s_col >= N) break;

            float dot = 0.0f;
            for (int dd = 0; dd < d; ++dd) {
                dot += Q_tile[local_row * Bd + dd] * K_tile[dd * Bc + local_col];
            }
            S[global_s_row * N + global_s_col] = dot;
        }
    }
}

// ----------------------------------------------------------------------
// Step 2: Online Softmax
//
// Given an input vector x of size N, compute softmax(x) in one pass.
// Demonstrates the rescaling mechanism.
// ----------------------------------------------------------------------
__global__ void step2_online_softmax(
    const float* __restrict__ x,  // [N] input vector
    float* __restrict__ y,        // [N] output softmax
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0) return;  // Single thread for demo

    // Online softmax in one pass
    float m = -INFINITY;   // running max
    float l = 0.0f;         // running sum of exp(x - m)

    // Pass 1: compute m and l online
    for (int i = 0; i < N; ++i) {
        float xi = x[i];
        float m_new = fmaxf(m, xi);

        if (xi > m) {
            // Rescale old l
            l = l * expf(m - m_new) + expf(xi - m_new);
        } else {
            l = l + expf(xi - m_new);
        }
        m = m_new;
    }

    // Pass 2: apply softmax using final m and l
    float inv_l = 1.0f / l;
    for (int i = 0; i < N; ++i) {
        y[i] = expf(x[i] - m) * inv_l;
    }
}

// ----------------------------------------------------------------------
// Step 3: Full FlashAttention V1 Forward
//
// This is the complete algorithm:
//
// For each Q block (outer loop over Tr blocks):
//   1. Load Q_i [Br, d] into shared memory
//   2. Initialize O_i = 0, m_i = -inf, l_i = 0
//   3. For each KV block (inner loop over Tc blocks):
//       a. Load K_j [Bc, d], V_j [Bc, d] into shared memory
//       b. Compute S_ij = Q_i @ K_j^T   [Br, Bc]
//       c. Update running softmax stats (m, l)
//       d. Compute P_ij @ V_j and accumulate into O_i (with rescaling)
//   4. Normalize O_i = O_i / l_i
//   5. Write O_i to HBM
//
// Grid: Tr = ceil(N/Br) blocks in x dimension
// Block: Br threads (one per query row)
// ----------------------------------------------------------------------
__global__ void flash_attention_v1_fwd(
    const float* __restrict__ Q,   // [N, d]
    const float* __restrict__ K,   // [N, d]
    const float* __restrict__ V,   // [N, d]
    float* __restrict__ O,         // [N, d]
    int N, int d)
{
    int q_start = blockIdx.x * Br;          // Which Q block
    int local_row = threadIdx.x;             // Which row in this Q block
    int global_row = q_start + local_row;
    bool valid_row = (global_row < N);

    float scale = rsqrtf(static_cast<float>(d));

    // Shared memory layout: [Q_tile: Br*d, K_tile: Bc*d, V_tile: Bc*d]
    extern __shared__ float smem[];
    float* Q_tile = smem;
    float* K_tile = smem + Br * d;
    float* V_tile = smem + Br * d + Bc * d;

    // Load Q_tile cooperatively
    for (int i = local_row; i < Br * d; i += blockDim.x) {
        int row = i / d;
        int col = i % d;
        int g_row = q_start + row;
        Q_tile[i] = (g_row < N && col < d) ? Q[g_row * d + col] : 0.0f;
    }
    __syncthreads();

    // Per-row online softmax state (in registers)
    float m_i = -INFINITY;
    float l_i = 0.0f;

    // Per-row output accumulator (partial O_i)
    float O_i_acc[64];  // Assume d <= 64 (register array)
    for (int j = 0; j < d; ++j) O_i_acc[j] = 0.0f;

    int Tc = (N + Bc - 1) / Bc;

    // Inner loop: iterate over KV tiles
    for (int j = 0; j < Tc; ++j) {
        int kv_start = j * Bc;
        int Bc_actual = min(Bc, N - kv_start);

        // Load K_tile and V_tile cooperatively
        // Load K: [Bc_actual, d]
        for (int i = local_row; i < Bc_actual * d; i += blockDim.x) {
            int row = i / d;
            int col = i % d;
            K_tile[i] = K[(kv_start + row) * d + col];
        }
        // Load V: [Bc_actual, d]
        for (int i = local_row; i < Bc_actual * d; i += blockDim.x) {
            int row = i / d;
            int col = i % d;
            V_tile[i] = V[(kv_start + row) * d + col];
        }
        __syncthreads();

        if (valid_row) {
            // Compute S_ij = Q_tile[local_row] @ K_tile^T  → [Bc_actual]
            float S_ij[64];  // Per-key scores
            float m_prev = m_i;

            for (int k = 0; k < Bc_actual; ++k) {
                float dot = 0.0f;
                for (int dd = 0; dd < d; ++dd) {
                    dot += Q_tile[local_row * d + dd] * K_tile[k * d + dd];
                }
                S_ij[k] = dot * scale;
            }

            // Online softmax update
            float m_new = m_i;
            for (int k = 0; k < Bc_actual; ++k) {
                if (S_ij[k] > m_new) m_new = S_ij[k];
            }

            float l_new = l_i * expf(m_i - m_new);
            for (int k = 0; k < Bc_actual; ++k) {
                l_new += expf(S_ij[k] - m_new);
            }

            float rescale = expf(m_i - m_new);

            // Update output accumulator
            for (int dd = 0; dd < d; ++dd) {
                float accum = O_i_acc[dd] * rescale;
                for (int k = 0; k < Bc_actual; ++k) {
                    accum += expf(S_ij[k] - m_new) * V_tile[k * d + dd];
                }
                O_i_acc[dd] = accum;
            }

            // Update running stats
            m_i = m_new;
            l_i = l_new;
        }
        __syncthreads();
    }

    // Final normalization and write to HBM
    if (valid_row) {
        float inv_l = 1.0f / l_i;
        for (int dd = 0; dd < d; ++dd) {
            O[global_row * d + dd] = O_i_acc[dd] * inv_l;
        }
    }
}

// ----------------------------------------------------------------------
// Host helper: benchmark FlashAttention V1
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

static float time_kernel(
    void (*kernel)(const float*, const float*, const float*, float*, int, int),
    const dim3& grid, const dim3& block, size_t smem,
    const float* d_Q, const float* d_K, const float* d_V, float* d_O,
    int N, int d, int warmup, int iters)
{
    // Warmup
    for (int w = 0; w < warmup; ++w)
        kernel<<<grid, block, smem>>>(d_Q, d_K, d_V, d_O, N, d);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
        kernel<<<grid, block, smem>>>(d_Q, d_K, d_V, d_O, N, d);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= iters;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

int main() {
    printf("FlashAttention V1 - Step-by-Step Implementation\n");
    printf("%s\n", std::string(80, '=').c_str());

    // Sanity check on small N
    int N_small = 128, d = 64;

    std::vector<float> h_Q(N_small * d), h_K(N_small * d), h_V(N_small * d), h_O(N_small * d);
    for (auto& x : h_Q) x = static_cast<float>(rand()) / RAND_MAX;
    for (auto& x : h_K) x = static_cast<float>(rand()) / RAND_MAX;
    for (auto& x : h_V) x = static_cast<float>(rand()) / RAND_MAX;

    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, N_small * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, N_small * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, N_small * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, N_small * d * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), N_small * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), N_small * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), N_small * d * sizeof(float), cudaMemcpyHostToDevice));

    // Step 1: Tiled matmul
    dim3 grid_step1(1, 1);
    dim3 block_step1(256);
    size_t smem_step1 = (Br * Bd + Bd * Bc) * sizeof(float);

    // Step 3: FlashAttention V1
    int Tr = (N_small + Br - 1) / Br;
    dim3 grid_fa(Tr);
    dim3 block_fa(Br);
    size_t smem_fa = (Br * d + Bc * d * 2) * sizeof(float);

    float ms_fa = time_kernel(flash_attention_v1_fwd,
                               grid_fa, block_fa, smem_fa,
                               d_Q, d_K, d_V, d_O, N_small, d, 5, 50);

    printf("  N=%d, d=%d\n", N_small, d);
    printf("  FlashAttention V1:          %8.3f ms\n", ms_fa);
    printf("  Shared memory per block:    %8zu bytes\n", smem_fa);

    // Verify output is finite
    CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, N_small * d * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < N_small * d; ++i) {
        if (!isfinite(h_O[i])) { ok = false; break; }
    }
    printf("  Output check:               %s\n", ok ? "PASS" : "FAIL");

    // Benchmark across sizes
    printf("\n");
    printf("%-8s %-12s %-12s %-12s\n", "N", "Time(ms)", "BW_IO(GB/s)", "TFLOPS");
    printf("%s\n", std::string(50, '-').c_str());

    for (int N : {128, 256, 512, 1024, 2048, 4096}) {
        std::vector<float> hQ(N * d), hK(N * d), hV(N * d), hO(N * d);
        for (auto& x : hQ) x = static_cast<float>(rand()) / RAND_MAX;
        for (auto& x : hK) x = static_cast<float>(rand()) / RAND_MAX;
        for (auto& x : hV) x = static_cast<float>(rand()) / RAND_MAX;

        float *dQ, *dK, *dV, *dO;
        CUDA_CHECK(cudaMalloc(&dQ, N * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dK, N * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dV, N * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dO, N * d * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), N * d * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dK, hK.data(), N * d * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dV, hV.data(), N * d * sizeof(float), cudaMemcpyHostToDevice));

        int Tr = (N + Br - 1) / Br;
        dim3 g(Tr);
        dim3 b(Br);
        size_t sm = (Br * d + Bc * d * 2) * sizeof(float);

        float ms = time_kernel(flash_attention_v1_fwd, g, b, sm,
                                dQ, dK, dV, dO, N, d, 5, 20);

        // IO bytes: read Q(N*d), read K(Tc*Bc*d*Tr) approximate, read V similarly, write O(N*d)
        // FlashAttention reads K and V Tr times
        float io_bytes = (2.0f * N * d + 2.0f * N * d * Tr) * sizeof(float);
        float bw_io = (io_bytes / 1e9f) / (ms / 1000.0f);
        float flops = 4.0f * N * N * d;  // approximate
        float tflops = (flops / 1e12f) / (ms / 1000.0f);

        printf("%-8d %-12.3f %-12.1f %-12.3f\n", N, ms, bw_io, tflops);

        CUDA_CHECK(cudaFree(dQ));
        CUDA_CHECK(cudaFree(dK));
        CUDA_CHECK(cudaFree(dV));
        CUDA_CHECK(cudaFree(dO));

        // Stop if too slow or OOM
        if (N >= 4096) break;
    }

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));

    printf("\n%s\n", std::string(80, '=').c_str());
    printf("Key results:\n");
    printf("  - No NxN intermediate matrix written to HBM\n");
    printf("  - K and V are re-read Tr times but this is still << O(N^2)\n");
    printf("  - Shared memory tiling enables high arithmetic intensity\n");
    printf("  - Compare with Chapter 02 naive GPU - expect large speedup\n");

    return 0;
}
