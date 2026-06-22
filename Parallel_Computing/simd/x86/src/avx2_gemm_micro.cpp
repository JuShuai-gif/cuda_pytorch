/*****************************************************************
 * AVX2 GEMM Micro-Kernel: C[M][N] += A[M][K] * B[K][N]
 *
 * This is the CANONICAL industrial SIMD workload. Every HPC/ML
 * library (OpenBLAS, BLIS, MKL, XNNPACK) revolves around highly
 * optimized GEMM micro-kernels.
 *
 * Micro-kernel size: 8x8 (8 rows of C, 8 columns of C per call)
 *
 * Register blocking:
 *   - 8 ymm registers as C accumulators (c0..c7, one per row of C)
 *   - Each ymm holds all 8 columns for that row
 *   - 1 ymm for broadcasting A values (reused per row)
 *   - 1 ymm for loading B values
 *   - Total: ~10 of 16 ymm registers used (63%), leaving headroom
 *     for further unrolling (e.g. 2x on K or 2x on rows)
 *
 * Inner loop (per k-iteration):                                *
 *   b_vec = load B_packed[k][0:8]       (1 contiguous load)    *
 *   For each row p in 0..7:                                     *
 *     c[p] = FMA(broadcast(A[p][k]), b_vec, c[p])              *
 *   → 1 B load + 8 A broadcasts + 8 FMAs = 16 flops / 9 loads  *
 *   → Arithmetic intensity ≈ 16 / (9 × 4 bytes) = 0.44 flop/byte
 *
 * Packing strategy:
 *   A is already row-major (K as inner dim) → contiguous along k
 *   B is packed so B_packed[k * NR + j] = B[k][col + j]
 *     This makes B access contiguous for SIMD loads:
 *     _mm256_loadu_ps(&B_packed[k * NR]) loads 8 B values at once.
 *
 * Cache blocking (outer tiling):
 *   The micro-kernel is called within a larger tiling framework.
 *   For M=N=K=64, the entire problem fits in L1 cache, so simple
 *   tiling over M and N in steps of MR=8, NR=8 is sufficient.
 *   Production libraries add KC (K panel size) tiling to keep
 *   A and B panels in L2/L3 cache for larger problems.
 *
 * Why 8x8 on AVX2:
 *   - Each ymm holds 8 f32 values → NR=8 is the natural column block
 *   - MR=8 gives a square 8x8 kernel, balancing rows and columns
 *   - BLIS uses 6x8 on Haswell (MR=6 for additional unrolling headroom,
 *     NR=8 to match the SIMD width). Our 8x8 is slightly more aggressive
 *     but demonstrates the full register allocation.
 *
 * Scaling to AVX-512:
 *   - zmm holds 16 f32 values → NR=16 is the natural column block
 *   - Kernel becomes 16x16 or 16x8 with 32 zmm registers available
 *   - 2x the FMA throughput (512-bit vs 256-bit) with 2x the registers
 *
 * Key teaching points:
 *   1. Register blocking reduces memory traffic from O(MNK) to O(MK+KN)
 *   2. Packing enables contiguous access, avoiding cache line splits
 *   3. The inner loop is compute-bound: 16 flops with only 9 loads
 *   4. Broadcast + FMA is the fundamental GEMM micro-op on SIMD
 *****************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include "../../common/benchmark.h"
#include "../../common/check.h"
#include "../../common/random_data.h"
#include "../../common/aligned_buffer.h"
#include "../../common/cpu_features.h"

/* ---- Problem size: small enough to run fast, large enough to show blocking ---- */

static const int M = 64;
static const int N = 64;
static const int K = 64;

/* Micro-tile dimensions */
static const int MR = 8;  /* rows per micro-kernel call */
static const int NR = 8;  /* columns per micro-kernel call */

/* Estimated single-core peak: 2 FMA units × 8 flops/cycle × ~3 GHz ≈ 48 GFLOPS.
 * Actual achievable depends on frequency, cache hierarchy, and memory bandwidth. */
static const double THEORETICAL_PEAK_GFLOPS = 48.0;

/* =================================================================
 * Horizontal reduction: sum all 8 lanes of __m256 into a float
 *
 * Strategy: swap lo/hi 128-bit halves → add → 2x hadd → extract.
 * This distributes port pressure (permute on port 5, add on 0/1)
 * instead of saturating port 5 with a chain of hadd instructions.
 * ================================================================= */

static inline float hsum_ps(__m256 v) {
    /* v = [a0, a1, a2, a3,  b0, b1, b2, b3] */
    __m256 swapped = _mm256_permute2f128_ps(v, v, 0x01);
    /* swapped = [b0, b1, b2, b3,  a0, a1, a2, a3] */
    v = _mm256_add_ps(v, swapped);
    /* v = [a0+b0, a1+b1, a2+b2, a3+b3,  same again] */
    v = _mm256_hadd_ps(v, v);
    v = _mm256_hadd_ps(v, v);
    return _mm256_cvtss_f32(v);
}

/* =================================================================
 * Level 0: Scalar Baseline
 *
 * Standard triple-loop GEMM: C[i][j] += sum_k A[i][k] * B[k][j]
 * Complexity: O(MNK) flops, O(MNK) memory loads (no reuse).
 * ================================================================= */

static void scalar_gemm(int M_p, int N_p, int K_p,
                        const float* A, int lda,
                        const float* B, int ldb,
                        float* C, int ldc) {
    for (int i = 0; i < M_p; i++) {
        for (int j = 0; j < N_p; j++) {
            float sum = C[i * ldc + j];
            for (int k = 0; k < K_p; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}

/* =================================================================
 * Level 1: Naive SIMD
 *
 * Vectorize the K dimension using AVX2 FMA. For each (i,j) pair,
 * load 8 A values (contiguous), gather 8 B values (strided by N),
 * and accumulate. The B gather is the bottleneck — 8 separate
 * scalar loads, no SIMD load possible due to non-unit stride.
 *
 * This demonstrates WHY packing is necessary:
 *   - Without packing, one operand is always strided
 *   - __m256_set_ps() compiles into 8 scalar movss + inserteps
 *   - Memory access pattern causes TLB/cache thrashing
 * ================================================================= */

static void gemm_naive_simd(int M_p, int N_p, int K_p,
                            const float* A, int lda,
                            const float* B, int ldb,
                            float* C, int ldc) {
    for (int i = 0; i < M_p; i++) {
        for (int j = 0; j < N_p; j++) {
            __m256 acc = _mm256_setzero_ps();
            int k = 0;
            for (; k + 8 <= K_p; k += 8) {
                /* A[i][k:k+8] — contiguous, single SIMD load */
                __m256 a_vec = _mm256_loadu_ps(&A[i * lda + k]);

                /* B[k:k+8][j] — strided by N_p, requires manual gather.
                 * _mm256_set_ps elements are in reverse order (hi to lo lane). */
                __m256 b_vec = _mm256_set_ps(
                    B[(k + 7) * ldb + j],
                    B[(k + 6) * ldb + j],
                    B[(k + 5) * ldb + j],
                    B[(k + 4) * ldb + j],
                    B[(k + 3) * ldb + j],
                    B[(k + 2) * ldb + j],
                    B[(k + 1) * ldb + j],
                    B[(k + 0) * ldb + j]
                );

                acc = _mm256_fmadd_ps(a_vec, b_vec, acc);
            }
            float sum = hsum_ps(acc);
            for (; k < K_p; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] += sum;
        }
    }
}

/* =================================================================
 * Packing helpers
 *
 * pack_B_Kx8: Convert B (row-major KxN) to B_packed (K rows, NR cols,
 * contiguous along columns). This transposes the column panel so that
 * _mm256_loadu_ps(&B_packed[k * NR]) loads 8 consecutive column
 * values at row k.
 *
 * pack_A_8xK: Copy 8 consecutive rows of A (already row-major m×K)
 * into a contiguous MR×K buffer for the micro-kernel.
 * A is already in the right layout, but copying isolates the panel
 * and ensures alignment. For simplicity we avoid actual copying when
 * A's stride (lda) equals K and data is aligned; we point directly.
 * ================================================================= */

static void pack_B_Kx8(float* B_packed,
                       const float* B, int ldb,
                       int row_start, int col_start,
                       int K_len, int NR_use) {
    for (int k = 0; k < K_len; k++) {
        for (int j = 0; j < NR_use; j++) {
            B_packed[k * NR + j] = B[(k + row_start) * ldb + col_start + j];
        }
    }
}

static void pack_A_8xK(float* A_packed,
                       const float* A, int lda,
                       int row_start, int col_start,
                       int K_len, int MR_use) {
    for (int i = 0; i < MR_use; i++) {
        for (int k = 0; k < K_len; k++) {
            A_packed[i * K_len + k] = A[(row_start + i) * lda + col_start + k];
        }
    }
}

/* =================================================================
 * Level 2: Packed SIMD
 *
 * After packing B, we can process C row by row, 8 columns at a time.
 * For each row i and column block j:
 *   c_acc = load C[i][j:j+7]   (1 SIMD load)
 *   For k in 0..K:
 *     a_val = A[i][k]          (broadcast from 1 scalar load)
 *     b_vec = load B_packed[k][0:7]  (1 SIMD load, contiguous)
 *     c_acc = FMA(broadcast(a_val), b_vec, c_acc)
 *   store C[i][j:j+7]          (1 SIMD store)
 *
 * This eliminates the gather in B, reducing memory traffic significantly.
 * But C is still loaded/stored per row — the register-blocked kernel
 * eliminates this by keeping all 8 rows in registers.
 * ================================================================= */

static void gemm_packed_simd(int M_p, int N_p, int K_p,
                             const float* A, int lda,
                             const float* B, int ldb,
                             float* C, int ldc) {
    float* B_packed = ALIGNED_ALLOC(float, (size_t)K_p * NR, 32);

    for (int mi = 0; mi < M_p; mi += MR) {
        int mr_use = (mi + MR <= M_p) ? MR : (M_p - mi);
        for (int nj = 0; nj < N_p; nj += NR) {
            int nr_use = (nj + NR <= N_p) ? NR : (N_p - nj);

            /* Pack B panel: K x NR, contiguous in column direction */
            pack_B_Kx8(B_packed, B, ldb, 0, nj, K_p, nr_use);

            /* Compute the mr_use × nr_use tile */
            for (int i = 0; i < mr_use; i++) {
                __m256 c_acc = _mm256_loadu_ps(&C[(mi + i) * ldc + nj]);
                for (int k = 0; k < K_p; k++) {
                    __m256 b_vec = _mm256_loadu_ps(&B_packed[k * NR]);
                    __m256 a_brd = _mm256_set1_ps(A[(mi + i) * lda + k]);
                    c_acc = _mm256_fmadd_ps(a_brd, b_vec, c_acc);
                }
                _mm256_storeu_ps(&C[(mi + i) * ldc + nj], c_acc);
            }
        }
    }

    ALIGNED_FREE(B_packed);
}

/* =================================================================
 * Level 3: Register-Blocked 8x8 Micro-Kernel
 *
 * This is the heart of the optimization. The micro-kernel computes:
 *   C[0:8][0:8] += A_packed[0:8][K] * B_packed[K][0:8]
 *
 * Register allocation (AVX2: 16 ymm registers total):
 *   c0..c7: 8 ymm accumulators for the 8 rows of C    (8 regs)
 *   b_vec:  1 ymm for loading B_packed[k][0:7]        (1 reg)
 *   a_brd:  1 ymm reused for broadcasting A values     (1 reg)
 *   ---------------------------------------------------
 *   Total: 10 registers (63% of ymm register file)
 *
 * Inner loop analysis (per k-iteration):
 *   1 load from B_packed  (256-bit contiguous)
 *   8 loads from A_packed (each a scalar broadcast, _mm256_set1_ps)
 *   8 FMA instructions    (_mm256_fmadd_ps, 2 uops each on Skylake+)
 *   = 9 loads + 8 FMAs = 17 instructions
 *
 * Arithmetic intensity (AI):
 *   flops  = 16 (8 FMAs × 2 ops via fused multiply-add)
 *   bytes  = 9 × 32 bits = 36 bytes loaded
 *   AI     = 16 / 36 ≈ 0.44 flops/byte
 *
 * Compare to naive triple-loop:
 *   Each scalar iteration: 2 loads (A[i][k], B[k][j]) + 1 FMA
 *   AI = 2 / 8 = 0.25 flops/byte (and 8× more index computations)
 *
 * Operation intensity of the inner loop:
 *   The inner loop issues 8 FMAs for every k. With K=64, each
 *   micro-kernel call does 512 FMAs = 1024 flops, reusing each
 *   A value 8 times and each B value 8 times.
 *
 * Comparison to BLIS on Haswell:
 *   BLIS uses 6×8 (MR=6, NR=8) with similar strategy:
 *   - 6 C accumulators instead of 8 (more headroom for unrolling)
 *   - Same B load pattern, same broadcast+FMA pattern for A
 *   - The 8×8 kernel here is slightly more aggressive on register usage
 * ================================================================= */

static void gemm_micro_8x8(int K_p,
                           const float* A_packed, int lda,
                           const float* B_packed, int ldb,
                           float* C, int ldc) {
    const float* a = A_packed;
    const float* b = B_packed;

    /* Load initial values from C into 8 accumulator registers */
    __m256 c0 = _mm256_loadu_ps(&C[0 * ldc]);
    __m256 c1 = _mm256_loadu_ps(&C[1 * ldc]);
    __m256 c2 = _mm256_loadu_ps(&C[2 * ldc]);
    __m256 c3 = _mm256_loadu_ps(&C[3 * ldc]);
    __m256 c4 = _mm256_loadu_ps(&C[4 * ldc]);
    __m256 c5 = _mm256_loadu_ps(&C[5 * ldc]);
    __m256 c6 = _mm256_loadu_ps(&C[6 * ldc]);
    __m256 c7 = _mm256_loadu_ps(&C[7 * ldc]);

    for (int k = 0; k < K_p; k++) {
        /*
         * Load 8 B values from the packed B panel.
         * B_packed[k * ldb + j] for j=0..7 is contiguous in memory
         * because ldb = NR = 8, and the packing function stores
         * B values for row k and all NR columns contiguously.
         */
        __m256 b_vec = _mm256_loadu_ps(&b[k * ldb]);

        /*
         * For each of the 8 rows of A, broadcast the scalar A value
         * across all 8 SIMD lanes, then FMA into the corresponding
         * C accumulator. This is the canonical BLIS-style inner loop:
         *
         *   c[p] += A[p][k] * B[k][0:NR]
         *
         * where the multiply is implicit in FMA: c = a * b + c
         */
        c0 = _mm256_fmadd_ps(_mm256_set1_ps(a[0 * lda + k]), b_vec, c0);
        c1 = _mm256_fmadd_ps(_mm256_set1_ps(a[1 * lda + k]), b_vec, c1);
        c2 = _mm256_fmadd_ps(_mm256_set1_ps(a[2 * lda + k]), b_vec, c2);
        c3 = _mm256_fmadd_ps(_mm256_set1_ps(a[3 * lda + k]), b_vec, c3);
        c4 = _mm256_fmadd_ps(_mm256_set1_ps(a[4 * lda + k]), b_vec, c4);
        c5 = _mm256_fmadd_ps(_mm256_set1_ps(a[5 * lda + k]), b_vec, c5);
        c6 = _mm256_fmadd_ps(_mm256_set1_ps(a[6 * lda + k]), b_vec, c6);
        c7 = _mm256_fmadd_ps(_mm256_set1_ps(a[7 * lda + k]), b_vec, c7);
    }

    /* Store the 8 accumulator registers back to C */
    _mm256_storeu_ps(&C[0 * ldc], c0);
    _mm256_storeu_ps(&C[1 * ldc], c1);
    _mm256_storeu_ps(&C[2 * ldc], c2);
    _mm256_storeu_ps(&C[3 * ldc], c3);
    _mm256_storeu_ps(&C[4 * ldc], c4);
    _mm256_storeu_ps(&C[5 * ldc], c5);
    _mm256_storeu_ps(&C[6 * ldc], c6);
    _mm256_storeu_ps(&C[7 * ldc], c7);
}

/* =================================================================
 * Level 4: Tiled GEMM using the 8x8 micro-kernel
 *
 * Outer tiling loop: iterate over M in steps of MR, N in steps of NR.
 * For each tile:
 *   1. Pack A panel (MR × K) into contiguous buffer
 *   2. Pack B panel (K × NR) into contiguous buffer
 *   3. Call gemm_micro_8x8 to compute C[MR][NR] += A × B
 *
 * This demonstrates the full BLIS-style approach. For small matrices
 * (M=N=K=64), cache blocking on K is unnecessary. Production libraries
 * add K-tiling (KC blocks) to keep A and B panels in fixed-size
 * L2/L3 cache buffers for arbitrary problem sizes.
 * ================================================================= */

static void gemm_micro_tiled(int M_p, int N_p, int K_p,
                             const float* A, int lda,
                             const float* B, int ldb,
                             float* C, int ldc) {
    float* A_packed = ALIGNED_ALLOC(float, (size_t)MR * (size_t)K_p, 32);
    float* B_packed = ALIGNED_ALLOC(float, (size_t)K_p * (size_t)NR, 32);

    for (int mi = 0; mi < M_p; mi += MR) {
        int mr_use = (mi + MR <= M_p) ? MR : (M_p - mi);

        /* Pack A panel: mr_use rows × K_p columns */
        pack_A_8xK(A_packed, A, lda, mi, 0, K_p, mr_use);

        for (int nj = 0; nj < N_p; nj += NR) {
            int nr_use = (nj + NR <= N_p) ? NR : (N_p - nj);

            /* Pack B panel: K_p rows × nr_use columns */
            pack_B_Kx8(B_packed, B, ldb, 0, nj, K_p, nr_use);

            /* Call the micro-kernel on this 8×8 (or smaller) tile */
            gemm_micro_8x8(K_p, A_packed, K_p,
                           B_packed, NR,
                           &C[mi * ldc + nj], ldc);
        }
    }

    ALIGNED_FREE(A_packed);
    ALIGNED_FREE(B_packed);
}

/* =================================================================
 * Compute GFLOPS
 * ================================================================= */

static double compute_gflops(double elapsed_ns) {
    /* 2 × M × N × K multiply-adds = 2 × M × N × K floating-point ops */
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double seconds = elapsed_ns * 1e-9;
    return (seconds > 0.0) ? flops / seconds / 1e9 : 0.0;
}

/* =================================================================
 * main
 * ================================================================= */

int main() {
    cpu_print_features();

    if (!cpu_has_avx2()) {
        printf("AVX2 not supported on this CPU. Exiting.\n");
        return 1;
    }

    printf("\n===== AVX2 GEMM Micro-Kernel (M=N=K=%d) =====\n\n", M);
    printf("Micro-tile: MR=%d × NR=%d | "
           "Total flops per GEMM: %d\n\n",
           MR, NR, 2 * M * N * K);

    /* ---- Allocate matrices ---- */

    float* A = ALIGNED_ALLOC(float, (size_t)M * (size_t)K, 32);
    float* B = ALIGNED_ALLOC(float, (size_t)K * (size_t)N, 32);
    float* C = ALIGNED_ALLOC(float, (size_t)M * (size_t)N, 32);

    /* Workspace: ref_c holds scalar result, C_work is reused for each variant */
    float* ref_c = ALIGNED_ALLOC(float, (size_t)M * (size_t)N, 32);
    float* C_work = ALIGNED_ALLOC(float, (size_t)M * (size_t)N, 32);

    if (!A || !B || !C || !ref_c || !C_work) {
        fprintf(stderr, "Allocation failed.\n");
        return 1;
    }

    /* ---- Fill with random data in [-1, 1] ---- */

    rand_xorshift64_seed(42);
    fill_random_f32(A, (size_t)M * (size_t)K);
    rand_xorshift64_seed(99);
    fill_random_f32(B, (size_t)K * (size_t)N);

    /* Initialize C to zero (we compute C += A*B, starting from zero) */
    memset(C, 0, (size_t)M * (size_t)N * sizeof(float));
    memset(ref_c, 0, (size_t)M * (size_t)N * sizeof(float));

    /* ---- Correctness Verification ---- */

    printf("--- Correctness ---\n");

    /* Compute scalar reference: C_ref = A * B (starting from zero C) */
    scalar_gemm(M, N, K, A, K, B, N, ref_c, N);

    /*
     * For each variant, copy C=0, compute C += A*B, compare against ref_c.
     * Floating-point addition is non-associative; different summation orders
     * (scalar vs SIMD vs packed vs micro-kernel) produce slightly different
     * results. Tolerance accounts for K accumulations per output element.
     */
    const float tol = 5e-4f * (float)K;  /* ~0.032 for K=64 */

    /* Verify naive SIMD */
    memset(C_work, 0, (size_t)M * (size_t)N * sizeof(float));
    gemm_naive_simd(M, N, K, A, K, B, N, C_work, N);
    CHECK_NEAR_ARRAY(C_work, ref_c, (size_t)M * (size_t)N, tol,
                     "gemm_naive_simd matches scalar");

    /* Verify packed SIMD */
    memset(C_work, 0, (size_t)M * (size_t)N * sizeof(float));
    gemm_packed_simd(M, N, K, A, K, B, N, C_work, N);
    CHECK_NEAR_ARRAY(C_work, ref_c, (size_t)M * (size_t)N, tol,
                     "gemm_packed_simd matches scalar");

    /* Verify micro-tiled */
    memset(C_work, 0, (size_t)M * (size_t)N * sizeof(float));
    gemm_micro_tiled(M, N, K, A, K, B, N, C_work, N);
    CHECK_NEAR_ARRAY(C_work, ref_c, (size_t)M * (size_t)N, tol,
                     "gemm_micro_tiled matches scalar");

    /* ---- Benchmark ---- */

    printf("\n--- Benchmark Results ---\n");
    printf("SIMD width: 256-bit (8× f32 per ymm register)\n");
    printf("FMA throughput: 2 units × 8 flops/cycle "
           "(theoretical peak: %.0f GFLOPS/core)\n\n", THEORETICAL_PEAK_GFLOPS);

    /*
     * Memory bytes: A(M×K) + B(K×N) + C read(M×N) + C write(M×N)
     * For M=N=K=64: (4096 + 4096 + 4096 + 4096) × 4 = 65536 bytes
     */
    const size_t nelem = (size_t)M * (size_t)N;
    const size_t bytes_rw = ((size_t)M * (size_t)K
                           + (size_t)K * (size_t)N
                           + (size_t)M * (size_t)N * 2) * sizeof(float);

    benchmark_result_t results[4];
    memset(results, 0, sizeof(results));

    /*
     * Each BENCH_COMPUTE warms up the function (3×), then runs iters times
     * taking the minimum wall-clock time. We use volatile to prevent the
     * compiler from optimizing away the computation.
     */

    /* Scalar baseline */
    {
        float* A_b = A; float* B_b = B; float* C_b = C_work;
        BENCH_COMPUTE(
            memset(C_b, 0, nelem * sizeof(float));
            volatile float* _p = C_b; (void)_p;
            scalar_gemm(M, N, K, A_b, K, B_b, N, C_b, N);
            volatile float* _q = C_b; (void)_q;,
            nelem, bytes_rw, 30, results[0]);
        results[0].name = "scalar";
    }

    /* Naive SIMD */
    {
        float* A_b = A; float* B_b = B; float* C_b2 = C_work;
        BENCH_COMPUTE(
            memset(C_b2, 0, nelem * sizeof(float));
            gemm_naive_simd(M, N, K, A_b, K, B_b, N, C_b2, N);
            volatile float* _q2 = C_b2; (void)_q2;,
            nelem, bytes_rw, 30, results[1]);
        results[1].name = "naive_simd";
    }

    /* Packed SIMD */
    {
        float* A_b = A; float* B_b = B; float* C_b3 = C_work;
        BENCH_COMPUTE(
            memset(C_b3, 0, nelem * sizeof(float));
            gemm_packed_simd(M, N, K, A_b, K, B_b, N, C_b3, N);
            volatile float* _q3 = C_b3; (void)_q3;,
            nelem, bytes_rw, 30, results[2]);
        results[2].name = "packed_simd";
    }

    /* Micro-tiled (register-blocked) */
    {
        float* A_b = A; float* B_b = B; float* C_b4 = C_work;
        BENCH_COMPUTE(
            memset(C_b4, 0, nelem * sizeof(float));
            gemm_micro_tiled(M, N, K, A_b, K, B_b, N, C_b4, N);
            volatile float* _q4 = C_b4; (void)_q4;,
            nelem, bytes_rw, 30, results[3]);
        results[3].name = "micro_tiled";
    }

    bench_report(results, 4);

    /* ---- GFLOPS Analysis ---- */

    printf("--- GFLOPS Analysis ---\n\n");
    for (int i = 0; i < 4; i++) {
        double gflops = compute_gflops(results[i].elapsed_ns);
        double pct_peak = (gflops / THEORETICAL_PEAK_GFLOPS) * 100.0;
        printf("  %-20s  %8.3f GFLOPS  (%5.1f%% of peak)\n",
               results[i].name, gflops, pct_peak);
    }

    /* --- Teaching Summary --- */

    printf("\n--- Teaching Notes ---\n\n");

    printf("Level 0 (scalar): Triple-loop GEMM. Each output element does\n"
           "  K multiply-adds, loading A and B from memory each time.\n"
           "  Arithmetic intensity: ~0.25 flops/byte. Memory-bound.\n\n");

    printf("Level 1 (naive SIMD): Vectorize the K dimension. A loads are\n"
           "  contiguous (good), but B loads require scattering (bad).\n"
           "  The _mm256_set_ps() gather compiles to 8 scalar loads,\n"
           "  8 inserts, and high port 5 pressure. Speedup is limited.\n\n");

    printf("Level 2 (packed SIMD): Pack B so that 8 column values are\n"
           "  contiguous. Now both A and B loads are efficient, and\n"
           "  C is loaded/stored once per output row (not per element).\n"
           "  However, C is still loaded/stored per row × per k-iteration,\n"
           "  though we amortize this by keeping C in a register.\n\n");

    printf("Level 3 (micro_tiled): Full register blocking. The 8×8\n"
           "  micro-kernel keeps all 8 rows of C in ymm registers,\n"
           "  reducing C memory traffic to 1 load + 1 store per row\n"
           "  (16 total, vs 16 × K × M×N/NR for naive). A values are\n"
           "  reused 8× (once per C column), B values are reused 8×\n"
           "  (once per C row). This is the core insight of BLIS/OpenBLAS.\n\n");

    printf("Inner loop of gemm_micro_8x8 (per k-iteration):\n"
           "  1 × _mm256_loadu_ps    (load B_packed[k][0:7])\n"
           "  8 × _mm256_set1_ps     (broadcast A values)\n"
           "  8 × _mm256_fmadd_ps    (FMA into C accumulators)\n"
           "  = 17 instructions for 16 flops\n\n");

    printf("Register usage: %d of 16 ymm registers (%.0f%%)\n"
           "  Headroom for: K-unrolling (process 2 k-iterations at once)\n"
           "                row-prefetching (keep next A row in register)\n\n",
           10, 10.0 / 16.0 * 100.0);

    printf("Scaling to AVX-512:\n"
           "  - zmm registers hold 16 f32 → NR=16 column block\n"
           "  - 32 zmm registers available → 16×8 or 16×16 kernel\n"
           "  - 2× SIMD width × 2× registers = 4× potential speedup\n"
           "  - Same broadcast+FMA pattern, just wider\n");

    ALIGNED_FREE(A);
    ALIGNED_FREE(B);
    ALIGNED_FREE(C);
    ALIGNED_FREE(ref_c);
    ALIGNED_FREE(C_work);

    return 0;
}
