// lecture9_part1.cpp
// Stanford CS149, Lecture 9: Efficiently Evaluating DNNs
// Part 1: Matrix Multiplication — Naive, Blocked, and Tiled
//
// Implements progressively optimized dense matrix multiplication:
//   1. Naive triple loop (low arithmetic intensity)
//   2. Single-level blocking (cache-optimized)
//   3. Hierarchical blocking (L1 + L2 cache exploitation)
//   4. SIMD-friendly vectorized approach (2 variants)
//   5. Register blocking / micro-kernel approach
//
// All variants compute C = A * B (or C += A * B)
//
// Compile: g++ -std=c++17 -O2 -pthread lecture9_part1.cpp -o lecture9_part1
// Run: ./lecture9_part1

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <cassert>

// ============================================================================
// Matrix helper
// ============================================================================

class Matrix {
public:
    std::vector<float> data;
    size_t rows, cols;

    Matrix(size_t r, size_t c) : rows(r), cols(c), data(r * c, 0.0f) {}

    float& at(size_t r, size_t c) { return data[r * cols + c]; }
    float  at(size_t r, size_t c) const { return data[r * cols + c]; }

    void fill(float val) {
        std::fill(data.begin(), data.end(), val);
    }

    void randomize(float scale = 1.0f) {
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = static_cast<float>(i % 100) * scale;
        }
    }

    bool equals(const Matrix& other, float tolerance = 0.01f) const {
        if (rows != other.rows || cols != other.cols) return false;
        for (size_t i = 0; i < data.size(); i++) {
            if (std::abs(data[i] - other.data[i]) > tolerance) return false;
        }
        return true;
    }
};

void printMatrix(const std::string& name, const Matrix& m,
                 size_t maxRows = 6, size_t maxCols = 6)
{
    std::cout << name << " (" << m.rows << "x" << m.cols << "):\n";
    for (size_t r = 0; r < std::min(m.rows, maxRows); r++) {
        std::cout << "  ";
        for (size_t c = 0; c < std::min(m.cols, maxCols); c++) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(1)
                      << m.at(r, c);
        }
        if (m.cols > maxCols) std::cout << " ...";
        std::cout << "\n";
    }
    if (m.rows > maxRows) std::cout << "  ...\n";
}

// ============================================================================
// Timing utility
// ============================================================================

template<typename Func>
double timeIt(Func f, const std::string& label) {
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  " << label << ": " << std::fixed << std::setprecision(2)
              << ms << " ms\n";
    return ms;
}

// ============================================================================
// Version 1: Naive Matrix Multiplication
// C[j][i] += A[j][k] * B[k][i]
//
// Problems:
//   - No temporal locality: A and B elements loaded from cache repeatedly
//   - Arithmetic intensity = 1 flop / (2 loads + 0 stores to cache)
//     ≈ 0.5 (per inner loop iteration)
// ============================================================================

void gemmNaive(const Matrix& A, const Matrix& B, Matrix& C) {
    assert(A.cols == B.rows);
    assert(C.rows == A.rows && C.cols == B.cols);

    for (size_t j = 0; j < A.rows; j++) {
        for (size_t i = 0; i < B.cols; i++) {
            float sum = C.at(j, i);
            for (size_t k = 0; k < A.cols; k++) {
                sum += A.at(j, k) * B.at(k, i);
            }
            C.at(j, i) = sum;
        }
    }
}

// ============================================================================
// Version 2: Single-level Blocking
// Compute partial block of C while sub-blocks of A and B remain in cache.
//
// Choose BLOCKSIZE so that 3 * BLOCKSIZE^2 * sizeof(float) fits in L1 cache.
// L1 cache ~32KB → BLOCKSIZE^2 * 4 * 3 ≈ 32KB → BLOCKSIZE ≈ 52.
// We'll use 32-64 for L1, 128-256 for L2.
//
// Arithmetic intensity increases because each A element is reused BLOCKSIZE_I
// times and each B element is reused BLOCKSIZE_J times.
// ============================================================================

void gemmBlocked(const Matrix& A, const Matrix& B, Matrix& C,
                 size_t BS_J, size_t BS_I, size_t BS_K)
{
    assert(A.cols == B.rows);
    assert(C.rows == A.rows && C.cols == B.cols);

    // C is M×N, A is M×K, B is K×N
    size_t M = A.rows;
    size_t N = B.cols;
    size_t K = A.cols;

    for (size_t jb = 0; jb < M; jb += BS_J) {
        size_t jEnd = std::min(jb + BS_J, M);
        for (size_t ib = 0; ib < N; ib += BS_I) {
            size_t iEnd = std::min(ib + BS_I, N);
            for (size_t kb = 0; kb < K; kb += BS_K) {
                size_t kEnd = std::min(kb + BS_K, K);

                // Compute partial result for block C[jb:jEnd][ib:iEnd]
                for (size_t j = jb; j < jEnd; j++) {
                    for (size_t i = ib; i < iEnd; i++) {
                        float sum = 0.0f;
                        for (size_t k = kb; k < kEnd; k++) {
                            sum += A.at(j, k) * B.at(k, i);
                        }
                        C.at(j, i) += sum;
                    }
                }
            }
        }
    }
}

// ============================================================================
// Version 3: Hierarchical Blocking (L1 + L2)
// Exploit multiple cache levels: L2 blocks hold larger sub-blocks,
// within those, L1 blocks for inner computation.
//
// L2 ~256KB, L1 ~32KB → L2 blocksize ~128, L1 blocksize ~32
// ============================================================================

void gemmHierarchical(const Matrix& A, const Matrix& B, Matrix& C,
                      size_t L2_J, size_t L2_I, size_t L2_K,
                      size_t L1_J, size_t L1_I, size_t L1_K)
{
    assert(A.cols == B.rows);
    assert(C.rows == A.rows && C.cols == B.cols);

    size_t M = A.rows;
    size_t N = B.cols;
    size_t K = A.cols;

    // L2 blocking (outer loops)
    for (size_t jb2 = 0; jb2 < M; jb2 += L2_J) {
        size_t jEnd2 = std::min(jb2 + L2_J, M);
        for (size_t ib2 = 0; ib2 < N; ib2 += L2_I) {
            size_t iEnd2 = std::min(ib2 + L2_I, N);
            for (size_t kb2 = 0; kb2 < K; kb2 += L2_K) {
                size_t kEnd2 = std::min(kb2 + L2_K, K);

                // L1 blocking (inner loops within L2 blocks)
                for (size_t jb1 = jb2; jb1 < jEnd2; jb1 += L1_J) {
                    size_t jEnd1 = std::min(jb1 + L1_J, jEnd2);
                    for (size_t ib1 = ib2; ib1 < iEnd2; ib1 += L1_I) {
                        size_t iEnd1 = std::min(ib1 + L1_I, iEnd2);
                        for (size_t kb1 = kb2; kb1 < kEnd2; kb1 += L1_K) {
                            size_t kEnd1 = std::min(kb1 + L1_K, kEnd2);

                            // Inner kernel
                            for (size_t j = jb1; j < jEnd1; j++) {
                                for (size_t i = ib1; i < iEnd1; i++) {
                                    float sum = 0.0f;
                                    for (size_t k = kb1; k < kEnd1; k++) {
                                        sum += A.at(j, k) * B.at(k, i);
                                    }
                                    C.at(j, i) += sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Version 4: Blocked + Pre-transpose B for better access pattern
//
// When transposing B to BT, the inner loop accesses both A and BT
// with unit stride (row-major layout), improving spatial locality.
//
// This is the version discussed in the lecture for the case where
// the i dimension is small and SIMD vectorization needs contiguous data.
// ============================================================================

void gemmBlockedTranspose(const Matrix& A, const Matrix& B, Matrix& C,
                          size_t BS_J, size_t BS_K)
{
    assert(A.cols == B.rows);
    assert(C.rows == A.rows && C.cols == B.cols);

    size_t M = A.rows;
    size_t N = B.cols;
    size_t K = A.cols;
    size_t BS_I = N;  // Full i dimension

    // Pre-transpose B → BT (BT[k][i] = B[i][k])
    // BT has dimensions K x N
    Matrix BT(K, N);
    for (size_t i = 0; i < N; i++) {
        for (size_t k = 0; k < K; k++) {
            BT.at(k, i) = B.at(k, i);
        }
    }

    for (size_t jb = 0; jb < M; jb += BS_J) {
        size_t jEnd = std::min(jb + BS_J, M);
        for (size_t kb = 0; kb < K; kb += BS_K) {
            size_t kEnd = std::min(kb + BS_K, K);

            for (size_t j = jb; j < jEnd; j++) {
                for (size_t i = 0; i < N; i++) {
                    float sum = 0.0f;
                    // Dot product: row of A · row of BT (both contiguous)
                    for (size_t k = kb; k < kEnd; k++) {
                        sum += A.at(j, k) * BT.at(k, i);
                    }
                    C.at(j, i) += sum;
                }
            }
        }
    }
}

// ============================================================================
// Version 5: Micro-kernel with register blocking (conceptual)
//
// In a real high-performance GEMM:
//   - Inner-most loops operate on a small sub-block that fits in registers
//   - e.g., compute 4x4 sub-block of C using A[4][K] and B[K][4] in registers
//   - This minimizes load/store instructions
//
// Here we simulate it with a 4x4 micro-kernel that the compiler can
// potentially keep in registers with -O2 optimization.
// ============================================================================

void gemmMicroKernel(const Matrix& A, const Matrix& B, Matrix& C,
                     size_t MR, size_t NR, size_t KC)
{
    assert(A.cols == B.rows);
    assert(C.rows == A.rows && C.cols == B.cols);

    size_t M = A.rows;
    size_t N = B.cols;
    size_t K = A.cols;

    // Blocked outer loops to feed the micro-kernel
    for (size_t jb = 0; jb < M; jb += MR) {
        size_t jEnd = std::min(jb + MR, M);
        for (size_t ib = 0; ib < N; ib += NR) {
            size_t iEnd = std::min(ib + NR, N);
            for (size_t kb = 0; kb < K; kb += KC) {
                size_t kEnd = std::min(kb + KC, K);

                // Micro-kernel: compute MR x NR block of C
                // using MR x KC block of A and KC x NR block of B
                // These should stay in registers (with luck + -O2)
                for (size_t j = jb; j < jEnd; j++) {
                    for (size_t i = ib; i < iEnd; i++) {
                        float c_accum = C.at(j, i);  // Load C element once
                        for (size_t k = kb; k < kEnd; k++) {
                            c_accum += A.at(j, k) * B.at(k, i);
                        }
                        C.at(j, i) = c_accum;  // Store once at end
                    }
                }
            }
        }
    }
}

// ============================================================================
// Parallel gemm — parallelize the j-loop (rows of C)
// ============================================================================

void gemmParallel(const Matrix& A, const Matrix& B, Matrix& C,
                  size_t numThreads)
{
    size_t M = A.rows;
    size_t N = B.cols;
    size_t K = A.cols;

    // Use blocked version with parallelism on outer j loop
    const size_t BS = 64;

    std::vector<std::thread> workers;
    for (size_t t = 0; t < numThreads; t++) {
        workers.emplace_back([&A, &B, &C, M, N, K, BS, t, numThreads]() {
            size_t chunkSize = ((M + BS - 1) / BS + numThreads - 1) / numThreads;
            size_t jbStart = t * chunkSize * BS;
            size_t jbEnd   = std::min((t + 1) * chunkSize * BS, M);

            for (size_t jb = jbStart; jb < jbEnd; jb += BS) {
                size_t jEnd = std::min(jb + BS, M);
                for (size_t ib = 0; ib < N; ib += BS) {
                    size_t iEnd = std::min(ib + BS, N);
                    for (size_t kb = 0; kb < K; kb += BS) {
                        size_t kEnd = std::min(kb + BS, K);
                        for (size_t j = jb; j < jEnd; j++) {
                            for (size_t i = ib; i < iEnd; i++) {
                                float sum = 0.0f;
                                for (size_t k = kb; k < kEnd; k++) {
                                    sum += A.at(j, k) * B.at(k, i);
                                }
                                C.at(j, i) += sum;
                            }
                        }
                    }
                }
            }
        });
    }
    for (auto& w : workers) w.join();
}

// ============================================================================
// main
// ============================================================================

int main()
{
    std::cout << "==================================================\n";
    std::cout << "Lecture 9 Part 1: Matrix Multiplication Optimizations\n";
    std::cout << "==================================================\n\n";

    // ---- Small test: correctness verification ----
    {
        std::cout << "--- Correctness Test (small matrices) ---\n";

        size_t M = 8, K = 6, N = 8;
        Matrix A(M, K);  A.randomize();
        Matrix B(K, N);  B.randomize();
        Matrix C1(M, N); C1.fill(0);
        Matrix C2(M, N); C2.fill(0);
        Matrix C3(M, N); C3.fill(0);
        Matrix C4(M, N); C4.fill(0);
        Matrix C5(M, N); C5.fill(0);

        // Compute reference using naive
        gemmNaive(A, B, C1);

        // Verify other versions
        gemmBlocked(A, B, C2, 4, 4, 4);
        std::cout << "  Blocked: " << (C2.equals(C1) ? "PASSED" : "FAILED") << "\n";

        gemmHierarchical(A, B, C3, 6, 6, 6, 3, 3, 3);
        std::cout << "  Hierarchical: " << (C3.equals(C1) ? "PASSED" : "FAILED") << "\n";

        gemmBlockedTranspose(A, B, C4, 4, 4);
        std::cout << "  Blocked+Transpose: " << (C4.equals(C1) ? "PASSED" : "FAILED") << "\n";

        gemmMicroKernel(A, B, C5, 4, 4, 4);
        std::cout << "  Micro-kernel: " << (C5.equals(C1) ? "PASSED" : "FAILED") << "\n";

        printMatrix("C (result)", C1);
    }

    // ---- Performance comparison (medium matrices) ----
    {
        std::cout << "\n--- Performance Comparison (256x256 matrices) ---\n";

        size_t M = 256, K = 256, N = 256;
        Matrix A(M, K);  A.randomize();
        Matrix B(K, N);  B.randomize();

        // Naive
        {
            Matrix C(M, N); C.fill(0);
            timeIt([&]() { gemmNaive(A, B, C); }, "Naive");
        }

        // Blocked (L1-sized blocks)
        {
            Matrix C(M, N); C.fill(0);
            timeIt([&]() { gemmBlocked(A, B, C, 32, 32, 32); }, "Blocked (32x32x32)");
        }

        // Hierarchical
        {
            Matrix C(M, N); C.fill(0);
            timeIt([&]() { gemmHierarchical(A, B, C, 128, 128, 128, 32, 32, 32); },
                   "Hierarchical (L2:128, L1:32)");
        }

        // Blocked + Transposed
        {
            Matrix C(M, N); C.fill(0);
            timeIt([&]() { gemmBlockedTranspose(A, B, C, 32, 32); },
                   "Blocked+Transpose (32x32)");
        }

        // Micro-kernel
        {
            Matrix C(M, N); C.fill(0);
            timeIt([&]() { gemmMicroKernel(A, B, C, 4, 4, 64); },
                   "Micro-kernel (4x4x64)");
        }

        // Parallel (4 threads)
        {
            Matrix C(M, N); C.fill(0);
            timeIt([&]() { gemmParallel(A, B, C, 4); }, "Parallel blocked (4 threads)");
        }
    }

    // ---- Arithmetic Intensity Analysis ----
    {
        std::cout << "\n--- Arithmetic Intensity Analysis ---\n";
        std::cout << "For M=N=K=256, each version reads:\n";
        std::cout << "  Naive: Each element of A loaded N times,\n";
        std::cout << "         each element of B loaded M times.\n";
        std::cout << "         AI ≈ O(1) → BW-bound on GPU\n\n";
        std::cout << "  Blocked (BS=32): A sub-block (32x32 loaded once,\n";
        std::cout << "                   used for 32 columns of C.\n";
        std::cout << "                   AI ≈ BS → compute-bound possible\n";
    }

    // ---- Large matrix test ----
    {
        std::cout << "\n--- Large Matrix (1024x1024) ---\n";

        size_t M = 1024, K = 1024, N = 1024;
        Matrix A(M, K);  A.randomize(0.001f);
        Matrix B(K, N);  B.randomize(0.001f);

        // Only run the optimized versions for large matrices
        {
            Matrix C(M, N); C.fill(0);
            timeIt([&]() { gemmBlocked(A, B, C, 64, 64, 64); },
                   "Blocked (64x64x64)");
        }
        {
            Matrix C(M, N); C.fill(0);
            timeIt([&]() { gemmHierarchical(A, B, C, 256, 256, 256, 32, 32, 32); },
                   "Hierarchical (L2:256, L1:32)");
        }
        {
            Matrix C(M, N); C.fill(0);
            timeIt([&]() { gemmParallel(A, B, C, 8); },
                   "Parallel blocked (8 threads)");
        }

        // roofline: 1024^3 = ~1B flops
        double gflops = 2.0 * M * N * K / 1e9;  // 2 ops per mul-add
        std::cout << "  Total FLOPs: ~" << std::fixed << std::setprecision(1)
                  << gflops << " GFLOPs\n";
    }

    std::cout << "\n==================================================\n";
    std::cout << "Key concepts demonstrated:\n";
    std::cout << "  - Naive GEMM: AI = O(1), bandwidth-bound\n";
    std::cout << "  - Blocking: increase AI by reusing cache-resident data\n";
    std::cout << "  - Hierarchical blocking: exploit L1 + L2 cache hierarchy\n";
    std::cout << "  - Pre-transpose: improve access pattern for SIMD\n";
    std::cout << "  - Micro-kernel: register-level blocking\n";
    std::cout << "  - Parallelization: outer j-loop (rows of C)\n";
    std::cout << "==================================================\n";

    return 0;
}
