// lecture7_part2.cpp
// Stanford CS149, Lecture 7: GPU Architecture & CUDA Programming
// Part 2: 1D Convolution — Naive vs. Shared Memory Approach
//
// Demonstrates the two versions of 1D convolution from the lecture:
//   Version 1: Each thread directly reads from global memory (no reuse)
//   Version 2: Threads cooperatively load data into shared memory (reuse)
//
// In a real GPU, shared memory is ~100x faster than global memory.
// This simulation counts "memory accesses" to quantify the difference.
//
// Compile: g++ -std=c++17 -pthread lecture7_part2.cpp -o lecture7_part2
// Run: ./lecture7_part2

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <iomanip>
#include <cstring>
#include <chrono>

// ============================================================================
// Simulation configuration
// ============================================================================

constexpr int THREADS_PER_BLK   = 128;
constexpr int CONV_FILTER_SIZE  = 3;

// Global counters for simulated memory accesses
// These represent reads from GPU global memory (DRAM)
std::atomic<long long> global_reads{0};
std::atomic<long long> shared_reads{0};

// ============================================================================
// Version 1: Naive — each thread reads directly from global memory
//
// CUDA equivalent:
//   __global__ void convolve(int N, float* input, float* output) {
//       int index = blockIdx.x * blockDim.x + threadIdx.x;
//       float result = 0.0f;
//       for (int i=0; i<3; i++)
//           result += input[index + i];   // direct global memory reads
//       output[index] = result / 3.f;
//   }
// ============================================================================

void convolveV1_naive(const float* input, float* output,
                      int startIndex, int blockThreads, int totalN)
{
    for (int t = 0; t < blockThreads; t++) {
        int index = startIndex + t;
        if (index >= totalN) continue;  // guard

        float result = 0.0f;
        // Each thread reads 3 elements from "global memory"
        for (int i = 0; i < CONV_FILTER_SIZE; i++) {
            result += input[index + i];
            global_reads++;  // Count global memory read
        }
        output[index] = result / static_cast<float>(CONV_FILTER_SIZE);
    }
}

// ============================================================================
// Version 2: Shared memory — threads cooperatively load data
//
// CUDA equivalent:
//   __global__ void convolve(int N, float* input, float* output) {
//       __shared__ float support[THREADS_PER_BLK+2];
//       int index = blockIdx.x * blockDim.x + threadIdx.x;
//       support[threadIdx.x] = input[index];
//       if (threadIdx.x < 2)
//           support[THREADS_PER_BLK+threadIdx.x] = input[index+THREADS_PER_BLK];
//       __syncthreads();
//       float result = 0.0f;
//       for (int i=0; i<3; i++)
//           result += support[threadIdx.x + i];
//       output[index] = result / 3.f;
//   }
// ============================================================================

void convolveV2_shared(const float* input, float* output,
                       int startIndex, int blockThreads, int totalN)
{
    // Per-block shared memory (simulated as stack array)
    float support[THREADS_PER_BLK + 2] = {};

    // Cooperative load: each thread loads one element into shared memory
    for (int t = 0; t < blockThreads; t++) {
        int index = startIndex + t;
        if (index < totalN) {
            support[t] = input[index];
            global_reads++;  // Load from global into shared
        }
    }

    // Extra threads load the "halo" elements (the +2 beyond block boundary)
    // Equivalent to: if (threadIdx.x < 2)
    int nextIndex = startIndex + blockThreads;
    if (nextIndex < totalN) {
        support[blockThreads] = input[nextIndex];
        global_reads++;
    }
    if (nextIndex + 1 < totalN) {
        support[blockThreads + 1] = input[nextIndex + 1];
        global_reads++;
    }

    // __syncthreads() barrier — all threads wait for cooperative load to finish
    // (In our sequential simulation, this is implicit)

    // Each thread computes its result from shared memory
    for (int t = 0; t < blockThreads; t++) {
        int index = startIndex + t;
        if (index >= totalN) continue;

        float result = 0.0f;
        for (int i = 0; i < CONV_FILTER_SIZE; i++) {
            result += support[t + i];
            shared_reads++;  // Read from shared memory (fast, on-chip)
        }
        output[index] = result / static_cast<float>(CONV_FILTER_SIZE);
    }
}

// ============================================================================
// Host side: launch all blocks
// ============================================================================

template<typename KernelFunc>
void launchKernel(KernelFunc kernel,
                  const float* input, float* output,
                  int totalN, int threadsPerBlk)
{
    int numBlocks = (totalN + threadsPerBlk - 1) / threadsPerBlk;

    std::vector<std::thread> blockThreads;
    for (int blk = 0; blk < numBlocks; blk++) {
        int startIndex = blk * threadsPerBlk;
        int blkThreads = std::min(threadsPerBlk, totalN - startIndex);

        blockThreads.emplace_back(
            kernel, input, output, startIndex, blkThreads, totalN
        );
    }

    for (auto& t : blockThreads) {
        t.join();
    }
}

// ============================================================================
// Performance timing helper
// ============================================================================

template<typename Func>
double measureTime(Func f)
{
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ============================================================================
// main
// ============================================================================

int main()
{
    std::cout << "==================================================\n";
    std::cout << "Lecture 7 Part 2: 1D Convolution Memory Analysis\n";
    std::cout << "==================================================\n\n";

    constexpr int N = 1 << 20;  // 1,048,576 elements (~1M)
    constexpr int outputN = N - (CONV_FILTER_SIZE - 1);

    // Allocate and initialize input data
    std::vector<float> input(N);
    for (int i = 0; i < N; i++) {
        input[i] = static_cast<float>(i + 1);
    }

    // Step 1: Version 1 — Naive (all global memory reads)
    {
        std::cout << "--- Version 1: Naive (global memory only) ---\n";
        std::vector<float> outputV1(outputN, 0.0f);
        global_reads = 0;
        shared_reads = 0;

        double timeV1 = measureTime([&]() {
            launchKernel(convolveV1_naive, input.data(), outputV1.data(),
                        outputN, THREADS_PER_BLK);
        });

        std::cout << "  Data size: " << outputN << " output elements\n";
        std::cout << "  Thread blocks: " << (outputN + THREADS_PER_BLK - 1) / THREADS_PER_BLK << "\n";
        std::cout << "  Global memory reads: " << global_reads.load() << "\n";
        std::cout << "    Per output element: "
                  << static_cast<double>(global_reads.load()) / outputN << "\n";
        std::cout << "  Time: " << std::fixed << std::setprecision(2) << timeV1 << " ms\n";
        std::cout << "  Sample output[0..4]: ";
        for (int i = 0; i < 5; i++) std::cout << outputV1[i] << " ";
        std::cout << "\n\n";
    }

    // Step 2: Version 2 — Shared memory (cooperative load)
    {
        std::cout << "--- Version 2: Shared Memory (cooperative load) ---\n";
        std::vector<float> outputV2(outputN, 0.0f);
        global_reads = 0;
        shared_reads = 0;

        double timeV2 = measureTime([&]() {
            launchKernel(convolveV2_shared, input.data(), outputV2.data(),
                        outputN, THREADS_PER_BLK);
        });

        int numBlocks = (outputN + THREADS_PER_BLK - 1) / THREADS_PER_BLK;
        std::cout << "  Data size: " << outputN << " output elements\n";
        std::cout << "  Thread blocks: " << numBlocks << "\n";
        std::cout << "  Global memory reads: " << global_reads.load() << "\n";
        std::cout << "    (Each block loads " << (THREADS_PER_BLK + 2)
                  << " elements into shared memory)\n";
        std::cout << "    Per output element: "
                  << static_cast<double>(global_reads.load()) / outputN << "\n";
        std::cout << "  Shared memory reads: " << shared_reads.load() << "\n";
        std::cout << "  Time: " << std::fixed << std::setprecision(2) << timeV2 << " ms\n";
        std::cout << "  Sample output[0..4]: ";
        for (int i = 0; i < 5; i++) std::cout << outputV2[i] << " ";
        std::cout << "\n\n";
    }

    // Comparison summary
    {
        std::cout << "--- Comparison ---\n";
        std::cout << "Version 1 (naive):\n";
        std::cout << "  Each of " << outputN << " threads reads " << CONV_FILTER_SIZE
                  << " elements from global memory.\n";
        std::cout << "  Total global reads: " << outputN << " x " << CONV_FILTER_SIZE
                  << " = " << outputN * CONV_FILTER_SIZE << "\n\n";

        int numBlocks = (outputN + THREADS_PER_BLK - 1) / THREADS_PER_BLK;
        std::cout << "Version 2 (shared memory):\n";
        std::cout << "  Each of " << numBlocks << " blocks cooperatively loads "
                  << (THREADS_PER_BLK + 2) << " elements.\n";
        std::cout << "  Total global reads: " << numBlocks << " x "
                  << (THREADS_PER_BLK + 2) << " = "
                  << numBlocks * (THREADS_PER_BLK + 2) << "\n\n";

        double reduction = 1.0 - static_cast<double>(numBlocks * (THREADS_PER_BLK + 2))
                                   / (outputN * CONV_FILTER_SIZE);
        std::cout << "Reduction in global memory reads: "
                  << std::fixed << std::setprecision(1) << reduction * 100 << "%\n";
    }

    std::cout << "\n==================================================\n";
    std::cout << "Key concepts demonstrated:\n";
    std::cout << "  - Naive approach: O(N*filter_size) global reads\n";
    std::cout << "  - Shared memory: O(N + 2*blocks) global reads\n";
    std::cout << "  - Cooperative data loading across threads\n";
    std::cout << "  - __syncthreads() barrier concept\n";
    std::cout << "  - Halo elements at block boundaries\n";
    std::cout << "==================================================\n";

    return 0;
}
