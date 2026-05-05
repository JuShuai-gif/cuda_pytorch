// lecture7_part1.cpp
// Stanford CS149, Lecture 7: GPU Architecture & CUDA Programming
// Part 1: Thread Hierarchy Simulation — grids, blocks, and threads on CPU
//
// This program simulates the CUDA thread hierarchy using C++ threads.
// It demonstrates how thread blocks and grids map work to data in a
// multi-dimensional fashion, similar to CUDA's <<<numBlocks, threadsPerBlock>>>.
//
// Compile: g++ -std=c++17 -pthread lecture7_part1.cpp -o lecture7_part1
// Run: ./lecture7_part1

#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <iomanip>

// ============================================================================
// Simulated CUDA types
// ============================================================================

struct dim3 {
    int x, y, z;
    dim3(int _x = 1, int _y = 1, int _z = 1) : x(_x), y(_y), z(_z) {}
};

// ============================================================================
// Configuration
// ============================================================================

constexpr int MATRIX_WIDTH  = 16;
constexpr int MATRIX_HEIGHT = 12;
constexpr int BLOCK_DIM_X   = 4;
constexpr int BLOCK_DIM_Y   = 3;

// ============================================================================
// Matrix addition: each "CUDA thread" computes one element of the result
// This is the device kernel equivalent
// ============================================================================

void matrixAddKernel(float A[][MATRIX_WIDTH],
                     float B[][MATRIX_WIDTH],
                     float C[][MATRIX_WIDTH],
                     int blockIdx_x, int blockIdx_y,
                     int threadIdx_x, int threadIdx_y,
                     int blockDim_x, int blockDim_y)
{
    // Compute global index from block + thread indices
    // Equivalent to: int i = blockIdx.x * blockDim.x + threadIdx.x;
    int global_x = blockIdx_x * blockDim_x + threadIdx_x;
    int global_y = blockIdx_y * blockDim_y + threadIdx_y;

    if (global_x < MATRIX_WIDTH && global_y < MATRIX_HEIGHT) {
        C[global_y][global_x] = A[global_y][global_x] + B[global_y][global_x];
    }
}

// ============================================================================
// Simulate a single thread block execution
// Each thread in the block computes its assigned element
// ============================================================================

void executeThreadBlock(float A[][MATRIX_WIDTH],
                        float B[][MATRIX_WIDTH],
                        float C[][MATRIX_WIDTH],
                        int blockIdx_x, int blockIdx_y,
                        dim3 blockDim)
{
    // Each block has blockDim.x * blockDim.y threads
    // In CUDA these would run concurrently; here we use std::thread
    std::vector<std::thread> blockThreads;

    for (int ty = 0; ty < blockDim.y; ty++) {
        for (int tx = 0; tx < blockDim.x; tx++) {
            blockThreads.emplace_back(
                matrixAddKernel,
                A, B, C,
                blockIdx_x, blockIdx_y,
                tx, ty,
                blockDim.x, blockDim.y
            );
        }
    }

    // Wait for all threads in this block to complete (__syncthreads equivalent)
    for (auto& t : blockThreads) {
        t.join();
    }
}

// ============================================================================
// Simulate a GPU grid launch: <<<numBlocks, threadsPerBlock>>>
// Launches thread blocks across the grid — blocks execute concurrently
// ============================================================================

void cudaKernelLaunch(float A[][MATRIX_WIDTH],
                      float B[][MATRIX_WIDTH],
                      float C[][MATRIX_WIDTH],
                      dim3 numBlocks, dim3 threadsPerBlock)
{
    std::vector<std::thread> blockThreads;

    for (int by = 0; by < numBlocks.y; by++) {
        for (int bx = 0; bx < numBlocks.x; bx++) {
            // Each thread block runs concurrently with other blocks
            blockThreads.emplace_back(
                executeThreadBlock,
                A, B, C,
                bx, by,
                threadsPerBlock
            );
        }
    }

    // Implicit barrier: kernel returns when all blocks complete
    for (auto& t : blockThreads) {
        t.join();
    }
}

// ============================================================================
// 1D convolution simulation (Lecture 7 example)
// Demonstrates thread-to-element mapping with overlapping data access
// ============================================================================

void convolution1D(const std::vector<float>& input,
                   std::vector<float>& output,
                   int totalElements,
                   int threadsPerBlk)
{
    // Simulating: convolve<<<N/THREADS_PER_BLK, THREADS_PER_BLK>>>(N, input, output)
    int numBlocks = (totalElements + threadsPerBlk - 1) / threadsPerBlk;

    std::vector<std::thread> blockThreads;
    const float* inPtr  = input.data();
    float*       outPtr = output.data();

    for (int blk = 0; blk < numBlocks; blk++) {
        blockThreads.emplace_back([=]() {
            for (int t = 0; t < threadsPerBlk; t++) {
                // Global index: blockIdx.x * blockDim.x + threadIdx.x
                int index = blk * threadsPerBlk + t;
                if (index >= totalElements || index + 2 >= totalElements) continue;

                // Convolution window of size 3
                float result = 0.0f;
                for (int i = 0; i < 3; i++) {
                    result += inPtr[index + i];
                }
                outPtr[index] = result / 3.0f;
            }
        });
    }

    for (auto& bt : blockThreads) {
        bt.join();
    }
}

// ============================================================================
// Print matrix helper
// ============================================================================

void printMatrix(const std::string& name, float mat[][MATRIX_WIDTH],
                 int height, int width)
{
    std::cout << "\n" << name << ":\n";
    for (int y = 0; y < height; y++) {
        std::cout << "  ";
        for (int x = 0; x < width; x++) {
            std::cout << std::setw(5) << std::fixed
                      << std::setprecision(0) << mat[y][x];
        }
        std::cout << "\n";
    }
}

// ============================================================================
// main
// ============================================================================

int main()
{
    std::cout << "==================================================\n";
    std::cout << "Lecture 7 Part 1: CUDA Thread Hierarchy Simulation\n";
    std::cout << "==================================================\n\n";

    // ----- Matrix Addition Demo -----
    std::cout << "--- Matrix Addition (2D thread hierarchy) ---\n";
    std::cout << "Matrix size: " << MATRIX_HEIGHT << "x" << MATRIX_WIDTH << "\n";
    std::cout << "Block size:  " << BLOCK_DIM_Y << "x" << BLOCK_DIM_X << "\n";

    // Allocate matrices
    float A[MATRIX_HEIGHT][MATRIX_WIDTH] = {};
    float B[MATRIX_HEIGHT][MATRIX_WIDTH] = {};
    float C[MATRIX_HEIGHT][MATRIX_WIDTH] = {};

    // Initialize A and B
    for (int y = 0; y < MATRIX_HEIGHT; y++) {
        for (int x = 0; x < MATRIX_WIDTH; x++) {
            A[y][x] = y * MATRIX_WIDTH + x;
            B[y][x] = (y * MATRIX_WIDTH + x) * 10;
        }
    }

    // Compute grid dimensions (with ceiling division for non-multiples)
    dim3 threadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 numBlocks((MATRIX_WIDTH  + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
                   (MATRIX_HEIGHT + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    std::cout << "Grid:  " << numBlocks.y << "x" << numBlocks.x
              << " blocks\n";
    std::cout << "Total threads: "
              << numBlocks.x * numBlocks.y * threadsPerBlock.x * threadsPerBlock.y
              << " (only " << MATRIX_HEIGHT * MATRIX_WIDTH
              << " elements need computation)\n";

    cudaKernelLaunch(A, B, C, numBlocks, threadsPerBlock);

    printMatrix("Matrix A", A, MATRIX_HEIGHT, MATRIX_WIDTH);
    printMatrix("Matrix B", B, MATRIX_HEIGHT, MATRIX_WIDTH);
    printMatrix("Matrix C = A + B", C, MATRIX_HEIGHT, MATRIX_WIDTH);

    // Verify
    bool correct = true;
    for (int y = 0; y < MATRIX_HEIGHT && correct; y++) {
        for (int x = 0; x < MATRIX_WIDTH && correct; x++) {
            if (C[y][x] != A[y][x] + B[y][x]) correct = false;
        }
    }
    std::cout << "\nVerification: " << (correct ? "PASSED" : "FAILED") << "\n";

    // ----- 1D Convolution Demo -----
    std::cout << "\n\n--- 1D Convolution Simulation ---\n";

    constexpr int N = 20;
    std::vector<float> signal(N);
    for (int i = 0; i < N; i++) {
        signal[i] = static_cast<float>(i + 1);
    }

    std::vector<float> result(N - 2);
    convolution1D(signal, result, N - 2, 8);  // 8 threads per block

    std::cout << "Input signal:  ";
    for (float v : signal) std::cout << v << " ";
    std::cout << "\n";

    std::cout << "Convolved (size 3, moving average): ";
    for (float v : result) std::cout << std::setprecision(2) << v << " ";
    std::cout << "\n";

    // Verify manually: output[0] = (1+2+3)/3 = 2, output[1] = (2+3+4)/3 = 3, etc.
    std::cout << "Expected:                              ";
    for (int i = 0; i < N - 2; i++) {
        float expected = (signal[i] + signal[i+1] + signal[i+2]) / 3.0f;
        std::cout << std::setprecision(2) << expected << " ";
    }
    std::cout << "\n";

    // ---- Summary ----
    std::cout << "\n==================================================\n";
    std::cout << "Key concepts demonstrated:\n";
    std::cout << "  - 2D thread hierarchy: grid of thread blocks\n";
    std::cout << "  - Global index computation: blockIdx*blockDim + threadIdx\n";
    std::cout << "  - Ceiling division for non-multiple sizes\n";
    std::cout << "  - Concurrent block execution via C++ threads\n";
    std::cout << "  - 1D convolution: overlapping data access pattern\n";
    std::cout << "==================================================\n";

    return 0;
}
