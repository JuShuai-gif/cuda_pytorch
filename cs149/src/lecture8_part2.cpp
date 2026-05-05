// lecture8_part2.cpp
// Stanford CS149, Lecture 8: Data-Parallel Thinking
// Part 2: Parallel Scan (Prefix Sum) Algorithms
//
// Implements four scan variants as discussed in the lecture:
//   1. Sequential scan (baseline, O(N))
//   2. Naive inclusive scan (O(N log N) work, O(log N) span)
//   3. Work-efficient exclusive scan (Blelloch) (O(N) work, O(log N) span)
//   4. SIMD-style warp scan (O(N log N) work, better for small N)
//   5. Multi-core scan (partition then combine)
//
// Compile: g++ -std=c++17 -pthread lecture8_part2.cpp -o lecture8_part2
// Run: ./lecture8_part2

#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <cassert>

// ============================================================================
// Utility: check if N is power of 2
// ============================================================================

bool isPowerOfTwo(size_t n) { return (n & (n - 1)) == 0; }

size_t nextPowerOfTwo(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

int ilog2(size_t n) {
    int log = 0;
    while (n >>= 1) log++;
    return log;
}

// ============================================================================
// Print array helper
// ============================================================================

void printArray(const std::string& label, const std::vector<int>& arr) {
    std::cout << label << ": [";
    for (size_t i = 0; i < arr.size(); i++) {
        std::cout << arr[i];
        if (i < arr.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

// ============================================================================
// 1. Sequential Scan (baseline)
// ============================================================================

std::vector<int> sequentialScan(const std::vector<int>& input) {
    std::vector<int> output(input.size());
    if (input.empty()) return output;

    output[0] = input[0];
    for (size_t i = 1; i < input.size(); i++) {
        output[i] = output[i - 1] + input[i];
    }
    return output;
}

std::vector<int> sequentialScanExclusive(const std::vector<int>& input) {
    std::vector<int> output(input.size());
    if (input.empty()) return output;

    output[0] = 0;  // Identity element
    int running = 0;
    for (size_t i = 0; i < input.size() - 1; i++) {
        running += input[i];
        output[i + 1] = running;
    }
    return output;
}

// ============================================================================
// 2. Naive Inclusive Scan (O(N log N) work, O(log N) span)
// Lecture slide: each step doubles the stride
//
// Pseudocode:
//   for d=0 to log2(N)-1:
//     forall k: if k >= 2^d:
//       a[k] = a[k-2^d] + a[k]
// ============================================================================

void naiveParallelScan(std::vector<int>& a) {
    size_t n = a.size();
    if (n == 0) return;

    for (int d = 0; d < ilog2(n); d++) {
        int stride = 1 << d;  // 2^d

        // Parallel forall k — simulate with threads
        int numThreads = static_cast<int>(n) / 2;
        numThreads = std::max(1, std::min(numThreads, 8));
        int chunkSize = (static_cast<int>(n) + numThreads - 1) / numThreads;

        std::vector<std::thread> workers;
        for (int w = 0; w < numThreads; w++) {
            int start = w * chunkSize;
            int end   = std::min(start + chunkSize, static_cast<int>(n));

            workers.emplace_back([&a, stride, start, end]() {
                for (int k = start; k < end; k++) {
                    if (k >= stride) {
                        a[k] = a[k - stride] + a[k];
                    }
                }
            });
        }
        for (auto& t : workers) t.join();
    }
}

// ============================================================================
// 3. Work-Efficient Exclusive Scan (Blelloch Algorithm)
// O(N) work, O(log N) span
//
// Up-sweep (reduce phase):
//   for d=0 to log2(N)-1:
//     forall k=0 to N-1 by 2^(d+1):
//       a[k + 2^(d+1) - 1] = a[k + 2^d - 1] + a[k + 2^(d+1) - 1]
//
// Down-sweep:
//   a[N-1] = 0
//   for d=log2(N)-1 down to 0:
//     forall k=0 to N-1 by 2^(d+1):
//       tmp = a[k + 2^d - 1]
//       a[k + 2^d - 1] = a[k + 2^(d+1) - 1]
//       a[k + 2^(d+1) - 1] = tmp + a[k + 2^(d+1) - 1]
// ============================================================================

void workEfficientScan(std::vector<int>& a) {
    size_t n = a.size();
    if (n < 2) return;
    assert(isPowerOfTwo(n));

    int logN = ilog2(n);

    // --- Up-sweep ---
    for (int d = 0; d < logN; d++) {
        int stride   = 1 << (d + 1);  // 2^(d+1)
        int offset   = 1 << d;         // 2^d

        std::vector<std::thread> workers;
        for (size_t k = 0; k < n; k += stride) {
            workers.emplace_back([&a, k, offset, stride, n]() {
                (void)n;
                size_t left  = k + offset - 1;
                size_t right = k + stride - 1;
                a[right] = a[left] + a[right];
            });
        }
        for (auto& t : workers) t.join();
    }

    // --- Down-sweep ---
    a[n - 1] = 0;  // Set last element to identity

    for (int d = logN - 1; d >= 0; d--) {
        int stride   = 1 << (d + 1);
        int offset   = 1 << d;

        std::vector<std::thread> workers;
        for (size_t k = 0; k < n; k += stride) {
            workers.emplace_back([&a, k, offset, stride]() {
                size_t left  = k + offset - 1;
                size_t right = k + stride - 1;
                int tmp      = a[left];
                a[left]      = a[right];
                a[right]     = tmp + a[right];
            });
        }
        for (auto& t : workers) t.join();
    }
}

// ============================================================================
// 4. SIMD-style Warp Scan (naive O(N log N), but better SIMD utilization)
// This is the version used inside a CUDA warp for 32-element scans.
// Even though it has more work, it maps better to SIMD hardware because
// each step has uniform control flow within the warp.
// ============================================================================

std::vector<int> warpScan(const std::vector<int>& input) {
    size_t n = input.size();
    std::vector<int> ptr = input;  // In-place computation
    std::vector<int> result(n, 0);

    int steps = ilog2(n);
    // For n=32, steps=5 (2^5=32)

    for (int i = 0; i < steps; i++) {
        int shift = 1 << i;
        for (size_t idx = 0; idx < n; idx++) {
            int lane = static_cast<int>(idx);
            if (lane >= shift) {
                ptr[idx] = ptr[idx - shift] + ptr[idx];
            }
        }
    }

    // Extract exclusive scan result
    for (size_t i = 0; i < n; i++) {
        result[i] = (i > 0) ? ptr[i - 1] : 0;
    }

    return result;
}

// ============================================================================
// 5. Multi-Core Scan (partition + sequential scan + add bases)
// This is the approach shown in the lecture for 2+ cores.
// Work ~1.5N with perfect spatial locality.
// ============================================================================

std::vector<int> multiCoreScan(const std::vector<int>& input,
                               size_t numWorkers)
{
    size_t n = input.size();
    std::vector<int> output(n, 0);

    if (numWorkers == 0) numWorkers = 1;

    // Partition the array
    size_t chunkSize = (n + numWorkers - 1) / numWorkers;

    // Step 1: Each worker does sequential scan on its chunk
    std::vector<int> partialSums(numWorkers, 0);

    // Compensate for last chunk potentially being smaller
    std::vector<size_t> chunkStarts(numWorkers);
    std::vector<size_t> chunkSizes(numWorkers);
    size_t pos = 0;
    for (size_t w = 0; w < numWorkers; w++) {
        chunkStarts[w] = pos;
        chunkSizes[w]  = (pos + chunkSize <= n) ? chunkSize : (n > pos ? n - pos : 0);
        pos += chunkSizes[w];
    }

    std::vector<std::thread> workers;
    for (size_t w = 0; w < numWorkers; w++) {
        workers.emplace_back([&input, &output, &partialSums, &chunkStarts,
                              &chunkSizes, w]() {
            size_t start = chunkStarts[w];
            size_t size  = chunkSizes[w];
            if (size == 0) return;

            // Sequential inclusive scan within chunk
            output[start] = input[start];
            for (size_t i = 1; i < size; i++) {
                output[start + i] = output[start + i - 1] + input[start + i];
            }
            partialSums[w] = output[start + size - 1];
        });
    }
    for (auto& t : workers) t.join();

    // Step 2: Compute bases (exclusive prefix sum of partial sums)
    std::vector<int> bases(numWorkers, 0);
    int runningBase = 0;
    for (size_t w = 0; w < numWorkers; w++) {
        bases[w]   = runningBase;
        runningBase += partialSums[w];
    }

    // Step 3: Add bases to each chunk (except first)
    workers.clear();
    for (size_t w = 1; w < numWorkers; w++) {
        workers.emplace_back([&output, &bases, &chunkStarts, &chunkSizes, w]() {
            size_t start = chunkStarts[w];
            size_t size  = chunkSizes[w];
            for (size_t i = 0; i < size; i++) {
                output[start + i] += bases[w];
            }
        });
    }
    for (auto& t : workers) t.join();

    return output;
}

// ============================================================================
// Verification
// ============================================================================

bool verify(const std::string& name,
            const std::vector<int>& result,
            const std::vector<int>& expected)
{
    if (result.size() != expected.size()) {
        std::cout << "  " << name << ": FAILED (size mismatch)\n";
        return false;
    }
    for (size_t i = 0; i < result.size(); i++) {
        if (result[i] != expected[i]) {
            std::cout << "  " << name << ": FAILED at index " << i
                      << " (got " << result[i] << ", expected " << expected[i] << ")\n";
            return false;
        }
    }
    std::cout << "  " << name << ": PASSED\n";
    return true;
}

// ============================================================================
// main
// ============================================================================

int main()
{
    std::cout << "==================================================\n";
    std::cout << "Lecture 8 Part 2: Parallel Scan (Prefix Sum) Algorithms\n";
    std::cout << "==================================================\n\n";

    // Test data
    std::vector<int> data = {3, 8, 4, 6, 3, 9, 2, 8};
    // Expected inclusive: [3, 11, 15, 21, 24, 33, 35, 43]
    // Expected exclusive: [0, 3, 11, 15, 21, 24, 33, 35]

    printArray("Input", data);

    // 1. Sequential baseline
    auto seqInclusive = sequentialScan(data);
    auto seqExclusive = sequentialScanExclusive(data);
    std::cout << "\n--- Baseline Sequential ---\n";
    printArray("  Inclusive", seqInclusive);
    printArray("  Exclusive", seqExclusive);

    // 2. Naive parallel scan (O(N log N))
    {
        std::cout << "\n--- Naive Parallel Scan (O(N log N)) ---\n";
        std::vector<int> naive = data;
        naiveParallelScan(naive);
        printArray("  Result", naive);
        verify("Naive inclusive", naive, seqInclusive);
    }

    // 3. Work-efficient scan (O(N)) — requires power-of-2 size
    {
        std::cout << "\n--- Work-Efficient Scan (Blelloch, O(N)) ---\n";
        // Pad to power of 2
        size_t paddedN = nextPowerOfTwo(data.size());
        std::vector<int> padded(paddedN, 0);
        std::copy(data.begin(), data.end(), padded.begin());

        workEfficientScan(padded);

        std::vector<int> blelloch(data.size());
        std::copy(padded.begin(), padded.begin() + data.size(),
                  blelloch.begin());
        printArray("  Result", blelloch);
        verify("Blelloch exclusive", blelloch, seqExclusive);
    }

    // 4. Warp scan (SIMD-style)
    {
        std::cout << "\n--- Warp Scan (SIMD-style, prefer for small N) ---\n";
        // Use exactly 32 elements for the warp scan
        std::vector<int> warpData(32, 0);
        for (size_t i = 0; i < data.size(); i++) warpData[i] = data[i];

        auto warpResult = warpScan(warpData);

        // Show only the relevant portion
        std::vector<int> warpSubset(data.size());
        std::copy(warpResult.begin(), warpResult.begin() + data.size(),
                  warpSubset.begin());
        printArray("  Result (exclusive)", warpSubset);

        // Expected exclusive scan for padded warp data
        std::vector<int> warpExpected(warpData.size());
        if (!warpData.empty()) {
            warpExpected[0] = 0;
            int running = 0;
            for (size_t i = 0; i < warpData.size() - 1; i++) {
                running += warpData[i];
                warpExpected[i + 1] = running;
            }
        }
        std::vector<int> warpExpectedSubset(data.size());
        std::copy(warpExpected.begin(), warpExpected.begin() + data.size(),
                  warpExpectedSubset.begin());
        verify("Warp exclusive", warpSubset, warpExpectedSubset);
    }

    // 5. Multi-core scan
    {
        std::cout << "\n--- Multi-Core Scan (partition + combine) ---\n";
        size_t numWorkers = 3;

        auto mcResult = multiCoreScan(data, numWorkers);
        printArray("  Inclusive (" + std::to_string(numWorkers) + " cores)", mcResult);
        verify("Multi-core inclusive", mcResult, seqInclusive);
    }

    // 6. Large array performance comparison
    {
        std::cout << "\n--- Large Array Scan (N=2^20 = 1,048,576) ---\n";
        size_t largeN = 1 << 20;  // 1,048,576
        std::vector<int> largeData(largeN);
        for (size_t i = 0; i < largeN; i++) largeData[i] = 1;  // All 1s

        // Sequential
        auto t0 = std::chrono::high_resolution_clock::now();
        auto largeSeq = sequentialScan(largeData);
        auto t1 = std::chrono::high_resolution_clock::now();
        double timeSeq = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Work-efficient
        std::vector<int> blellochData = largeData;
        auto t2 = std::chrono::high_resolution_clock::now();
        workEfficientScan(blellochData);
        auto t3 = std::chrono::high_resolution_clock::now();
        double timeBlelloch = std::chrono::duration<double, std::milli>(t3 - t2).count();

        // Multi-core (8 workers)
        auto t4 = std::chrono::high_resolution_clock::now();
        auto largeMC = multiCoreScan(largeData, 8);
        auto t5 = std::chrono::high_resolution_clock::now();
        double timeMC = std::chrono::duration<double, std::milli>(t5 - t4).count();

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Sequential scan:     " << timeSeq << " ms\n";
        std::cout << "  Work-efficient scan: " << timeBlelloch << " ms\n";
        std::cout << "  Multi-core scan (8): " << timeMC << " ms\n";

        // Verify correctness
        bool mcOk = true;
        for (size_t i = 0; i < largeN && mcOk; i++) {
            if (largeMC[i] != largeSeq[i]) mcOk = false;
        }
        std::cout << "  Multi-core correctness: " << (mcOk ? "PASSED" : "FAILED") << "\n";
    }

    std::cout << "\n==================================================\n";
    std::cout << "Key concepts demonstrated:\n";
    std::cout << "  - Naive scan: O(N log N) work, better SIMD utilization\n";
    std::cout << "  - Blelloch scan: O(N) work, up-sweep + down-sweep\n";
    std::cout << "  - Warp scan: SIMD-friendly for small arrays (32)\n";
    std::cout << "  - Multi-core scan: partition + sequential + add bases\n";
    std::cout << "  - Different strategies at different levels of the machine\n";
    std::cout << "==================================================\n";

    return 0;
}
