// lecture8_part1.cpp
// Stanford CS149, Lecture 8: Data-Parallel Thinking
// Part 1: Map and Reduce (Fold) Operations
//
// Implements the core data-parallel primitives:
//   - map: apply function to all elements (trivially parallel)
//   - reduce/fold: combine elements with binary associative operator
//   - filter: select elements matching predicate
//   - parallel histogram via map + sort
//
// Compile: g++ -std=c++17 -pthread lecture8_part1.cpp -o lecture8_part1
// Run: ./lecture8_part1

#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <functional>
#include <chrono>
#include <cassert>

// ============================================================================
// Utility: divide work into chunks for parallel execution
// ============================================================================

struct WorkRange {
    size_t start;
    size_t end;
};

std::vector<WorkRange> partitionWork(size_t total, size_t numWorkers) {
    std::vector<WorkRange> ranges(numWorkers);
    size_t chunkSize = (total + numWorkers - 1) / numWorkers;
    for (size_t i = 0; i < numWorkers; i++) {
        ranges[i].start = i * chunkSize;
        ranges[i].end   = std::min(ranges[i].start + chunkSize, total);
    }
    return ranges;
}

// ============================================================================
// 1. Parallel Map
// Applies function f to all elements of input, writing to output.
// Trivially parallel because f is side-effect-free (pure function).
// ============================================================================

template<typename InputIt, typename OutputIt, typename UnaryFunc>
void parallelMap(InputIt first, InputIt last, OutputIt d_first,
                 UnaryFunc f, size_t numThreads)
{
    size_t n = static_cast<size_t>(std::distance(first, last));
    auto ranges = partitionWork(n, numThreads);

    std::vector<std::thread> workers;
    for (const auto& r : ranges) {
        workers.emplace_back([first, d_first, f, r]() {
            for (size_t i = r.start; i < r.end; i++) {
                *(d_first + i) = f(*(first + i));
            }
        });
    }

    for (auto& t : workers) t.join();
}

// ============================================================================
// 2. Parallel Reduce (Fold)
// Applies binary associative operation to combine all elements.
// Uses a two-phase approach:
//   Phase 1: Each worker computes local partial result
//   Phase 2: Combine partial results
// ============================================================================

template<typename InputIt, typename T, typename BinaryFunc>
T parallelReduce(InputIt first, InputIt last, T identity, BinaryFunc f,
                 size_t numThreads)
{
    size_t n = static_cast<size_t>(std::distance(first, last));
    auto ranges = partitionWork(n, numThreads);

    // Phase 1: Each worker computes its partial result
    std::vector<T> partials(numThreads, identity);
    std::vector<std::thread> workers;

    for (size_t w = 0; w < numThreads; w++) {
        workers.emplace_back([first, f, identity, w, &ranges, &partials]() {
            T local = identity;
            for (size_t i = ranges[w].start; i < ranges[w].end; i++) {
                local = f(local, *(first + i));
            }
            partials[w] = local;
        });
    }

    for (auto& t : workers) t.join();

    // Phase 2: Combine partial results (sequential — only numThreads elements)
    T result = identity;
    for (size_t w = 0; w < numThreads; w++) {
        result = f(result, partials[w]);
    }
    return result;
}

// ============================================================================
// 3. Parallel Filter
// Keeps elements that satisfy predicate, output to new sequence.
// Two passes: count matching elements (allocation), then copy.
// ============================================================================

template<typename InputIt, typename OutputIt, typename Predicate>
size_t parallelFilter(InputIt first, InputIt last, OutputIt d_first,
                      Predicate pred, size_t numThreads)
{
    size_t n = static_cast<size_t>(std::distance(first, last));
    auto ranges = partitionWork(n, numThreads);

    // Phase 1: Compute prefix sum of match counts to determine output offsets
    std::vector<size_t> matchCounts(numThreads, 0);
    std::vector<std::thread> workers;

    for (size_t w = 0; w < numThreads; w++) {
        workers.emplace_back([first, pred, w, &ranges, &matchCounts]() {
            size_t count = 0;
            for (size_t i = ranges[w].start; i < ranges[w].end; i++) {
                if (pred(*(first + i))) count++;
            }
            matchCounts[w] = count;
        });
    }
    for (auto& t : workers) t.join();

    // Compute offsets (exclusive prefix sum of matchCounts)
    std::vector<size_t> offsets(numThreads, 0);
    for (size_t w = 1; w < numThreads; w++) {
        offsets[w] = offsets[w - 1] + matchCounts[w - 1];
    }
    size_t totalMatches = offsets.back() + matchCounts.back();

    // Phase 2: Write matched elements to output at computed offsets
    workers.clear();
    for (size_t w = 0; w < numThreads; w++) {
        workers.emplace_back([first, d_first, pred, w, &ranges, &offsets]() {
            size_t pos = offsets[w];
            for (size_t i = ranges[w].start; i < ranges[w].end; i++) {
                if (pred(*(first + i))) {
                    *(d_first + pos) = *(first + i);
                    pos++;
                }
            }
        });
    }
    for (auto& t : workers) t.join();

    return totalMatches;
}

// ============================================================================
// 4. Parallel Histogram via Map + Sort (Data-parallel approach)
// ============================================================================

std::vector<int> parallelHistogram(const std::vector<int>& data,
                                   int numBins, size_t numThreads)
{
    size_t n = data.size();

    // Step 1: Map — compute bin for each element
    std::vector<int> binIds(n);
    parallelMap(data.begin(), data.end(), binIds.begin(),
                [numBins](int v) {
                    int bin = v % numBins;
                    return (bin < 0) ? bin + numBins : bin;
                },
                numThreads);

    // Step 2: Sort binIds (SIMD-style parallel sort not shown — use std::sort)
    // In a real GPU implementation, efficient parallel sort would be used
    std::vector<int> sortedBinIds = binIds;
    std::sort(sortedBinIds.begin(), sortedBinIds.end());

    // Step 3: Count elements per bin (using pre-sorted data)
    std::vector<int> histogram(numBins, 0);
    for (size_t i = 0; i < sortedBinIds.size(); i++) {
        histogram[sortedBinIds[i]]++;
    }

    return histogram;
}

// ============================================================================
// Demonstrations
// ============================================================================

void demoMap()
{
    std::cout << "--- 1. Parallel Map ---\n";

    std::vector<int> input  = {3, 8, 4, 6, 3, 9, 2, 8};
    std::vector<int> output(input.size());

    // f(x) = x + 10 (same as lecture example)
    parallelMap(input.begin(), input.end(), output.begin(),
                [](int x) { return x + 10; }, 4);

    std::cout << "Input:  ";
    for (int v : input)  std::cout << v << " ";
    std::cout << "\n";
    std::cout << "map +10: ";
    for (int v : output) std::cout << v << " ";
    std::cout << "\n";

    // Verify against std::transform
    std::vector<int> expected(input.size());
    std::transform(input.begin(), input.end(), expected.begin(),
                   [](int x) { return x + 10; });
    bool ok = (output == expected);
    std::cout << "Verification: " << (ok ? "PASSED" : "FAILED") << "\n\n";
}

void demoReduce()
{
    std::cout << "--- 2. Parallel Reduce (Fold) ---\n";

    std::vector<int> data = {3, 8, 4, 6, 3, 9, 2, 8};

    // fold 10 (+) data = 10+3+8+4+6+3+9+2+8 = 53
    int result = parallelReduce(data.begin(), data.end(), 10,
                                std::plus<int>(), 4);

    std::cout << "Data: ";
    for (int v : data) std::cout << v << " ";
    std::cout << "\n";
    std::cout << "fold 10 (+) data = " << result << "\n";

    // Verify
    int expected = 10 + std::accumulate(data.begin(), data.end(), 0);
    std::cout << "Verification: " << (result == expected ? "PASSED" : "FAILED")
              << " (expected " << expected << ")\n\n";
}

void demoFilter()
{
    std::cout << "--- 3. Parallel Filter ---\n";

    std::vector<int> data = {3, 8, 4, 6, 3, 9, 2, 8};
    std::vector<int> output(data.size());

    // Keep only even numbers
    size_t count = parallelFilter(data.begin(), data.end(), output.begin(),
                                  [](int x) { return x % 2 == 0; }, 4);
    output.resize(count);

    std::cout << "Input:     ";
    for (int v : data) std::cout << v << " ";
    std::cout << "\n";
    std::cout << "filter even: ";
    for (size_t i = 0; i < count; i++) std::cout << output[i] << " ";
    std::cout << "\n";
    std::cout << "(Filtered " << data.size() - count << " elements)\n\n";
}

void demoHistogram()
{
    std::cout << "--- 4. Parallel Histogram (Map + Sort approach) ---\n";

    std::vector<int> data = {0, 3, 4, 1, 9, 2, 8, 4, 1, 7,
                             5, 6, 2, 3, 9, 0, 1, 5, 8, 4};
    constexpr int NUM_BINS = 10;

    auto hist = parallelHistogram(data, NUM_BINS, 4);

    std::cout << "Data: ";
    for (int v : data) std::cout << v << " ";
    std::cout << "\nHistogram (data-parallel via map+sort):\n";
    for (int b = 0; b < NUM_BINS; b++) {
        std::cout << "  bin[" << b << "]: " << hist[b] << "  ";
        for (int i = 0; i < hist[b]; i++) std::cout << "#";
        std::cout << "\n";
    }

    // Verify by counting directly
    bool ok = true;
    for (int b = 0; b < NUM_BINS; b++) {
        int expected = std::count_if(data.begin(), data.end(),
                                     [b](int v) { return v % NUM_BINS == b; });
        if (hist[b] != expected) ok = false;
    }
    std::cout << "Verification: " << (ok ? "PASSED" : "FAILED") << "\n\n";
}

// ============================================================================
// main
// ============================================================================

int main()
{
    std::cout << "==================================================\n";
    std::cout << "Lecture 8 Part 1: Map, Reduce, Filter, Histogram\n";
    std::cout << "==================================================\n\n";

    demoMap();
    demoReduce();
    demoFilter();
    demoHistogram();

    std::cout << "==================================================\n";
    std::cout << "Key concepts demonstrated:\n";
    std::cout << "  - map: side-effect-free function → trivially parallel\n";
    std::cout << "  - reduce: associative operator → partial results then combine\n";
    std::cout << "  - filter: 2-pass approach with prefix sum for offsets\n";
    std::cout << "  - histogram: map f → sort → count (parallel-friendly)\n";
    std::cout << "==================================================\n";

    return 0;
}
