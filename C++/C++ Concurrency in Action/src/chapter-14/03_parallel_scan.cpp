// 03_parallel_scan.cpp — 并行前缀和 (Parallel Prefix Sum)
// 演示: 两阶段并行 scan、与串行对比

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

// ===== 1. 串行前缀和 (参考实现) =====
template <typename It>
void serial_scan(It first, It last, It output) {
    if (first == last) return;
    typename std::iterator_traits<It>::value_type sum = *first;
    *output++ = sum;
    for (auto it = first + 1; it != last; ++it) {
        sum += *it;
        *output++ = sum;
    }
}

// ===== 2. 两阶段并行前缀和 =====
template <typename It>
void parallel_scan_two_phase(It first, It last, It output) {
    using T = typename std::iterator_traits<It>::value_type;
    const size_t n = std::distance(first, last);
    if (n == 0) return;

    const unsigned num_threads = std::thread::hardware_concurrency();
    const size_t chunk_size = std::max(size_t(1), n / num_threads);

    // Phase 1: 每线程计算 chunk 内前缀和，并记录每 chunk 的总和
    std::vector<T> chunk_sums(num_threads);
    std::vector<std::jthread> threads;

    for (unsigned t = 0; t < num_threads; ++t) {
        size_t begin = t * chunk_size;
        size_t end = (t == num_threads - 1) ? n : begin + chunk_size;

        threads.emplace_back([&, t, begin, end]() {
            T local_sum = 0;
            for (size_t i = begin; i < end; ++i) {
                local_sum += first[i];
                output[i] = local_sum;
            }
            chunk_sums[t] = local_sum;
        });
    }
    threads.clear();

    // Phase 2: 计算跨 chunk 的偏移量并修正
    // chunk_sums[0] 不需要修正，从 chunk 1 开始
    for (unsigned t = 1; t < num_threads; ++t) {
        T offset = chunk_sums[t - 1];
        chunk_sums[t] += offset; // 前缀化 chunk_sums

        size_t begin = t * chunk_size;
        size_t end = (t == num_threads - 1) ? n : begin + chunk_size;

        threads.emplace_back([&, begin, end, offset]() {
            for (size_t i = begin; i < end; ++i) {
                output[i] += offset;
            }
        });
    }
    threads.clear();
}

// ===== 3. 性能对比 =====
void benchmark_scan() {
    std::cout << "=== 并行前缀和 (Parallel Scan) ===\n";

    const size_t kSize = 5'000'000;
    std::vector<int> input(kSize, 1); // 全 1，前缀和 = 1, 2, 3, ...

    std::vector<int> output_serial(kSize);
    std::vector<int> output_parallel(kSize);

    // 串行
    {
        auto start = std::chrono::high_resolution_clock::now();
        serial_scan(input.begin(), input.end(), output_serial.begin());
        auto elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - start);
        std::cout << "  串行 scan:  " << elapsed.count() << " us\n";
    }

    // 并行 (两阶段)
    {
        auto start = std::chrono::high_resolution_clock::now();
        parallel_scan_two_phase(
            input.begin(), input.end(), output_parallel.begin());
        auto elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - start);
        std::cout << "  并行 scan:  " << elapsed.count() << " us\n";
    }

    // 正确性验证
    bool correct = (output_serial == output_parallel);
    std::cout << "  结果正确: " << (correct ? "OK" : "FAIL") << "\n";
    if (correct) {
        std::cout << "  验证: output[0]=" << output_parallel[0]
                  << ", output[last]=" << output_parallel[kSize - 1]
                  << " (期望 " << kSize << ")\n";
    }
}

// ===== 4. 更多 scan 示例: 最大值前缀 =====
void demo_max_scan() {
    std::cout << "\n=== 最大值前缀扫描 ===\n";

    std::vector<int> data = {3, 1, 4, 1, 5, 9, 2, 6};
    std::vector<int> result(data.size());

    std::inclusive_scan(data.begin(), data.end(), result.begin(),
        [](int a, int b) { return std::max(a, b); });

    std::cout << "  Input:     ";
    for (int v : data) std::cout << v << " ";
    std::cout << "\n  Max-Scan:  ";
    for (int v : result) std::cout << v << " ";
    std::cout << "\n  (每位置=该位置及之前所有元素的最大值)\n";
}

int main() {
    benchmark_scan();
    demo_max_scan();

    std::cout << "\n前缀和是许多并行算法的基础（排序、稀疏矩阵等）。\n";
    return 0;
}
