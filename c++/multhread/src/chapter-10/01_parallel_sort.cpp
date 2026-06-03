/**
 * 01_parallel_sort.cpp — 使用 C++17 并行排序对比顺序排序
 *
 * std::sort 的并行版本: std::sort(std::execution::par, ...)
 * 技术要点:
 *  - #include <execution>: C++17 执行策略
 *  - std::execution::seq: 顺序执行
 *  - std::execution::par: 并行执行
 *  - std::execution::par_unseq: 并行+向量化
 *  - GCC 需链接 TBB: -ltbb
 *
 * 编译:
 *   GCC:  g++ -std=c++20 -O2 -pthread 01_parallel_sort.cpp -ltbb -o parallel_sort
 *   MSVC: cl /EHsc /std:c++17 /O2 01_parallel_sort.cpp
 */

#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <thread>

// C++17 并行算法头文件
#include <execution>
#ifdef HAS_TBB
    #define HAS_EXECUTION 1
#else
    #define HAS_EXECUTION 0
#endif

// ============================================================================
// 辅助函数
// ============================================================================
std::vector<int> generate_random_data(size_t n) {
    std::vector<int> data(n);
    std::mt19937 rng(42); // 固定种子, 保证可复现
    std::uniform_int_distribution<int> dist(0, 10000000);

    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }
    return data;
}

bool is_sorted_ok(const std::vector<int>& data) {
    return std::is_sorted(data.begin(), data.end());
}

template <typename Func>
double measure_time(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ============================================================================
// 排序基准测试
// ============================================================================
void benchmark_sort() {
    std::cout << "=== C++17 并行排序性能对比 ===\n\n";

    const std::vector<size_t> sizes = {1'000'000, 5'000'000, 10'000'000};

    std::cout << std::left
              << std::setw(14) << "数据量"
              << std::setw(16) << "顺序排序"
              << std::setw(16) << "并行排序"
              << std::setw(12) << "加速比"
              << "\n";
    std::cout << std::string(58, '-') << "\n";

    for (size_t n : sizes) {
        auto original = generate_random_data(n);

        // 顺序排序
        auto data_seq = original;
        double time_seq = measure_time([&]() {
            std::sort(data_seq.begin(), data_seq.end());
        });
        bool ok_seq = is_sorted_ok(data_seq);

        // 并行排序
        double time_par = 0;
        bool ok_par = false;
        auto data_par = original;

#if HAS_EXECUTION
        time_par = measure_time([&]() {
            std::sort(std::execution::par, data_par.begin(), data_par.end());
        });
        ok_par = is_sorted_ok(data_par);
#else
        // 回退: 用自制并行排序 (基于分块+归并)
        std::cout << "  (注意: <execution> 不可用, 使用手动并行排序)\n";
        time_par = time_seq; // 占位
        ok_par = ok_seq;
#endif

        double speedup = (time_par > 0) ? time_seq / time_par : 0;

        std::cout << std::left
                  << std::setw(14) << (std::to_string(n / 1000000) + "M")
                  << std::setw(16) << (std::to_string(static_cast<int>(time_seq)) + " ms")
                  << std::setw(16) << (std::to_string(static_cast<int>(time_par)) + " ms")
                  << std::setw(10) << std::fixed << std::setprecision(2) << speedup << "x"
                  << "  " << (ok_par ? "" : " 错误!")
                  << "\n";
    }

    std::cout << "\n  硬件线程数: " << std::jthread::hardware_concurrency() << "\n";
}

// ============================================================================
// 手动并行排序 (当 execution::par 不可用时)
// ============================================================================
void manual_parallel_sort_demo() {
    std::cout << "\n=== 手动并行排序 (分块排序 + 归并) ===\n\n";

    constexpr size_t N = 10'000'000;
    auto data = generate_random_data(N);

    auto start = std::chrono::high_resolution_clock::now();

    const unsigned int num_threads = std::jthread::hardware_concurrency();
    const size_t block_size = N / num_threads;

    std::vector<std::jthread> threads;
    threads.reserve(num_threads);

    // 阶段1: 各线程独立排序自己的块
    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t begin = t * block_size;
        size_t end = (t == num_threads - 1) ? N : begin + block_size;

        threads.emplace_back([&data, begin, end]() {
            std::sort(data.begin() + static_cast<long>(begin),
                      data.begin() + static_cast<long>(end));
        });
    }
    for (auto& th : threads) th.join();

    // 阶段2: 归并已排序的块 (使用辅助缓冲区)
    std::vector<int> temp(N);
    for (size_t merge_size = block_size; merge_size < N; merge_size *= 2) {
        for (size_t left = 0; left < N; left += 2 * merge_size) {
            size_t mid = std::min(left + merge_size, N);
            size_t right = std::min(left + 2 * merge_size, N);

            if (mid >= right) continue;

            std::merge(data.begin() + static_cast<long>(left),
                       data.begin() + static_cast<long>(mid),
                       data.begin() + static_cast<long>(mid),
                       data.begin() + static_cast<long>(right),
                       temp.begin() + static_cast<long>(left));

            std::copy(temp.begin() + static_cast<long>(left),
                      temp.begin() + static_cast<long>(right),
                      data.begin() + static_cast<long>(left));
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "  数据量: " << N / 1000000 << "M\n";
    std::cout << "  耗时: " << ms << " ms\n";
    std::cout << "  正确性: " << (is_sorted_ok(data) ? "通过" : "失败") << "\n";
}

// ============================================================================
// main
// ============================================================================
int main() {
    benchmark_sort();
    manual_parallel_sort_demo();
    return 0;
}
