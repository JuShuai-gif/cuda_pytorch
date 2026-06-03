/**
 * 01_parallel_for_each.cpp — 并行 for_each 实现
 *
 * 将数据分块, 多线程并行处理每个元素。
 * 技术要点:
 *  - 计算最优线程数 (hardware_concurrency)
 *  - 均匀分块, 每线程处理连续范围 (缓存友好)
 *  - std::jthread 自动 join
 *  - 支持索引访问和迭代器版本
 *
 * 编译: g++ -std=c++20 -O2 -pthread 01_parallel_for_each.cpp -o parallel_for_each
 */

#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
#include <chrono>
#include <cmath>
#include <mutex>

// ============================================================================
// parallel_for_each — 按索引分块的并行处理
// ============================================================================
template <typename Index, typename Func>
void parallel_for_each(Index first, Index last, Func&& func) {
    const auto length = static_cast<size_t>(last - first);
    if (length == 0) return;

    // 计算线程数: 不超过元素数, 不超过硬件线程数
    const unsigned int hw_threads = std::jthread::hardware_concurrency();
    const unsigned int num_threads = std::min(
        hw_threads,
        static_cast<unsigned int>(length));

    // 每个线程处理的元素数
    const size_t block_size = length / num_threads;
    const size_t remainder = length % num_threads;

    std::vector<std::jthread> threads;
    threads.reserve(num_threads);

    Index block_start = first;
    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t current_block = block_size + (t < remainder ? 1 : 0);
        Index block_end = block_start + static_cast<Index>(current_block);

        threads.emplace_back([block_start, block_end, &func]() {
            for (Index i = block_start; i < block_end; ++i) {
                func(i);
            }
        });

        block_start = block_end;
    }
    // jthread 析构时自动 join
}

// ============================================================================
// parallel_for_each — 迭代器版本
// ============================================================================
template <typename Iterator, typename Func>
void parallel_for_each_iter(Iterator first, Iterator last, Func&& func) {
    const auto length = static_cast<size_t>(std::distance(first, last));
    if (length == 0) return;

    const unsigned int hw_threads = std::jthread::hardware_concurrency();
    const unsigned int num_threads = std::min(hw_threads, static_cast<unsigned int>(length));
    const size_t block_size = length / num_threads;
    const size_t remainder = length % num_threads;

    std::vector<std::jthread> threads;
    threads.reserve(num_threads);

    Iterator block_start = first;
    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t current_block = block_size + (t < remainder ? 1 : 0);
        Iterator block_end = std::next(block_start, static_cast<long>(current_block));

        threads.emplace_back([block_start, block_end, &func]() {
            for (auto it = block_start; it != block_end; ++it) {
                func(*it);
            }
        });

        block_start = block_end;
    }
}

// ============================================================================
// 性能对比测试
// ============================================================================
void benchmark() {
    std::cout << "=== 并行 for_each 性能对比 ===\n\n";

    constexpr size_t N = 50'000'000;
    std::vector<double> data(N);
    std::iota(data.begin(), data.end(), 0.0);

    // 顺序版本
    {
        std::vector<double> copy = data;
        auto start = std::chrono::high_resolution_clock::now();
        std::for_each(copy.begin(), copy.end(), [](double& x) {
            x = std::sqrt(x) * std::log(x + 1.0) * std::sin(x * 0.001);
        });
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "  顺序 for_each: " << ms << " ms\n";
    }

    // 并行版本 (索引)
    {
        std::vector<double> copy = data;
        auto start = std::chrono::high_resolution_clock::now();
        parallel_for_each(size_t(0), copy.size(), [&](size_t i) {
            copy[i] = std::sqrt(copy[i]) * std::log(copy[i] + 1.0) * std::sin(copy[i] * 0.001);
        });
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "  并行 for_each (索引): " << ms << " ms\n";
    }

    // 并行版本 (迭代器)
    {
        std::vector<double> copy = data;
        auto start = std::chrono::high_resolution_clock::now();
        parallel_for_each_iter(copy.begin(), copy.end(), [](double& x) {
            x = std::sqrt(x) * std::log(x + 1.0) * std::sin(x * 0.001);
        });
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "  并行 for_each (迭代器): " << ms << " ms\n";
    }

    unsigned int hw = std::jthread::hardware_concurrency();
    std::cout << "\n  硬件线程数: " << hw << "\n";
}

// ============================================================================
// 正确性验证
// ============================================================================
void correctness_test() {
    std::cout << "\n=== 正确性验证 ===\n";

    constexpr size_t N = 1000000;
    std::vector<int> data(N, 0);

    // 每个元素设置为索引值
    parallel_for_each(size_t(0), data.size(), [&](size_t i) {
        data[i] = static_cast<int>(i) * 2;
    });

    // 验证
    bool ok = true;
    for (size_t i = 0; i < N; ++i) {
        if (data[i] != static_cast<int>(i) * 2) {
            std::cerr << "  错误: data[" << i << "] = " << data[i] << "\n";
            ok = false;
            break;
        }
    }
    std::cout << "  结果: " << (ok ? "通过" : "失败") << "\n";
}

// ============================================================================
// main
// ============================================================================
int main() {
    benchmark();
    correctness_test();
    return 0;
}
