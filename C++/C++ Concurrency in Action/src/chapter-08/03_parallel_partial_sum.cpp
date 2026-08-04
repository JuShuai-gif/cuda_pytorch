/**
 * 03_parallel_partial_sum.cpp — 并行前缀和 (Parallel Partial Sum / Scan)
 *
 * 分两阶段:
 *  阶段1 (并行): 每个线程独立计算其分块的局部和, 存储于中间数组
 *  阶段2 (顺序): 主线程计算跨块前缀偏移, 传递到后续块
 *  阶段3 (并行): 每个线程将偏移加到自己的分块上
 *
 * 技术要点:
 *  - 分块并行计算 -> barrier -> 跨块前缀传递 -> 并行加偏移
 *  - std::barrier (C++20) 或自定义 barrier
 *  - 性能接近 O(n/p + log p)
 *
 * 编译: g++ -std=c++20 -O2 -pthread 03_parallel_partial_sum.cpp -o parallel_partial_sum
 */

#include <iostream>
#include <thread>
#include <vector>
#include <numeric>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <functional>
#include <mutex>
#include <condition_variable>

// ============================================================================
// SimpleBarrier — C++17 兼容的简易 barrier
// ============================================================================
class SimpleBarrier {
private:
    std::mutex mutex_;
    std::condition_variable cv_;
    unsigned int count_;
    unsigned int generation_{0};
    unsigned int waiting_{0};

public:
    explicit SimpleBarrier(unsigned int count) : count_(count) {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        unsigned int gen = generation_;
        ++waiting_;

        if (waiting_ == count_) {
            waiting_ = 0;
            ++generation_;
            cv_.notify_all();
        } else {
            cv_.wait(lock, [this, gen] {
                return gen != generation_;
            });
        }
    }
};

// ============================================================================
// parallel_partial_sum — 并行前缀和
// ============================================================================
template <typename InputIt, typename OutputIt>
void parallel_partial_sum(InputIt first, InputIt last, OutputIt d_first) {
    using ValueType = typename std::iterator_traits<InputIt>::value_type;

    const auto n = static_cast<size_t>(std::distance(first, last));
    if (n == 0) return;

    const unsigned int num_threads = std::min(
        std::jthread::hardware_concurrency(),
        static_cast<unsigned int>(n));
    const size_t block_size = n / num_threads;
    const size_t remainder = n % num_threads;

    // 每个块的局部和
    std::vector<ValueType> block_sums(num_threads, ValueType{0});

    std::vector<std::jthread> threads;
    threads.reserve(num_threads);

    // 阶段1: 每个线程并行计算自己块内的前缀和 + 块总和
    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t start_idx = t * block_size + std::min(static_cast<size_t>(t), remainder);
        size_t my_block_size = block_size + (t < remainder ? 1 : 0);

        threads.emplace_back([=, &block_sums]() {
            auto it_in = std::next(first, static_cast<long>(start_idx));
            auto it_out = std::next(d_first, static_cast<long>(start_idx));

            // 块内前缀和 (顺序)
            ValueType running = ValueType{0};
            for (size_t i = 0; i < my_block_size; ++i, ++it_in, ++it_out) {
                running += *it_in;
                *it_out = running;
            }
            block_sums[t] = running;
        });
    }

    // 等待阶段1完成
    for (auto& th : threads) th.join();
    threads.clear();

    // 阶段2: 主线程计算跨块前缀 (顺序 — O(num_threads))
    {
        ValueType offset = ValueType{0};
        for (unsigned int t = 0; t < num_threads; ++t) {
            ValueType block_total = block_sums[t];
            block_sums[t] = offset; // 存储该块的起始偏移而非总和
            offset += block_total;
        }
    }

    // 阶段3: 每个线程将偏移加到自己的块上
    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t start_idx = t * block_size + std::min(static_cast<size_t>(t), remainder);
        size_t my_block_size = block_size + (t < remainder ? 1 : 0);

        threads.emplace_back([=, &block_sums]() {
            ValueType offset = block_sums[t];
            auto it_out = std::next(d_first, static_cast<long>(start_idx));
            for (size_t i = 0; i < my_block_size; ++i, ++it_out) {
                *it_out += offset;
            }
        });
    }

    for (auto& th : threads) th.join();
}

// ============================================================================
// 性能对比与正确性验证
// ============================================================================
void benchmark() {
    std::cout << "=== 并行前缀和 (Partial Sum) ===\n\n";

    constexpr size_t N = 100'000'000;
    std::vector<long long> input(N, 1); // 全1数组: 前缀和 = 1,2,3,...
    std::vector<long long> output_serial(N);
    std::vector<long long> output_parallel(N);

    // 顺序版本
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::partial_sum(input.begin(), input.end(), output_serial.begin());
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "  顺序 partial_sum: " << ms << " ms\n";
    }

    // 并行版本
    {
        auto start = std::chrono::high_resolution_clock::now();
        parallel_partial_sum(input.begin(), input.end(), output_parallel.begin());
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "  并行 partial_sum: " << ms << " ms\n";
    }

    // 正确性验证
    bool correct = (output_serial == output_parallel);
    std::cout << "\n  正确性: " << (correct ? "通过" : "失败") << "\n";

    if (correct) {
        std::cout << "  首5个值: ";
        for (size_t i = 0; i < 5; ++i) std::cout << output_parallel[i] << " ";
        std::cout << "\n";
        std::cout << "  尾5个值: ";
        for (size_t i = N - 5; i < N; ++i) std::cout << output_parallel[i] << " ";
        std::cout << "\n";
    }

    std::cout << "\n  硬件线程数: " << std::jthread::hardware_concurrency() << "\n";
}

// ============================================================================
// main
// ============================================================================
int main() {
    benchmark();
    return 0;
}
