/**
 * 02_parallel_find.cpp — 并行 find 实现, 支持提前退出
 *
 * 使用分治策略递归划分搜索空间, std::async 异步执行。
 * 技术要点:
 *  - atomic<bool> 标志: 找到后通知其他线程停止
 *  - 递归二分: 直到子范围小于阈值时顺序查找
 *  - std::async + std::future 携带结果
 *  - 避免伪共享: done 标志独立缓存行
 *
 * 编译: g++ -std=c++20 -O2 -pthread 02_parallel_find.cpp -o parallel_find
 */

#include <iostream>
#include <thread>
#include <vector>
#include <future>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <random>
#include <optional>

// ============================================================================
// parallel_find — 并行查找
// 返回找到的第一个匹配元素的迭代器
// ============================================================================
template <typename Iterator, typename Predicate>
Iterator parallel_find(Iterator first, Iterator last, Predicate pred) {
    const auto length = static_cast<size_t>(std::distance(first, last));
    if (length == 0) return last;

    // 原子标志: 一旦找到, 通知所有线程停止
    struct SearchState {
        alignas(64) std::atomic<bool> found{false}; // 缓存行对齐
        Iterator result;
        std::mutex result_mutex;
    };
    auto state = std::make_shared<SearchState>();

    // 递归异步查找
    std::function<Iterator(Iterator, Iterator, size_t)> async_find =
        [&](Iterator begin, Iterator end, size_t depth) -> Iterator {

        const auto len = static_cast<size_t>(std::distance(begin, end));

        // 已找到, 立即返回
        if (state->found.load(std::memory_order_acquire)) {
            return end;
        }

        // 基础情况: 范围小或深度过深, 顺序查找
        constexpr size_t kThreshold = 10000;
        constexpr size_t kMaxDepth = 10;

        if (len <= kThreshold || depth >= kMaxDepth) {
            auto it = std::find_if(begin, end, pred);
            if (it != end) {
                state->found.store(true, std::memory_order_release);
                std::lock_guard<std::mutex> lock(state->result_mutex);
                state->result = it;
            }
            return it;
        }

        // 二分
        Iterator mid = std::next(begin, static_cast<long>(len / 2));

        // 异步处理前半
        auto future_first = std::async(std::launch::async,
            [&async_find, begin, mid, depth]() {
                return async_find(begin, mid, depth + 1);
            });

        // 当前线程处理后半
        Iterator result_second = async_find(mid, end, depth + 1);

        // 等待前半
        Iterator result_first = future_first.get();

        if (result_first != end) return result_first;
        if (result_second != end) return result_second;

        // 检查是否被其他线程找到
        if (state->found.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lock(state->result_mutex);
            return state->result;
        }

        return end;
    };

    return async_find(first, last, 0);
}

// ============================================================================
// parallel_find_any — 更简单的提前退出版本 (使用 atomic_flag + 分块)
// ============================================================================
template <typename Iterator, typename Predicate>
Iterator parallel_find_chunked(Iterator first, Iterator last, Predicate pred) {
    const auto length = static_cast<size_t>(std::distance(first, last));
    if (length == 0) return last;

    const unsigned int hw_threads = std::jthread::hardware_concurrency();
    const unsigned int num_threads = std::min(hw_threads, static_cast<unsigned int>(length));
    const size_t block_size = length / num_threads;
    const size_t remainder = length % num_threads;

    alignas(64) std::atomic<bool> found{false};
    Iterator found_pos = last;
    std::mutex found_mutex;

    std::vector<std::jthread> threads;
    threads.reserve(num_threads);

    Iterator block_start = first;
    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t current_block = block_size + (t < remainder ? 1 : 0);
        Iterator block_end = std::next(block_start, static_cast<long>(current_block));

        threads.emplace_back([block_start, block_end, &pred, &found, &found_pos, &found_mutex]() {
            for (auto it = block_start; it != block_end; ++it) {
                if (found.load(std::memory_order_acquire)) {
                    return; // 其他线程已找到, 提前退出
                }
                if (pred(*it)) {
                    found.store(true, std::memory_order_release);
                    std::lock_guard<std::mutex> lock(found_mutex);
                    found_pos = it;
                    return;
                }
            }
        });

        block_start = block_end;
    }
    // jthread 自动 join
    threads.clear();

    return found_pos;
}

// ============================================================================
// 性能对比测试
// ============================================================================
void benchmark() {
    std::cout << "=== 并行 find 性能对比 ===\n\n";

    constexpr size_t N = 100'000'000;
    std::vector<int> data(N);
    std::iota(data.begin(), data.end(), 0);

    // 查找的值位于尾部 (最坏情况)
    const int target = static_cast<int>(N - 100);

    // 顺序查找
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto it = std::find(data.begin(), data.end(), target);
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "  顺序 find:        " << ms << " ms";
        std::cout << " (结果: " << *it << ")\n";
    }

    // 并行分块查找 (提前退出)
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto it = parallel_find_chunked(data.begin(), data.end(),
            [target](int x) { return x == target; });
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "  并行 find (分块):  " << ms << " ms";
        if (it != data.end()) {
            std::cout << " (结果: " << *it << ")\n";
        } else {
            std::cout << " (未找到)\n";
        }
    }

    // 并行递归查找 (早期版本 — 无提前退出优化)
    {
        // 仅在小数据集上测试, 避免过多 async 开销
        std::vector<int> small_data(1'000'000);
        std::iota(small_data.begin(), small_data.end(), 0);
        int small_target = 500000;

        auto start = std::chrono::high_resolution_clock::now();
        auto it = parallel_find(small_data.begin(), small_data.end(),
            [small_target](int x) { return x == small_target; });
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "  并行 find (递归):  " << ms << " ms";
        std::cout << " (结果: " << *it << ")\n";
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
