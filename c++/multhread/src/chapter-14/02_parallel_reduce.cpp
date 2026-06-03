// 02_parallel_reduce.cpp — 手动实现 Parallel Reduce
// 演示: 分治归约、chunk 划分、与串行对比

#include <chrono>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

// ===== 1. 基础并行归约 =====
template <typename It, typename T, typename BinaryOp>
T parallel_reduce_basic(It first, It last, T init, BinaryOp op) {
    const size_t n = std::distance(first, last);
    if (n == 0) return init;

    const unsigned num_threads = std::thread::hardware_concurrency();
    const size_t chunk_size = std::max(size_t(1), n / num_threads);

    std::vector<T> partial_results(num_threads, init);
    std::vector<std::jthread> threads;

    for (unsigned t = 0; t < num_threads; ++t) {
        auto begin = first + t * chunk_size;
        auto end = (t == num_threads - 1) ? last : begin + chunk_size;

        threads.emplace_back([&partial_results, t, begin, end, op]() {
            T local = partial_results[t];
            for (auto it = begin; it != end; ++it) {
                local = op(local, *it);
            }
            partial_results[t] = local;
        });
    }
    threads.clear();

    T result = init;
    for (const auto& pr : partial_results) {
        result = op(result, pr);
    }
    return result;
}

// ===== 2. 树型归约 (更少的合并操作) =====
template <typename T, typename BinaryOp>
T tree_reduce(std::vector<T>& data, size_t begin, size_t end, BinaryOp op) {
    if (end - begin <= 1) return data[begin];
    size_t mid = begin + (end - begin) / 2;
    T left = tree_reduce(data, begin, mid, op);
    T right = tree_reduce(data, mid, end, op);
    return op(left, right);
}

// ===== 3. 性能对比 =====
void benchmark_reduce() {
    std::cout << "=== Parallel Reduce 性能对比 ===\n";

    const size_t kSize = 10'000'000;
    std::vector<int> data(kSize);

    // 填充随机数据
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(1, 100);
    for (auto& v : data) v = dist(rng);

    const int kRounds = 5;
    auto op = std::plus<int>{};
    int init = 0;

    // 串行版本
    {
        long long total_time = 0;
        int result = 0;
        for (int r = 0; r < kRounds; ++r) {
            auto start = std::chrono::high_resolution_clock::now();
            result = std::accumulate(data.begin(), data.end(), init, op);
            auto elapsed =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start);
            total_time += elapsed.count();
        }
        std::cout << "  串行 accumulate: avg "
                  << total_time / kRounds << " us, result="
                  << result << "\n";
    }

    // 并行版本 (chunk-based)
    {
        long long total_time = 0;
        int result = 0;
        for (int r = 0; r < kRounds; ++r) {
            auto start = std::chrono::high_resolution_clock::now();
            result = parallel_reduce_basic(
                data.begin(), data.end(), init, op);
            auto elapsed =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start);
            total_time += elapsed.count();
        }
        std::cout << "  并行 reduce:      avg "
                  << total_time / kRounds << " us, result="
                  << result << "\n";
    }
}

// ===== 4. 结合律测试: 浮点数 =====
void demo_floating_point_associativity() {
    std::cout << "\n=== 浮点数归约陷阱 ===\n";

    std::vector<double> data = {1e20, -1e20, 1.0, 1.0, 1.0};

    // 串行累加 (左结合)
    double serial = 0;
    for (double v : data) serial += v;

    // 分块累加 (不同结合顺序)
    double parallel = (data[0] + data[1]) + (data[2] + data[3] + data[4]);

    std::cout << "  串行 (左结合):    " << serial << "\n";
    std::cout << "  并行 (分组结合):  " << parallel << "\n";
    std::cout << "  预期: 1e20 + (-1e20) + 1 + 1 + 1 = 3\n";
    std::cout << "  结论: 浮点数并行归约结果可能不等于串行结果，"
              << "这是 IEEE 754 的固有特性\n";
}

int main() {
    benchmark_reduce();
    demo_floating_point_associativity();

    std::cout << "\n并行归约的核心: 分治 + 合并。操作必须满足结合律。\n";
    return 0;
}
