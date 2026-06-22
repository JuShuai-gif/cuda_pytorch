/**
 * 03_parallel_transform_reduce.cpp — 使用 C++17 并行 MapReduce
 *
 * std::transform_reduce 是 C++17 引入的并行归约操作。
 * Map 阶段: 对每个元素应用变换函数
 * Reduce 阶段: 用二元操作归约所有结果
 *
 * 使用场景:
 *  - 计算向量点积 (dot product)
 *  - 计算平方和
 *  - MapReduce 大数据处理
 *
 * 编译:
 *   GCC:  g++ -std=c++20 -O2 -pthread 03_parallel_transform_reduce.cpp -ltbb -o transform_reduce
 */

#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <thread>
#include <cmath>
#include <execution>

#include <execution>
#ifdef HAS_TBB
    #define HAS_EXECUTION 1
#else
    #define HAS_EXECUTION 0
#endif

// ============================================================================
// 辅助
// ============================================================================
template <typename Func>
double measure(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ============================================================================
// 场景1: 向量点积 (Dot Product)
// ============================================================================
void dot_product_demo() {
    std::cout << "=== 并行向量点积 ===\n\n";

    constexpr size_t N = 100'000'000;
    std::vector<double> a(N), b(N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < N; ++i) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }

    // 顺序版本
    double seq_result = 0;
    double seq_time = measure([&]() {
#if HAS_EXECUTION
        seq_result = std::transform_reduce(
            a.begin(), a.end(), b.begin(), 0.0);
#else
        seq_result = std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
#endif
    });
    std::cout << "  顺序: " << static_cast<int>(seq_time) << " ms, 结果="
              << std::fixed << std::setprecision(6) << seq_result << "\n";

    // 并行版本
#if HAS_EXECUTION
    double par_result = 0;
    double par_time = measure([&]() {
        par_result = std::transform_reduce(
            std::execution::par,
            a.begin(), a.end(), b.begin(), 0.0);
    });
    std::cout << "  并行: " << static_cast<int>(par_time) << " ms, 结果="
              << std::fixed << std::setprecision(6) << par_result << "\n";
    std::cout << "  加速比: " << std::fixed << std::setprecision(2)
              << (seq_time / par_time) << "x\n";
#else
    std::cout << "  <execution> 不可用\n";
#endif

    std::cout << "\n";
}

// ============================================================================
// 场景2: 平方和计算 (自定义二元操作)
// ============================================================================
void sum_of_squares_demo() {
    std::cout << "=== 并行平方和 (自定义 reduce) ===\n\n";

    constexpr size_t N = 50'000'000;
    std::vector<double> data(N);
    std::iota(data.begin(), data.end(), 0.0);

    // 顺序版本
    double seq_result = 0;
    double seq_time = measure([&]() {
        seq_result = std::transform_reduce(
            data.begin(), data.end(),
            0.0,
            std::plus<>{},           // reduce: 求和
            [](double x) { return x * x; } // transform: 平方
        );
    });
    std::cout << "  顺序: " << static_cast<int>(seq_time) << " ms, sum(x^2)="
              << std::fixed << std::setprecision(0) << seq_result << "\n";

#if HAS_EXECUTION
    // 并行版本
    double par_result = 0;
    double par_time = measure([&]() {
        par_result = std::transform_reduce(
            std::execution::par,
            data.begin(), data.end(),
            0.0,
            std::plus<>{},
            [](double x) { return x * x; }
        );
    });
    std::cout << "  并行: " << static_cast<int>(par_time) << " ms, sum(x^2)="
              << std::fixed << std::setprecision(0) << par_result << "\n";
    std::cout << "  加速比: " << std::fixed << std::setprecision(2)
              << (seq_time / par_time) << "x\n";

    // 验证
    double expected = static_cast<double>(N - 1) * N * (2 * N - 1) / 6.0;
    std::cout << "  期望值: " << std::fixed << std::setprecision(0) << expected
              << "  正确: " << (std::abs(par_result - expected) < 1e-3 ? "是" : "否")
              << "\n";
#endif

    std::cout << "\n";
}

// ============================================================================
// 场景3: MapReduce 模式 — 单词计数模拟
// ============================================================================
void mapreduce_demo() {
    std::cout << "=== MapReduce 模式: 数据统计 ===\n\n";

    // 模拟 10M 条温度记录
    constexpr size_t N = 10'000'000;
    std::vector<double> temperatures(N);

    std::mt19937 rng(123);
    std::normal_distribution<double> dist(25.0, 10.0); // 均值25, 标准差10
    for (auto& t : temperatures) t = dist(rng);

    // 统计 > 30 度的天数 (Map: 条件判断, Reduce: 计数)
#if HAS_EXECUTION
    auto start = std::chrono::high_resolution_clock::now();

    long long hot_days = std::transform_reduce(
        std::execution::par,
        temperatures.begin(), temperatures.end(),
        0LL,
        std::plus<>{},
        [](double t) -> long long { return t > 30.0 ? 1 : 0; }
    );

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // 均值统计 (Map: 恒等, Reduce: 求和后除N)
    double sum = std::transform_reduce(
        std::execution::par,
        temperatures.begin(), temperatures.end(),
        0.0,
        std::plus<>{},
        [](double t) { return t; }
    );
    double mean = sum / N;

    std::cout << "  记录数: " << N / 1000000.0 << "M\n";
    std::cout << "  平均温度: " << std::fixed << std::setprecision(2) << mean << " C\n";
    std::cout << "  高温天数 (>30C): " << hot_days
              << " (" << (100.0 * hot_days / N) << "%)\n";
    std::cout << "  耗时: " << ms << " ms\n";
#else
    std::cout << "  <execution> 不可用\n";
#endif

    std::cout << "\n";
}

// ============================================================================
// 场景4: 最小值/最大值归约
// ============================================================================
void minmax_demo() {
    std::cout << "=== 并行归约: 最小/最大值 ===\n\n";

    constexpr size_t N = 50'000'000;
    std::vector<int> data(N);
    std::mt19937 rng(99);
    std::uniform_int_distribution<int> dist(1, 100000000);
    for (auto& v : data) v = dist(rng);

#if HAS_EXECUTION
    // 使用 reduce (C++17)
    auto start = std::chrono::high_resolution_clock::now();

    int min_val = std::reduce(
        std::execution::par,
        data.begin(), data.end(),
        std::numeric_limits<int>::max(),
        [](int a, int b) { return std::min(a, b); }
    );

    int max_val = std::reduce(
        std::execution::par,
        data.begin(), data.end(),
        std::numeric_limits<int>::min(),
        [](int a, int b) { return std::max(a, b); }
    );

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "  数据量: " << N / 1000000.0 << "M\n";
    std::cout << "  最小值: " << min_val << "\n";
    std::cout << "  最大值: " << max_val << "\n";
    std::cout << "  耗时: " << ms << " ms\n";

    // 验证
    auto [real_min, real_max] = std::minmax_element(data.begin(), data.end());
    std::cout << "  正确性: "
              << ((min_val == *real_min && max_val == *real_max) ? "通过" : "失败")
              << "\n";
#endif
}

// ============================================================================
// main
// ============================================================================
int main() {
    dot_product_demo();
    sum_of_squares_demo();
    mapreduce_demo();
    minmax_demo();
    return 0;
}
