/**
 * 04_execution_policies.cpp — C++17 不同执行策略的对比
 *
 * C++17 定义了四种执行策略:
 *  - std::execution::sequenced_policy      (seq)
 *  - std::execution::parallel_policy       (par)
 *  - std::execution::parallel_unsequenced_policy (par_unseq)
 *  - std::execution::unsequenced_policy    (unseq, C++20)
 *
 * 技术要点:
 *  - seq: 在调用线程上顺序执行, 保证元素顺序
 *  - par: 多线程并行执行, 线程间不交错 (对每个线程内的元素保证顺序)
 *  - par_unseq: 多线程+SIMD向量化, 允许向量化交错
 *  - 策略选择影响: 同步操作、锁、内存分配等有副作用操作的安全性
 *
 * 编译:
 *   GCC:  g++ -std=c++20 -O2 -pthread 04_execution_policies.cpp -ltbb -o execution_policies
 */

#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>
#include <random>
#include <chrono>
#include <iomanip>
#include <thread>
#include <mutex>
#include <atomic>
#include <string>

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
// 策略对比: 性能
// ============================================================================
void policy_performance_comparison() {
    std::cout << "=== 执行策略性能对比 ===\n\n";

    constexpr size_t N = 50'000'000;
    std::vector<int> original(N);
    std::iota(original.begin(), original.end(), 0);

    auto workload = [](int& x) {
        // 模拟一定计算量
        x = (x * 3 + 7) % 10007;
        for (int i = 0; i < 5; ++i) {
            x = (x * x + 1) % 1000003;
        }
    };

    std::cout << std::left
              << std::setw(22) << "策略"
              << std::setw(12) << "耗时"
              << std::setw(10) << "结果"
              << "\n";
    std::cout << std::string(44, '-') << "\n";

    // seq
    {
        auto data = original;
        double ms = measure([&]() {
#if HAS_EXECUTION
            std::for_each(std::execution::seq, data.begin(), data.end(), workload);
#else
            std::for_each(data.begin(), data.end(), workload);
#endif
        });
        std::cout << std::left << std::setw(22) << "seq (顺序)"
                  << std::setw(12) << (std::to_string(static_cast<int>(ms)) + " ms")
                  << std::setw(10) << std::accumulate(data.begin(), data.begin() + 5, 0LL)
                  << "\n";
    }

#if HAS_EXECUTION
    // par
    {
        auto data = original;
        double ms = measure([&]() {
            std::for_each(std::execution::par, data.begin(), data.end(), workload);
        });
        std::cout << std::left << std::setw(22) << "par (并行)"
                  << std::setw(12) << (std::to_string(static_cast<int>(ms)) + " ms")
                  << std::setw(10) << std::accumulate(data.begin(), data.begin() + 5, 0LL)
                  << "\n";
    }

    // par_unseq
    {
        auto data = original;
        double ms = measure([&]() {
            std::for_each(std::execution::par_unseq, data.begin(), data.end(), workload);
        });
        std::cout << std::left << std::setw(22) << "par_unseq (并行+向量化)"
                  << std::setw(12) << (std::to_string(static_cast<int>(ms)) + " ms")
                  << std::setw(10) << std::accumulate(data.begin(), data.begin() + 5, 0LL)
                  << "\n";
    }
#endif

    std::cout << "\n  硬件线程数: " << std::jthread::hardware_concurrency() << "\n";
}

// ============================================================================
// 策略安全性: 带锁操作的并行算法
// ============================================================================
void policy_safety_demo() {
    std::cout << "\n=== 执行策略安全注意事项 ===\n\n";

#if HAS_EXECUTION
    constexpr size_t N = 1'000'000;
    std::vector<int> data(N);
    std::iota(data.begin(), data.end(), 0);

    // ❌ 危险: par_unseq 下使用锁可能导致死锁
    // 因为 vectorization 可能导致同一线程重复获取已持有的锁
    std::cout << "  [警告] par_unseq 策略下不应使用互斥锁、内存分配等操作\n";
    std::cout << "  原因: 向量化执行可能在同一线程上交错执行迭代\n\n";

    // ✅ 安全: par 策略下可以使用锁 (每个线程内的迭代保证顺序)
    std::mutex mtx;
    std::vector<int> log;

    std::for_each(std::execution::par, data.begin(), data.begin() + 100,
        [&](int x) {
            // par 策略下安全: 同一线程内不会交错
            std::lock_guard<std::mutex> lock(mtx);
            log.push_back(x);
        });

    std::cout << "  par 策略 + mutex: 安全, 记录了 " << log.size() << " 条\n";

    // ✅ 安全: 使用原子操作
    std::atomic<long long> counter{0};
    std::for_each(std::execution::par, data.begin(), data.end(),
        [&](int x) {
            counter.fetch_add(x, std::memory_order_relaxed);
        });
    std::cout << "  par 策略 + atomic: 安全, counter=" << counter.load() << "\n";

#else
    std::cout << "  <execution> 不可用, 跳过安全演示\n";
#endif
}

// ============================================================================
// 策略选择决策树
// ============================================================================
void policy_decision_guide() {
    std::cout << "\n=== 执行策略选择指南 ===\n\n";
    std::cout << "  ┌─ 元素间有依赖? ──→ seq (必须顺序)\n";
    std::cout << "  ├─ 数据量 < 10000? ──→ seq (并行开销 > 收益)\n";
    std::cout << "  ├─ 操作包含锁/内存分配? ──→ par (不能用 par_unseq)\n";
    std::cout << "  ├─ 操作是纯粹计算且无副作用? ──→ par_unseq (最优)\n";
    std::cout << "  └─ 通用场景 ──→ par\n\n";

    std::cout << "  策略限制:\n";
    std::cout << "  - seq:     保证确定性执行顺序\n";
    std::cout << "  - par:     允许并行, 禁止向量化交错; 允许同步操作\n";
    std::cout << "  - par_unseq: 允许并行和向量化; 禁止同步操作 (锁, 分配)\n";
    std::cout << "  - unseq:   C++20, 单线程向量化; 禁止同步\n";
}

// ============================================================================
// 手动策略分发: 运行时选择策略
// ============================================================================
void runtime_policy_selection() {
    std::cout << "\n=== 运行时策略选择 ===\n\n";

    constexpr size_t N = 10'000'000;
    std::vector<int> data(N);
    std::iota(data.begin(), data.end(), 0);

    // 根据硬件线程数和数据量决定策略
    unsigned int hw = std::jthread::hardware_concurrency();
    size_t threshold = 10000;

    std::cout << "  数据量: " << N << ", 硬件线程: " << hw << "\n";

#if HAS_EXECUTION
    long long sum = 0;

    if (hw >= 4 && N > threshold) {
        std::cout << "  选择: par (并行执行)\n";
        sum = std::transform_reduce(
            std::execution::par,
            data.begin(), data.end(), 0LL,
            std::plus<>{},
            [](int x) -> long long { return x; }
        );
    } else {
        std::cout << "  选择: seq (顺序执行)\n";
        sum = std::transform_reduce(
            std::execution::seq,
            data.begin(), data.end(), 0LL,
            std::plus<>{},
            [](int x) -> long long { return x; }
        );
    }

    std::cout << "  结果: sum = " << sum;
    long long expected = static_cast<long long>(N - 1) * N / 2;
    std::cout << " (期望: " << expected << ", " << (sum == expected ? "通过" : "失败") << ")\n";
#endif
}

// ============================================================================
// main
// ============================================================================
int main() {
    policy_performance_comparison();
    policy_safety_demo();
    policy_decision_guide();
    runtime_policy_selection();
    return 0;
}
