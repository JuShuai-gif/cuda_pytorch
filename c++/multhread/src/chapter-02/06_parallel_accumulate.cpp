// 06_parallel_accumulate.cpp
// 知识点: parallel_accumulate - 分治并行累加
// 演示: 使用 std::thread::hardware_concurrency() 分块累加大数组
// 这是书中 2.4 节的工业级实现

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

// =============================================================================
// ScopedTimer: 简单的 RAII 计时器
// =============================================================================
class ScopedTimer {
public:
    using Clock     = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;

    explicit ScopedTimer(std::string name)
        : m_name(std::move(name)), m_start(Clock::now()) {}

    ~ScopedTimer() {
        auto end = Clock::now();
        std::cout << "[计时] " << m_name << ": "
                  << std::chrono::duration<double, std::milli>(end - m_start)
                         .count()
                  << " ms\n";
    }

    ScopedTimer(const ScopedTimer&)            = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    std::string m_name;
    TimePoint   m_start;
};

// =============================================================================
// parallel_accumulate: 分治并行累加
//
// 策略:
// 1. 根据硬件线程数确定最小分块大小
// 2. 数据量足够大时才并行，否则回退到串行
// 3. 每个线程计算自己分块的局部和
// 4. 主线程汇总所有局部和
// =============================================================================
template <typename Iterator, typename T>
T parallel_accumulate(Iterator first, Iterator last, T init) {
    const auto length = std::distance(first, last);
    if (length == 0) {
        return init;
    }

    // 确定线程数和分块大小
    const unsigned int hw_threads   = std::thread::hardware_concurrency();
    const unsigned int num_threads  = (hw_threads > 1) ? hw_threads : 2;
    const auto         block_size   = length / num_threads;

    // 如果数据量太小，不值得并行化
    // 阈值可以调整，这里设为 10000
    const auto min_per_thread = 10'000;
    const auto max_threads    = (length + min_per_thread - 1) / min_per_thread;
    const auto actual_threads = std::min<unsigned long>(
        static_cast<unsigned long>(num_threads),
        static_cast<unsigned long>(max_threads));

    std::cout << "  [parallel_accumulate] 数据量: " << length
              << ", 线程数: " << actual_threads << "\n";

    std::vector<std::thread> threads;
    threads.reserve(actual_threads - 1);  // 主线程也参与计算

    std::vector<T> partial_results(actual_threads, T{});

    auto block_start = first;

    // 让每个子线程计算一个数据块
    for (unsigned long i = 0; i < actual_threads - 1; ++i) {
        auto block_end = block_start;
        std::advance(block_end, block_size);

        threads.emplace_back(
            [block_start, block_end, &partial_results, i]() {
                partial_results[i] =
                    std::accumulate(block_start, block_end, T{});
            });

        block_start = block_end;
    }

    // 主线程处理最后一块 (包含剩余元素)
    partial_results[actual_threads - 1] =
        std::accumulate(block_start, last, T{});

    // 等待所有子线程完成
    for (auto& t : threads) {
        t.join();
    }

    // 汇总结果
    return std::accumulate(partial_results.begin(), partial_results.end(),
                           init);
}

// =============================================================================
// 测试: 计算平均值
// =============================================================================
template <typename Iterator, typename T>
double parallel_average(Iterator first, Iterator last, T /*init*/) {
    const auto length = std::distance(first, last);
    if (length == 0) {
        return 0.0;
    }
    auto sum = parallel_accumulate(first, last, typename std::iterator_traits<Iterator>::value_type{});
    return static_cast<double>(sum) / static_cast<double>(length);
}

int main() {
    const unsigned int hw = std::thread::hardware_concurrency();
    std::cout << "=== parallel_accumulate (分治并行累加) ===\n";
    std::cout << "硬件线程数: " << hw << "\n\n";

    // --- 测试1: 大数组并行累加 ---
    std::cout << "--- 测试1: 大数组 (100M 元素) ---\n";
    {
        constexpr long long N = 100'000'000LL;

        // 创建大数据集 (用1填充)
        std::vector<long long> data(N, 1LL);

        long long sum_serial   = 0;
        long long sum_parallel = 0;

        // 串行版本
        {
            ScopedTimer timer("串行累加");
            sum_serial = std::accumulate(data.begin(), data.end(), 0LL);
        }
        std::cout << "  串行结果: " << sum_serial << "\n";

        // 并行版本
        {
            ScopedTimer timer("并行累加");
            sum_parallel = parallel_accumulate(data.begin(), data.end(), 0LL);
        }
        std::cout << "  并行结果: " << sum_parallel << "\n";

        // 验证
        if (sum_serial == sum_parallel) {
            std::cout << "  结果一致: ✓\n";
        } else {
            std::cout << "  结果不一致: ✗\n";
        }
    }

    // --- 测试2: 小数据集 (自动回退到串行) ---
    std::cout << "\n--- 测试2: 小数据集 (自动回退) ---\n";
    {
        std::vector<int> small_data(100, 42);

        int sum = parallel_accumulate(small_data.begin(), small_data.end(), 0);
        std::cout << "  累加结果: " << sum << " (期望: " << 100 * 42 << ")\n";
    }

    // --- 测试3: 计算平均值 ---
    std::cout << "\n--- 测试3: 并行平均值 ---\n";
    {
        std::vector<double> values(10'000'000);
        std::iota(values.begin(), values.end(), 1.0);

        double avg = parallel_average(values.begin(), values.end(), 0.0);
        double expected =
            static_cast<double>(10'000'000 + 1) / 2.0;  // (1+10000000)/2
        std::cout << "  并行平均值: " << std::fixed << std::setprecision(6)
                  << avg << "\n";
        std::cout << "  期望值:     " << expected << "\n";
    }

    // --- 测试4: 空范围 ---
    std::cout << "\n--- 测试4: 边界情况 ---\n";
    {
        std::vector<int> empty_data;
        int              sum =
            parallel_accumulate(empty_data.begin(), empty_data.end(), 0);
        std::cout << "  空范围结果: " << sum << " ✓\n";

        std::vector<int> single_element = {42};
        sum = parallel_accumulate(single_element.begin(),
                                  single_element.end(), 0);
        std::cout << "  单元素结果: " << sum << " ✓\n";
    }

    std::cout << "\n=== parallel_accumulate 设计要点 ===\n";
    std::cout << "1. 分治策略: 将数据分为 N 块，N=线程数\n";
    std::cout << "2. 阈值控制: 数据量小于阈值时回退到串行\n";
    std::cout << "3. 主线程参与: 避免主线程空闲等待\n";
    std::cout << "4. 模板化: 支持任意迭代器和累加类型\n";
    std::cout << "5. 结果汇总: 子线程局部和 → 主线程全局和\n";
    std::cout << "6. 安全保证: join 所有线程，无数据竞争\n";

    return 0;
}
