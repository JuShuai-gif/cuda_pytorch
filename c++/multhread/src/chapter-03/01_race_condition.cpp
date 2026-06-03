// 01_race_condition.cpp
// 知识点: 数据竞争 (Data Race) 的危害
// 演示: 多个线程同时递增一个非原子变量，展示数据竞争导致的错误结果
// 对应书中 3.1 节

#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

// =============================================================================
// ScopedTimer: RAII 计时器
// =============================================================================
class ScopedTimer {
public:
    explicit ScopedTimer(std::string name)
        : m_name(std::move(name))
        , m_start(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "[计时] " << m_name << ": "
                  << std::chrono::duration<double, std::milli>(end - m_start)
                         .count()
                  << " ms\n";
    }

    ScopedTimer(const ScopedTimer&)            = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    std::string m_name;
    std::chrono::high_resolution_clock::time_point m_start;
};

// =============================================================================
// 有数据竞争的递增函数 (非原子，无线程同步)
// =============================================================================
void fct_unsafe_increment(int& counter, int iterations) {
    for (int i = 0; i < iterations; ++i) {
        // 数据竞争! 多线程同时读写 counter，没有同步
        // counter++ 不是原子操作:
        //   1. 读取 counter 到寄存器
        //   2. 寄存器值 +1
        //   3. 写回 counter
        // 两个线程可能交错执行，导致丢失更新
        ++counter;
    }
}

int main() {
    std::cout << "=== 数据竞争演示 ===\n\n";

    const int num_threads  = 8;
    const int increments   = 1'000'000;
    const int expected     = num_threads * increments;

    // --- 有数据竞争的版本 ---
    std::cout << "--- 有数据竞争的递增 ---\n";
    {
        int                      counter = 0;
        std::vector<std::thread> threads;
        threads.reserve(num_threads);

        {
            ScopedTimer timer("有竞争版本");
            for (int i = 0; i < num_threads; ++i) {
                threads.emplace_back(fct_unsafe_increment, std::ref(counter),
                                     increments);
            }
            for (auto& t : threads) {
                t.join();
            }
        }

        int lost = expected - counter;
        std::cout << "  期望值: " << expected << "\n";
        std::cout << "  实际值: " << counter << "\n";
        std::cout << "  丢失更新数: " << lost << " ("
                  << (double(lost) / expected * 100.0) << "%)\n";
    }

    // --- 使用 std::atomic 修复 ---
    std::cout << "\n--- 使用 std::atomic 修复 ---\n";
    {
        std::atomic<int>         counter{0};
        std::vector<std::thread> threads;
        threads.reserve(num_threads);

        {
            ScopedTimer timer("原子变量版本");
            for (int i = 0; i < num_threads; ++i) {
                threads.emplace_back(
                    [&counter, increments]() {
                        for (int j = 0; j < increments; ++j) {
                            // atomic 的 operator++ 是原子的
                            ++counter;
                        }
                    });
            }
            for (auto& t : threads) {
                t.join();
            }
        }

        std::cout << "  期望值: " << expected << "\n";
        std::cout << "  实际值: " << counter.load() << "\n";
        std::cout << "  结果一致: ✓\n";
    }

    // --- 情况说明 ---
    std::cout << "\n=== 数据竞争 (Data Race) 说明 ===\n";
    std::cout << "1. 两个及以上线程同时访问同一内存位置\n";
    std::cout << "2. 至少有一个是写操作\n";
    std::cout << "3. 没有使用同步机制(互斥量/原子操作)\n";
    std::cout << "4. 后果: 未定义行为(UB)，结果不可预测\n";
    std::cout << "\n=== 解决方案 ===\n";
    std::cout << "1. std::mutex + std::lock_guard (互斥量)\n";
    std::cout << "2. std::atomic (原子操作，简单计数器场景)\n";
    std::cout << "3. 线程安全的数据结构 (避免直接共享)\n";

    return 0;
}
