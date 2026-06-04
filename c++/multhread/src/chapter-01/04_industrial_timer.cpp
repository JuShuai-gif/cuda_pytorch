// 04_industrial_timer.cpp
// 知识点: RAII 计时器，测量并发任务执行时间
// 演示: 工业级 ScopedTimer 封装，用于性能基准测试

#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

// =============================================================================
// ScopedTimer: RAII 计时器，构造时开始计时，析构时输出耗时
// 适用于任何作用域的性能测量
// =============================================================================
class ScopedTimer {
public:
    using Clock     = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;

    // 创建计时器并命名
    explicit ScopedTimer(std::string name, bool auto_print = true)
        : m_name(std::move(name))
        , m_auto_print(auto_print)
        , m_start(Clock::now()) {}

    // 禁止拷贝
    ScopedTimer(const ScopedTimer&)            = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

    // 允许移动 (用于从工厂函数返回)
    ScopedTimer(ScopedTimer&&) noexcept            = default;
    ScopedTimer& operator=(ScopedTimer&&) noexcept = default;

    // 析构时自动输出耗时
    ~ScopedTimer() {
        if (m_auto_print && !m_stopped) {
            stop();
        }
    }

    // 手动停止并返回耗时(毫秒)
    double stop() {
        if (m_stopped) {
            return m_elapsed_ms;
        }
        m_stopped    = true;
        auto end     = Clock::now();
        m_elapsed_ms = std::chrono::duration<double, std::milli>(end - m_start)
                           .count();

        if (m_auto_print) {
            std::cout << "[计时器] " << m_name << " 耗时: " << std::fixed
                      << std::setprecision(3) << m_elapsed_ms << " ms\n";
        }
        return m_elapsed_ms;
    }

    // 获取当前已流逝的时间(不停止计时)
    [[nodiscard]] double elapsed_ms() const {
        auto now = Clock::now();
        return std::chrono::duration<double, std::milli>(now - m_start)
            .count();
    }

    // 重置计时器
    void reset() {
        m_start   = Clock::now();
        m_stopped = false;
    }

    [[nodiscard]] const std::string& name() const { return m_name; }

private:
    std::string m_name;
    bool        m_auto_print;
    TimePoint   m_start;
    bool        m_stopped     = false;
    double      m_elapsed_ms  = 0.0;
};

// =============================================================================
// 将被测量的工作函数
// =============================================================================

// CPU密集型: 计算斐波那契数列
long long fct_fibonacci(int n) {
    if (n <= 1) return n;
    long long a = 0, b = 1;
    for (int i = 2; i <= n; ++i) {
        long long c = a + b;
        a           = b;
        b           = c;
    }
    return b;
}

// 模拟IO密集型: 睡眠
void fct_simulate_io(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

int main() {
    std::cout << "=== 工业级 ScopedTimer 演示 ===\n\n";

    // --- 用法1: 自动计时 (RAII) ---
    std::cout << "--- 用法1: 自动计时作用域 ---\n";
    {
        ScopedTimer timer("单线程斐波那契计算");

        long long result = fct_fibonacci(50'000'000);
        // 计时器在此作用域结束时自动输出耗时
        std::cout << "  计算结果: fib(50M) 共有 " << std::to_string(result).size()
                  << " 位数字\n";
    }  // <- timer 析构，自动打印耗时

    std::cout << "\n--- 用法2: 手动停止 ---\n";
    {
        ScopedTimer timer("手动控制", false);  // 不自动打印
        fct_fibonacci(30'000'000);
        double elapsed = timer.stop();
        std::cout << "  手动获取耗时: " << elapsed << " ms\n";
    }

    // --- 用法3: 并行任务计时 ---
    std::cout << "\n--- 用法3: 并行任务计时 ---\n";
    {
        const unsigned int hw   = std::thread::hardware_concurrency();
        const unsigned int n    = (hw > 0) ? hw : 4;
        const int          work = 10'000'000;

        // 单线程基准
        {
            ScopedTimer timer("单线程基准 (" + std::to_string(n) + " 个任务)");
            for (unsigned int i = 0; i < n; ++i) {
                fct_fibonacci(work);
            }
        }

        // 多线程并行
        {
            ScopedTimer timer("多线程并行 (" + std::to_string(n) + " 个线程)");
            std::vector<std::jthread> threads;  // C++20 jthread
            threads.reserve(n);
            for (unsigned int i = 0; i < n; ++i) {
                threads.emplace_back([work]() { fct_fibonacci(work); });
            }
            // jthread 析构时自动 join
        }  // timer 析构时所有 jthread 已 join 完毕

        std::cout << "  (jthread 在析构时自动 join，无需手动管理)\n";
    }

    // --- 用法4: 嵌套计时 ---
    std::cout << "\n--- 用法4: 嵌套计时 ---\n";
    {
        ScopedTimer outer("外层操作");
        {
            ScopedTimer inner("内层操作-1");
            fct_simulate_io(50);
        }
        {
            ScopedTimer inner("内层操作-2");
            fct_simulate_io(30);
        }
    }

    // --- 用法5: 重置计时器 ---
    std::cout << "\n--- 用法5: 重置计时器 ---\n";
    {
        ScopedTimer timer("可重置计时器");
        fct_simulate_io(100);
        std::cout << "  第1次: " << timer.elapsed_ms() << " ms\n";

        timer.reset();
        fct_simulate_io(50);
        timer.stop();
    }

    std::cout << "\n=== ScopedTimer 设计要点 ===\n";
    std::cout << "1. RAII: 构造开始计时，析构输出耗时\n";
    std::cout << "2. 移动语义: 支持从工厂函数返回\n";
    std::cout << "3. 手动控制: stop()/reset()/elapsed_ms()\n";
    std::cout << "4. 禁止拷贝: 每个计时器实例独一无二\n";
    std::cout << "5. 高精度: 使用 high_resolution_clock\n";

    return 0;
}
