// 05_hierarchical_mutex.cpp
// 知识点: hierarchical_mutex - 运行时死锁检测
// 演示: 实现书中 3.2.8 节的分层互斥量，强制锁定顺序
// 对应原书 Listing 3.8

#include <chrono>
#include <climits>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

// =============================================================================
// HierarchicalMutex: 分层互斥量
//
// 原理: 每个互斥量有一个层级值(hierarchy value)
// 规则: 线程只能锁定比当前已锁定层级更低的互斥量
// 即: 总是按照层级从高到低的顺序锁定
//
// 如果违反顺序 → 运行时错误
// 这是一种死锁避免策略 (Lock Ordering)
// =============================================================================
class HierarchicalMutex {
public:
    explicit HierarchicalMutex(unsigned long hierarchy_value)
        : m_hierarchy_value(hierarchy_value) {}

    void lock() {
        check_hierarchy_violation();
        m_internal_mutex.lock();
        update_hierarchy();
    }

    void unlock() {
        // 恢复之前的值
        if (m_this_thread_hierarchy_value != m_hierarchy_value) {
            // 锁层级不匹配 → 逻辑错误
            throw std::logic_error("mutex hierarchy violated on unlock");
        }
        m_this_thread_hierarchy_value = m_previous_hierarchy_value;
        m_internal_mutex.unlock();
    }

    [[nodiscard]] bool try_lock() {
        check_hierarchy_violation();
        if (!m_internal_mutex.try_lock()) {
            return false;
        }
        update_hierarchy();
        return true;
    }

    [[nodiscard]] unsigned long hierarchy_value() const noexcept {
        return m_hierarchy_value;
    }

private:
    void check_hierarchy_violation() const {
        if (m_this_thread_hierarchy_value <= m_hierarchy_value) {
            // 当前已锁的层级 <= 要锁的层级 → 违规!
            // 必须是 当前已锁的层级 > 要锁的层级 (从高到低)
            throw std::logic_error(
                "hierarchical_mutex: lock order violated! "
                "current=" +
                std::to_string(m_this_thread_hierarchy_value) +
                " <= target=" + std::to_string(m_hierarchy_value));
        }
    }

    void update_hierarchy() {
        m_previous_hierarchy_value   = m_this_thread_hierarchy_value;
        m_this_thread_hierarchy_value = m_hierarchy_value;
    }

    std::mutex        m_internal_mutex;
    unsigned long     m_hierarchy_value;

    // 线程局部存储: 每个线程独立维护当前锁层级
    static thread_local unsigned long m_this_thread_hierarchy_value;

    // 用于在解锁时恢复之前的值
    unsigned long m_previous_hierarchy_value = 0;
};

// 线程局部存储初始化: 初始值为 unsigned long 的最大值 (可以锁任何层级)
thread_local unsigned long
    HierarchicalMutex::m_this_thread_hierarchy_value = ULONG_MAX;

// =============================================================================
// 演示场景: 有三层锁
// =============================================================================

// 层级定义: 数值越大，层级越高
// 规则: 只能从高层锁到低层 (high → middle → low)
HierarchicalMutex g_high_mutex(10000);    // 高层锁
HierarchicalMutex g_middle_mutex(6000);   // 中层锁
HierarchicalMutex g_low_mutex(1000);      // 低层锁

// 模拟资源
int g_resource_a = 0;
int g_resource_b = 0;
int g_resource_c = 0;

// 正确的锁定顺序: high → middle → low
void fct_correct_order() {
    std::cout << "  [正确顺序] high → middle → low\n";
    std::lock_guard<HierarchicalMutex> lock_high(g_high_mutex);
    g_resource_a = 42;

    std::lock_guard<HierarchicalMutex> lock_middle(g_middle_mutex);
    g_resource_b = 84;

    std::lock_guard<HierarchicalMutex> lock_low(g_low_mutex);
    g_resource_c = 168;

    std::cout << "  操作完成: a=" << g_resource_a << " b=" << g_resource_b
              << " c=" << g_resource_c << "\n";
}

// 错误的锁定顺序: low → high (违反层级规则)
void fct_wrong_order() {
    std::cout << "  [错误顺序] low → high (违规!)\n";
    try {
        std::lock_guard<HierarchicalMutex> lock_low(g_low_mutex);
        std::cout << "  获取了底层锁...\n";

        // 尝试获取更高层级的锁 → 应该抛出异常!
        std::lock_guard<HierarchicalMutex> lock_high(g_high_mutex);
        std::cout << "  这行不应该被执行\n";
    } catch (const std::logic_error& e) {
        std::cout << "  捕获异常: " << e.what() << "\n";
    }
}

// 部分顺序: high → low (跳过 middle，从高到低，合法!)
void fct_high_to_low() {
    std::cout << "  [合法] high → low (跳过 middle)\n";
    try {
        std::lock_guard<HierarchicalMutex> lock_high(g_high_mutex);
        g_resource_a = 10;

        std::lock_guard<HierarchicalMutex> lock_low(g_low_mutex);
        g_resource_c = 20;

        std::cout << "  操作完成: a=" << g_resource_a << " c=" << g_resource_c
                  << "\n";
    } catch (const std::logic_error& e) {
        std::cout << "  不应该抛异常: " << e.what() << "\n";
    }
}

// 相同层级: 不允许 (需要当前层级 > 目标层级)
void fct_same_level() {
    std::cout << "  [违规] 同层级互斥量\n";
    try {
        HierarchicalMutex another_high(10000);
        std::lock_guard<HierarchicalMutex> lock1(g_high_mutex);
        std::lock_guard<HierarchicalMutex> lock2(another_high);
        std::cout << "  这行不应该被执行\n";
    } catch (const std::logic_error& e) {
        std::cout << "  捕获异常: " << e.what() << "\n";
    }
}

int main() {
    std::cout << "=== HierarchicalMutex: 层级锁死锁避免 ===\n\n";
    std::cout << "规则: 只能从高层锁到低层 (层级值从大到小)\n";
    std::cout << "层级: high(10000) > middle(6000) > low(1000)\n\n";

    // 每个测试在独立线程中执行 (线程局部存储隔离)
    {
        std::cout << "--- 测试1: 正确顺序 high→middle→low ---\n";
        std::thread t(fct_correct_order);
        t.join();
    }

    {
        std::cout << "\n--- 测试2: 错误顺序 low→high (应抛异常) ---\n";
        std::thread t(fct_wrong_order);
        t.join();
    }

    {
        std::cout << "\n--- 测试3: 合法顺序 high→low (跳过middle) ---\n";
        std::thread t(fct_high_to_low);
        t.join();
    }

    {
        std::cout << "\n--- 测试4: 相同层级 (违规) ---\n";
        std::thread t(fct_same_level);
        t.join();
    }

    // 并发测试: 两个线程用不同顺序操作
    std::cout << "\n--- 测试5: 两个线程正确操作 ---\n";
    {
        std::thread t1([]() {
            for (int i = 0; i < 3; ++i) {
                std::lock_guard<HierarchicalMutex> lock_h(g_high_mutex);
                std::lock_guard<HierarchicalMutex> lock_l(g_low_mutex);
                ++g_resource_a;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });

        std::thread t2([]() {
            for (int i = 0; i < 3; ++i) {
                std::lock_guard<HierarchicalMutex> lock_h(g_high_mutex);
                std::lock_guard<HierarchicalMutex> lock_m(g_middle_mutex);
                ++g_resource_b;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });

        t1.join();
        t2.join();

        std::cout << "  最终: a=" << g_resource_a << " b=" << g_resource_b
                  << "\n";
    }

    std::cout << "\n=== HierarchicalMutex 设计要点 ===\n";
    std::cout << "1. 每个互斥量分配一个层级值\n";
    std::cout << "2. 线程只能锁定比当前更低的层级 (值从大到小)\n";
    std::cout << "3. 使用 thread_local 存储每个线程的当前层级\n";
    std::cout << "4. 违规锁定 → std::logic_error 异常\n";
    std::cout << "5. 这是一种运行时死锁避免策略\n";
    std::cout << "6. 生产环境可用 C++20 std::latch/barrier 替代部分场景\n";

    return 0;
}
