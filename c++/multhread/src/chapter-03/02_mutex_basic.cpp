// 02_mutex_basic.cpp
// 知识点: std::mutex + std::lock_guard 保护共享数据
// 演示: 使用互斥量保护临界区，lock_guard 的 RAII 用法
// 对应书中 3.2 节

#include <chrono>
#include <iostream>
#include <list>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// =============================================================================
// 线程安全计数器: 封装 mutex + 数据
// =============================================================================
class ThreadSafeCounter {
public:
    ThreadSafeCounter() = default;

    // 禁止拷贝 (互斥量不可拷贝)
    ThreadSafeCounter(const ThreadSafeCounter&)            = delete;
    ThreadSafeCounter& operator=(const ThreadSafeCounter&) = delete;

    void increment() {
        // lock_guard: RAII，构造时 lock，析构时 unlock
        // 即使抛异常也能正确 unlock
        std::lock_guard<std::mutex> lock(m_mutex);
        ++m_value;
    }

    void decrement() {
        std::lock_guard<std::mutex> lock(m_mutex);
        --m_value;
    }

    [[nodiscard]] long long value() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_value;
    }

    // 原子化的"获取并重置"操作
    [[nodiscard]] long long get_and_reset() {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto                        old = m_value;
        m_value                        = 0;
        return old;
    }

private:
    mutable std::mutex m_mutex;  // mutable 允许在 const 方法中锁定
    long long          m_value = 0;
};

// =============================================================================
// 线程安全列表: 简单封装
// =============================================================================
class ThreadSafeList {
public:
    void push_back(int value) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_list.push_back(value);
    }

    [[nodiscard]] bool pop_front(int& value) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_list.empty()) {
            return false;
        }
        value = m_list.front();
        m_list.pop_front();
        return true;
    }

    [[nodiscard]] size_t size() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_list.size();
    }

private:
    mutable std::mutex m_mutex;
    std::list<int>     m_list;
};

// =============================================================================
// 演示不当的接口设计: 返回指针/引用破坏封装
// =============================================================================
class BadCounter {
public:
    // 危险! 返回指向受保护数据的指针/引用
    // 调用者可以绕过互斥量直接访问
    // const long long& value() const { return m_value; }   // 危险!
    // long long* ptr() { return &m_value; }                // 危险!

    void increment() {
        std::lock_guard<std::mutex> lock(m_mutex);
        ++m_value;
    }

    // 正确做法: 通过受控接口访问
    [[nodiscard]] long long value() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_value;
    }

private:
    mutable std::mutex m_mutex;
    long long          m_value = 0;
};

int main() {
    std::cout << "=== std::mutex + std::lock_guard ===\n\n";

    // --- 测试1: 线程安全计数器 ---
    std::cout << "--- 测试1: 线程安全计数器 ---\n";
    {
        ThreadSafeCounter     counter;
        const int             num_threads = 8;
        const int             ops         = 100'000;
        std::vector<std::jthread> threads;
        threads.reserve(num_threads);

        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&counter, ops]() {
                for (int j = 0; j < ops; ++j) {
                    counter.increment();
                }
            });
        }
        // jthread 析构时自动 join

        long long expected = static_cast<long long>(num_threads) * ops;
        std::cout << "  期望值: " << expected << "\n";
        std::cout << "  实际值: " << counter.value() << "\n";
        std::cout << "  结果: " << (expected == counter.value() ? "✓" : "✗")
                  << "\n";
    }

    // --- 测试2: 线程安全列表 (生产者-消费者) ---
    std::cout << "\n--- 测试2: 线程安全列表 ---\n";
    {
        ThreadSafeList list;
        const int      num_producers = 4;
        const int      items_per_prod = 100;

        // 生产者线程
        std::vector<std::jthread> producers;
        producers.reserve(num_producers);
        for (int p = 0; p < num_producers; ++p) {
            producers.emplace_back([&list, p, items_per_prod]() {
                for (int i = 0; i < items_per_prod; ++i) {
                    list.push_back(p * 1000 + i);
                }
            });
        }

        // 等待所有生产者完成
        for (auto& t : producers) {
            t.join();
        }

        std::cout << "  列表大小: " << list.size()
                  << " (期望: " << num_producers * items_per_prod << ")\n";

        // 消费者: 逐个取出
        int value    = 0;
        int consumed = 0;
        while (list.pop_front(value)) {
            ++consumed;
        }
        std::cout << "  消费数量: " << consumed << "\n";
    }

    // --- 测试3: 不当接口的危害 (代码演示) ---
    std::cout << "\n--- 测试3: 接口设计原则 ---\n";
    {
        BadCounter bc;

        // 正确做法
        bc.increment();
        std::cout << "  安全访问: " << bc.value() << "\n";

        // 如果 BadCounter 返回了引用:
        // auto& ref = bc.value();  // 获取引用，但锁已释放
        // 之后 ref 可能被其他线程修改 → 数据竞争!
        std::cout << "  原则: 绝不返回受保护数据的指针/引用\n";
    }

    std::cout << "\n=== lock_guard 使用要点 ===\n";
    std::cout << "1. RAII: 构造时 lock()，析构时 unlock()\n";
    std::cout << "2. 异常安全: 即使抛出异常也会解锁\n";
    std::cout << "3. 不可拷贝: 防止意外复制锁\n";
    std::cout << "4. 粒度控制: 临界区应尽可能小\n";
    std::cout << "5. 接口设计: 不返回受保护数据的指针/引用\n";

    return 0;
}
