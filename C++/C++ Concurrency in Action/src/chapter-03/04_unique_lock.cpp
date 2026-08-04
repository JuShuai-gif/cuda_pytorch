// 04_unique_lock.cpp
// 知识点: std::unique_lock 的灵活用法
// 演示: defer_lock, try_lock, adopt_lock, 提前解锁, 移动所有权
// 对应书中 3.2.6-3.2.8 节

#include <chrono>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// =============================================================================
// 共享资源: 模拟数据库连接池
// =============================================================================
class ConnectionPool {
public:
    explicit ConnectionPool(int max_connections)
        : m_available(max_connections) {}

    // 尝试获取连接 (try_lock)
    [[nodiscard]] bool try_acquire() {
        std::unique_lock lock(m_mutex, std::try_to_lock);
        if (!lock.owns_lock() || m_available <= 0) {
            return false;
        }
        --m_available;
        return true;
    }

    // 获取连接，带超时 (使用 sleep_for 模拟, 生产环境应用条件变量)
    [[nodiscard]] bool acquire_with_timeout(int timeout_ms) {
        auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms);

        while (std::chrono::steady_clock::now() < deadline) {
            {
                std::unique_lock lock(m_mutex, std::try_to_lock);
                if (lock.owns_lock() && m_available > 0) {
                    --m_available;
                    return true;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return false;
    }

    // 释放连接
    void release() {
        std::lock_guard<std::mutex> lock(m_mutex);
        ++m_available;
    }

    [[nodiscard]] int available() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_available;
    }

private:
    mutable std::mutex m_mutex;
    int                m_available;
};

// =============================================================================
// 演示 unique_lock 的各种用法
// =============================================================================

std::mutex g_mutex_a;
std::mutex g_mutex_b;
int       g_shared_data = 0;

int main() {
    std::cout << "=== std::unique_lock 灵活用法 ===\n\n";

    // --- 用法1: defer_lock (延迟锁定) ---
    std::cout << "--- 用法1: std::defer_lock (延迟锁定) ---\n";
    {
        std::unique_lock<std::mutex> lock_a(g_mutex_a, std::defer_lock);
        std::unique_lock<std::mutex> lock_b(g_mutex_b, std::defer_lock);

        std::cout << "  互斥量尚未锁定\n";
        std::cout << "  lock_a.owns_lock() = " << lock_a.owns_lock() << "\n";
        std::cout << "  lock_b.owns_lock() = " << lock_b.owns_lock() << "\n";

        // 同时锁定: 使用 std::lock
        std::lock(lock_a, lock_b);
        std::cout << "  同时锁定后:\n";
        std::cout << "  lock_a.owns_lock() = " << lock_a.owns_lock() << "\n";
        std::cout << "  lock_b.owns_lock() = " << lock_b.owns_lock() << "\n";

        g_shared_data = 42;
    }
    std::cout << "  作用域结束后自动解锁\n";

    // --- 用法2: try_to_lock (尝试锁定) ---
    std::cout << "\n--- 用法2: std::try_to_lock ---\n";
    {
        std::mutex m;
        m.lock();  // 主线程先锁定

        std::thread t([&m]() {
            std::unique_lock<std::mutex> lock(m, std::try_to_lock);
            if (lock.owns_lock()) {
                std::cout << "  子线程: 成功获取锁\n";
            } else {
                std::cout << "  子线程: 无法获取锁，执行其他任务\n";
                // 不阻塞，去做别的事情
            }
        });

        t.join();
        m.unlock();
    }

    // --- 用法3: adopt_lock (接管已锁定互斥量) ---
    std::cout << "\n--- 用法3: std::adopt_lock ---\n";
    {
        std::mutex m;
        m.lock();  // 手动锁定

        // unique_lock 接管所有权，析构时负责解锁
        std::unique_lock<std::mutex> lock(m, std::adopt_lock);
        std::cout << "  owns_lock = " << lock.owns_lock() << "\n";
    }  // lock 析构时自动 unlock
    std::cout << "  互斥量已自动解锁\n";

    // --- 用法4: 提前解锁 ---
    std::cout << "\n--- 用法4: 提前解锁 ---\n";
    {
        std::mutex                     m;
        std::unique_lock<std::mutex>   lock(m);

        std::cout << "  持锁中...\n";
        // 执行一些需要锁保护的操作
        g_shared_data = 100;

        // 提前解锁，缩短临界区
        lock.unlock();
        std::cout << "  已提前解锁，临界区结束\n";

        // 可以做一些不需要锁的事情
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // 如果需要，可以再次锁定
        lock.lock();
        std::cout << "  再次获得锁\n";
        g_shared_data = 200;
    }

    // --- 用法5: 移动所有权 ---
    std::cout << "\n--- 用法5: 移动所有权 ---\n";
    {
        std::mutex m;

        auto factory = [&m]() -> std::unique_lock<std::mutex> {
            std::unique_lock<std::mutex> lock(m);
            std::cout << "  工厂函数持有锁\n";
            g_shared_data = 999;
            return lock;  // 移动返回，所有权转移
        };

        auto consumer = [](std::unique_lock<std::mutex> lock) {
            std::cout << "  消费者接管了锁\n";
            std::cout << "  g_shared_data = " << g_shared_data << "\n";
            // lock 在此析构，解锁
        };

        // 从工厂获取锁，传给消费者
        consumer(factory());
    }

    // --- 用法6: 连接池 (实际场景) ---
    std::cout << "\n--- 用法6: 数据库连接池 (try_to_lock 实战) ---\n";
    {
        ConnectionPool pool(3);  // 最多3个连接

        std::vector<std::jthread> threads;
        threads.reserve(8);

        for (int i = 0; i < 8; ++i) {
            threads.emplace_back([&pool, i]() {
                if (pool.try_acquire()) {
                    std::cout << "  [线程 " << i << "] 获得连接，工作中...\n";
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(50));
                    pool.release();
                    std::cout << "  [线程 " << i << "] 释放连接\n";
                } else {
                    std::cout << "  [线程 " << i
                              << "] 无可用连接，稍后重试\n";
                    // 可以实现重试逻辑
                }
                // jthread 析构时自动 join
            });
        }
    }

    std::cout << "\n=== unique_lock vs lock_guard ===\n";
    std::cout << "特性                 lock_guard      unique_lock\n";
    std::cout << "----------------------------------------------\n";
    std::cout << "开销                   小(零成本)       稍大\n";
    std::cout << "自动锁定/解锁           ✓               ✓\n";
    std::cout << "延迟锁定(defer_lock)     ✗               ✓\n";
    std::cout << "尝试锁定(try_to_lock)     ✗               ✓\n";
    std::cout << "提前解锁                  ✗               ✓\n";
    std::cout << "移动所有权                ✗               ✓\n";
    std::cout << "与条件变量配合             ✗               ✓\n";
    std::cout << "\n建议: 默认使用 lock_guard，需要灵活性时用 unique_lock\n";

    return 0;
}
