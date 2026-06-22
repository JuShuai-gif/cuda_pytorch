/**
 * 01_deadlock_demo.cpp — 死锁场景演示与修复
 *
 * 死锁四个必要条件 (Coffman 条件):
 *  1. 互斥: 资源一次只能被一个线程持有
 *  2. 持有并等待: 线程持有资源同时等待其他资源
 *  3. 不可抢占: 资源不能被强制释放
 *  4. 循环等待: 存在线程-资源的环形等待链
 *
 * 解决方案:
 *  1. std::lock() — 同时锁定多个互斥锁
 *  2. std::scoped_lock (C++17) — RAII 多锁管理
 *  3. 固定锁顺序 (Lock Ordering) — 总是按相同顺序获取锁
 *  4. std::unique_lock + std::defer_lock — 延迟加锁
 *  5. 层级锁 (Hierarchical Mutex) — 防止锁序反转
 *
 * 编译: g++ -std=c++20 -O2 -pthread 01_deadlock_demo.cpp -o deadlock_demo
 */

#include <iostream>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <chrono>
#include <vector>
#include <algorithm>
#include <condition_variable>

// ============================================================================
// 场景1: 经典死锁 — 两个锁, 相反的顺序
// ============================================================================
class BankAccount {
private:
    std::string name_;
    double balance_;
    mutable std::mutex mutex_;

public:
    BankAccount(std::string name, double initial)
        : name_(std::move(name)), balance_(initial) {}

    std::string name() const { return name_; }
    double balance() const { return balance_; }

    // ❌ 危险版本: 两个账户互相转账, 锁序不一致会死锁
    static void unsafe_transfer(BankAccount& from, BankAccount& to, double amount) {
        std::lock_guard<std::mutex> lock_from(from.mutex_);
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // 放大死锁窗口
        std::lock_guard<std::mutex> lock_to(to.mutex_);

        from.balance_ -= amount;
        to.balance_ += amount;
    }

    // ✅ 安全版本1: 使用 std::lock 同时锁定
    static void safe_transfer_lock(BankAccount& from, BankAccount& to, double amount) {
        std::lock(from.mutex_, to.mutex_);
        std::lock_guard<std::mutex> lock_from(from.mutex_, std::adopt_lock);
        std::lock_guard<std::mutex> lock_to(to.mutex_, std::adopt_lock);

        from.balance_ -= amount;
        to.balance_ += amount;
    }

    // ✅ 安全版本2: 使用 std::scoped_lock (C++17)
    static void safe_transfer_scoped(BankAccount& from, BankAccount& to, double amount) {
        std::scoped_lock lock(from.mutex_, to.mutex_);

        from.balance_ -= amount;
        to.balance_ += amount;
    }

    // ✅ 安全版本3: 固定锁顺序 (基于地址比较)
    static void safe_transfer_ordered(BankAccount& from, BankAccount& to, double amount) {
        // 总是先锁地址较小的那个
        BankAccount& first = (&from < &to) ? from : to;
        BankAccount& second = (&from < &to) ? to : from;

        std::lock_guard<std::mutex> lock_first(first.mutex_);
        std::lock_guard<std::mutex> lock_second(second.mutex_);

        from.balance_ -= amount;
        to.balance_ += amount;
    }
};

void deadlock_demo() {
    std::cout << "=== 场景1: 转账死锁演示 ===\n\n";

    BankAccount alice("Alice", 1000.0);
    BankAccount bob("Bob", 1000.0);

    std::cout << "  初始: Alice=" << alice.balance() << ", Bob=" << bob.balance() << "\n";

    // 安全转账 (不会死锁)
    std::cout << "  执行 10 次安全并发转账 (scoped_lock)...\n";

    std::vector<std::jthread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&alice, &bob]() {
            BankAccount::safe_transfer_scoped(alice, bob, 10.0);
        });
        threads.emplace_back([&alice, &bob]() {
            BankAccount::safe_transfer_scoped(bob, alice, 10.0);
        });
    }

    for (auto& t : threads) t.join();

    std::cout << "  结果: Alice=" << alice.balance() << ", Bob=" << bob.balance()
              << " (总和=" << (alice.balance() + bob.balance()) << ")\n\n";

    std::cout << "  *** 注意: unsafe_transfer 版本有死锁风险, 未在此运行 ***\n";
    std::cout << "  *** 如需观察死锁, 可注释掉 run_deadlock_unsafe() 调用 ***\n\n";
}

// -----------------------------------------------------------------------
// 如果确实想看死锁, 取消注释下面这个函数:
// -----------------------------------------------------------------------
// void run_deadlock_unsafe() {
//     BankAccount alice("Alice", 1000.0);
//     BankAccount bob("Bob", 1000.0);
//
//     std::cout << "  ⚠️ 运行不安全版本 (预期死锁) — 将在 2 秒后超时...\n";
//
//     std::jthread t1([&]() { BankAccount::unsafe_transfer(alice, bob, 10.0); });
//     std::jthread t2([&]() { BankAccount::unsafe_transfer(bob, alice, 10.0); });
//
//     auto start = std::chrono::steady_clock::now();
//     while (true) {
//         if (std::chrono::steady_clock::now() - start > std::chrono::seconds(2)) {
//             std::cout << "  ⚠️ 2 秒后仍在运行 — 确认死锁! 强制 detach\n";
//             t1.detach();
//             t2.detach();
//             break;
//         }
//     }
// }

// ============================================================================
// 场景2: 层级锁 (Hierarchical Mutex) — 编译期防止锁序反转
// ============================================================================
class HierarchicalMutex {
private:
    std::mutex internal_mutex_;
    const unsigned long hierarchy_value_;
    unsigned long previous_hierarchy_value_;

    // 线程局部: 当前线程持有的锁层级
    static thread_local unsigned long this_thread_hierarchy_value;

    void check_for_hierarchy_violation() {
        if (this_thread_hierarchy_value <= hierarchy_value_) {
            throw std::logic_error(
                "锁层级违规: 当前层级=" + std::to_string(this_thread_hierarchy_value) +
                ", 请求层级=" + std::to_string(hierarchy_value_));
        }
    }

    void update_hierarchy_value() {
        previous_hierarchy_value_ = this_thread_hierarchy_value;
        this_thread_hierarchy_value = hierarchy_value_;
    }

public:
    explicit HierarchicalMutex(unsigned long value)
        : hierarchy_value_(value), previous_hierarchy_value_(0) {}

    void lock() {
        check_for_hierarchy_violation();
        internal_mutex_.lock();
        update_hierarchy_value();
    }

    void unlock() {
        if (this_thread_hierarchy_value != hierarchy_value_) {
            throw std::logic_error("锁层级异常: unlock 顺序不匹配");
        }
        this_thread_hierarchy_value = previous_hierarchy_value_;
        internal_mutex_.unlock();
    }

    bool try_lock() {
        check_for_hierarchy_violation();
        if (!internal_mutex_.try_lock()) return false;
        update_hierarchy_value();
        return true;
    }
};

thread_local unsigned long HierarchicalMutex::this_thread_hierarchy_value =
    std::numeric_limits<unsigned long>::max();

void hierarchical_mutex_demo() {
    std::cout << "=== 场景2: 层级锁 (Hierarchical Mutex) ===\n\n";

    // 定义锁层级: 高层级锁必须先于低层级锁获取
    HierarchicalMutex high_level_mutex(10000);  // 高层级
    HierarchicalMutex mid_level_mutex(5000);    // 中层级
    HierarchicalMutex low_level_mutex(1000);    // 低层级

    // ✅ 正确: 从高到低获取锁
    {
        std::lock_guard<HierarchicalMutex> lock_high(high_level_mutex);
        std::lock_guard<HierarchicalMutex> lock_mid(mid_level_mutex);
        std::lock_guard<HierarchicalMutex> lock_low(low_level_mutex);
        std::cout << "  ✅ 高层 -> 中层 -> 低层: 正确, 无违规\n";
    }

    // ❌ 错误: 从低到高获取锁
    try {
        std::lock_guard<HierarchicalMutex> lock_low(low_level_mutex);
        std::lock_guard<HierarchicalMutex> lock_high(high_level_mutex); // 违规!
        std::cout << "  不应该到达这里\n";
    } catch (const std::logic_error& e) {
        std::cout << "  ❌ 低层 -> 高层: 捕获异常 — " << e.what() << "\n";
    }

    std::cout << "\n";
}

// ============================================================================
// 场景3: 带超时的锁 — try_lock_for 避免死锁
// ============================================================================
void try_lock_demo() {
    std::cout << "=== 场景3: try_lock_for 避免死锁 ===\n\n";

    std::timed_mutex mtx1, mtx2;

    auto transfer_with_timeout = [&](std::timed_mutex& first, std::timed_mutex& second,
                                      const std::string& name) {
        while (true) {
            std::unique_lock<std::timed_mutex> lock_first(first, std::defer_lock);

            if (!lock_first.try_lock_for(std::chrono::milliseconds(100))) {
                std::cout << "  " << name << ": 无法获取第一把锁, 重试...\n";
                continue;
            }

            std::unique_lock<std::timed_mutex> lock_second(second, std::defer_lock);
            if (!lock_second.try_lock_for(std::chrono::milliseconds(100))) {
                std::cout << "  " << name << ": 无法获取第二把锁, 释放第一把并重试...\n";
                lock_first.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            // 两把锁都已获取
            std::cout << "  " << name << ": 成功获取两把锁, 执行转账\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            return;
        }
    };

    std::jthread t1([&]() {
        transfer_with_timeout(mtx1, mtx2, "线程1");
    });

    std::jthread t2([&]() {
        transfer_with_timeout(mtx2, mtx1, "线程2");
    });

    t1.join();
    t2.join();

    std::cout << "\n  两线程均已完成, 无死锁!\n\n";
}

// ============================================================================
// main
// ============================================================================
int main() {
    deadlock_demo();
    hierarchical_mutex_demo();
    try_lock_demo();

    std::cout << "=== 死锁预防总结 ===\n";
    std::cout << "  1. std::lock() / std::scoped_lock: 原子获取多个锁\n";
    std::cout << "  2. 固定锁顺序: 按地址或 ID 排序后获取\n";
    std::cout << "  3. 层级锁: 编译/运行时检查获取顺序\n";
    std::cout << "  4. try_lock 超时: 失败则回退重试\n";
    std::cout << "  5. 减少锁粒度: 细粒度锁, 避免嵌套\n";

    return 0;
}
