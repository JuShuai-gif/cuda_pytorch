// 03_scoped_lock.cpp
// 知识点: std::scoped_lock (C++17) 同时锁多个互斥量，避免死锁
// 演示: scoped_lock vs 手动加锁 vs lock_guard+std::lock
// 对应书中 3.2.4-3.2.5 节

#include <chrono>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// =============================================================================
// 银行账户: 转账场景需要同时锁定两个账户
// =============================================================================
class BankAccount {
public:
    explicit BankAccount(std::string owner, double balance = 0.0)
        : m_owner(std::move(owner)), m_balance(balance) {}

    // 移动构造: 每个账户独立拥有互斥量，移动时不转移锁
    BankAccount(BankAccount&& other) noexcept
        : m_owner(std::move(other.m_owner)), m_balance(other.m_balance) {}

    BankAccount& operator=(BankAccount&& other) noexcept {
        if (this != &other) {
            std::lock_guard<std::mutex> lock_this(m_mutex);
            std::lock_guard<std::mutex> lock_other(other.m_mutex);
            m_owner   = std::move(other.m_owner);
            m_balance = other.m_balance;
        }
        return *this;
    }

    [[nodiscard]] const std::string& owner() const { return m_owner; }

    [[nodiscard]] double balance() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_balance;
    }

    // 存款 (只需锁定自己)
    void deposit(double amount) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_balance += amount;
    }

    // 取款 (只需锁定自己)
    bool withdraw(double amount) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_balance >= amount) {
            m_balance -= amount;
            return true;
        }
        return false;
    }

    // 转账 - 使用 std::scoped_lock 同时锁定两个账户
    // C++17 的方式: 简洁、安全、无死锁
    static void transfer_scoped_lock(BankAccount& from, BankAccount& to,
                                      double amount) {
        // scoped_lock 使用与 std::lock 相同的死锁避免算法
        // 自动同时锁定 from.m_mutex 和 to.m_mutex
        // 如果无法同时获取，会重试，不会持有一个锁等待另一个
        std::scoped_lock lock(from.m_mutex, to.m_mutex);

        if (from.m_balance >= amount) {
            from.m_balance -= amount;
            to.m_balance += amount;
        }
    }

    // 转账 - 使用 std::lock + std::lock_guard (C++14 方式)
    static void transfer_lock_guard(BankAccount& from, BankAccount& to,
                                     double amount) {
        // std::lock: 同时尝试锁定多个互斥量
        std::lock(from.m_mutex, to.m_mutex);

        // std::adopt_lock: 表示互斥量已经被锁定，lock_guard 只负责解锁
        std::lock_guard<std::mutex> lock_from(from.m_mutex, std::adopt_lock);
        std::lock_guard<std::mutex> lock_to(to.m_mutex, std::adopt_lock);

        if (from.m_balance >= amount) {
            from.m_balance -= amount;
            to.m_balance += amount;
        }
    }

    // 错误示范: 可能导致死锁的写法
    static void transfer_bad(BankAccount& from, BankAccount& to,
                              double amount) {
        std::lock_guard<std::mutex> lock_from(from.m_mutex);
        // 如果此时另一个线程执行 transfer_bad(to, from, x)
        // 就会出现死锁: 线程A锁住了from等待to，线程B锁住了to等待from
        std::this_thread::sleep_for(std::chrono::milliseconds(1));  // 模拟延迟
        std::lock_guard<std::mutex> lock_to(to.m_mutex);

        if (from.m_balance >= amount) {
            from.m_balance -= amount;
            to.m_balance += amount;
        }
    }

private:
    std::string     m_owner;
    double          m_balance;
    mutable std::mutex m_mutex;  // mutable 允许在 const 成员中锁定
};

// =============================================================================
// 死锁演示
// =============================================================================
void fct_deadlock_demo() {
    std::cout << "--- 死锁演示 (transfer_bad) ---\n";
    std::cout << "此演示可能导致死锁，使用超时保护\n";

    BankAccount alice("Alice", 1000.0);
    BankAccount bob("Bob", 1000.0);

    // 使用 scoped_lock 安全版本 (避免真正死锁)
    auto safe_task = [](BankAccount& a, BankAccount& b, const std::string& dir) {
        for (int i = 0; i < 100; ++i) {
            BankAccount::transfer_scoped_lock(a, b, 10.0);
        }
        std::cout << "  [" << dir << "] 转账完成\n";
    };

    std::thread t1(safe_task, std::ref(alice), std::ref(bob), "A->B");
    std::thread t2(safe_task, std::ref(bob), std::ref(alice), "B->A");

    t1.join();
    t2.join();

    std::cout << "  Alice 余额: $" << alice.balance() << "\n";
    std::cout << "  Bob   余额: $" << bob.balance() << "\n";
    std::cout << "  总和不变: " << (alice.balance() + bob.balance() == 2000.0
                                         ? "✓"
                                         : "✗")
              << "\n";
}

int main() {
    std::cout << "=== std::scoped_lock (C++17) ===\n\n";

    // --- 测试1: 基本使用 ---
    std::cout << "--- 测试1: std::scoped_lock 基本使用 ---\n";
    {
        BankAccount acc_a("Account-A", 500.0);
        BankAccount acc_b("Account-B", 500.0);

        std::cout << "  转账前: A=$" << acc_a.balance()
                  << " B=$" << acc_b.balance() << "\n";

        BankAccount::transfer_scoped_lock(acc_a, acc_b, 200.0);

        std::cout << "  转账后: A=$" << acc_a.balance()
                  << " B=$" << acc_b.balance() << "\n";
    }

    // --- 测试2: 死锁安全演示 ---
    std::cout << "\n";
    fct_deadlock_demo();

    // --- 测试3: 大量并发转账压力测试 ---
    std::cout << "\n--- 测试3: 并发转账压力测试 ---\n";
    {
        const int num_accounts = 5;
        const int num_threads  = 8;
        const int num_transfers = 1000;

        std::vector<BankAccount> accounts;
        accounts.reserve(num_accounts);
        for (int i = 0; i < num_accounts; ++i) {
            accounts.emplace_back("Account-" + std::to_string(i), 10000.0);
        }

        double initial_total = 0.0;
        for (auto& acc : accounts) {
            initial_total += acc.balance();
        }

        std::vector<std::jthread> threads;
        threads.reserve(num_threads);

        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&accounts, num_accounts, num_transfers]() {
                // 使用 deterministic 伪随机避免每次都从 std::rand() 加锁
                unsigned seed =
                    static_cast<unsigned>(std::hash<std::thread::id>{}(
                        std::this_thread::get_id()));
                for (int i = 0; i < num_transfers; ++i) {
                    int from_idx = (seed + i) % num_accounts;
                    int to_idx   = (seed + i * 3 + 1) % num_accounts;
                    if (from_idx != to_idx) {
                        BankAccount::transfer_scoped_lock(
                            accounts[from_idx], accounts[to_idx], 1.0);
                    }
                }
            });
        }

        threads.clear();  // jthread 自动 join

        double final_total = 0.0;
        for (auto& acc : accounts) {
            final_total += acc.balance();
        }

        std::cout << "  初始总金额: $" << initial_total << "\n";
        std::cout << "  最终总金额: $" << final_total << "\n";
        std::cout << "  金额守恒: "
                  << (std::abs(initial_total - final_total) < 0.01 ? "✓"
                                                                       : "✗")
                  << "\n";
    }

    std::cout << "\n=== 多锁策略对比 ===\n";
    std::cout << "1. 单独 lock_guard: 可能出现死锁\n";
    std::cout << "2. std::lock + lock_guard(adopt): C++11/14 方式，较繁琐\n";
    std::cout << "3. std::scoped_lock: C++17 推荐方式，简洁安全\n";
    std::cout << "4. 原则: 同时锁定多个互斥量时，一次锁定所有\n";
    std::cout << "5. 死锁避免: 固定顺序 | std::lock | scoped_lock\n";

    return 0;
}
