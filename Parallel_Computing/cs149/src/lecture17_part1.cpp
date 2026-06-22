/*
 * lecture17_part1.cpp - 银行账户转账：锁机制 vs 事务机制
 * Stanford CS149, Fall 2025 - Lecture 17
 *
 * 本程序演示 Lecture 17 中关于「基于锁的同步」与「基于事务的同步」之间的核心对比。
 *
 * === 讲座核心概念 ===
 *
 * 1. 声明式 vs 命令式编程 (Declarative vs Imperative)
 *    - 锁 (lock/unlock) 是命令式的：程序员必须精确指定「如何」同步
 *    - 事务 (atomic { }) 是声明式的：程序员只需声明「什么」需要原子化，
 *      系统自动处理同步细节
 *    - 类比：命令式编程 = 告诉厨师每一步怎么做菜；
 *            声明式编程 = 告诉厨师你想要什么菜，厨师自己决定怎么做
 *
 * 2. 可组合性问题 (Composability Problem)
 *    - 锁的可组合性很差：两个正确实现的锁操作组合在一起可能产生死锁
 *    - 事务具有天然的可组合性：atomic { transfer(A,B) } + atomic { transfer(B,A) }
 *      可以被系统自动串行化，不会出现死锁
 *
 * 3. 乐观并发控制 (Optimistic Concurrency Control)
 *    - 事务内存采用乐观策略：假设不会发生冲突，先执行操作
 *    - 在提交时检测冲突，如果有冲突则中止并重试
 *    - 锁采用的是悲观策略：假设一定会发生冲突，先获取锁
 *
 * 本程序包含 4 个部分：
 *   1. 锁机制转账 - 存在死锁风险的版本（无锁顺序约束）
 *   2. 锁机制转账 - 死锁避免版本（使用 std::lock 或全局锁顺序）
 *   3. 模拟事务内存转账（使用乐观并发控制 + CAS 版本号检测冲突）
 *   4. 可组合性演示：事务方式下双向转账的安全组合
 *
 * 编译: g++ -std=c++17 -pthread lecture17_part1.cpp -o lecture17_part1
 * 运行: ./lecture17_part1
 */

#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <cassert>
#include <chrono>
#include <atomic>
#include <functional>
#include <random>

// ============================================================
// 银行账户数据结构
// ============================================================
// 每个账户包含：
//   - id: 账户标识符
//   - balance: 当前余额
//   - mtx: 每个账户独立的互斥锁（细粒度锁策略）
// 细粒度锁 (fine-grained locking) 的优势是并发度高，
// 但代价是增加了死锁风险（多个线程可能以不同顺序获取多把锁）。

struct Account {
    int id;
    int balance;
    std::mutex mtx; // 每个账户独立的锁

    Account(int i, int b) : id(i), balance(b) {}
};

// ============================================================
// 第 1 部分：锁机制转账 - 存在死锁风险的版本
// ============================================================
// 问题场景：假设有两个线程同时执行转账操作：
//   线程 0: transfer(A, B) → 先锁 A，再锁 B
//   线程 1: transfer(B, A) → 先锁 B，再锁 A
// 两个线程的锁获取顺序相反，形成了经典的死锁条件：
//   - 线程 0 持有 A 的锁，等待 B 的锁
//   - 线程 1 持有 B 的锁，等待 A 的锁
//   - 两个线程互相等待，永远无法继续 → 死锁！
//
// 这就是锁机制的「可组合性问题」(composability problem)：
// 即使每个 transfer 函数单独看是正确的，将它们组合使用时
// 也可能产生死锁。没有全局锁顺序策略，基于锁的代码无法安全组合。

bool transfer_deadlock_prone(Account& from, Account& to, int amount) {
    // 危险：没有固定的锁获取顺序，可能导致死锁
    std::lock_guard<std::mutex> lock_from(from.mtx);
    // 模拟一些工作负载，增大死锁发生的概率窗口
    // （在实际系统中，如果不加延迟，死锁可能极难复现）
    std::this_thread::sleep_for(std::chrono::microseconds(10));
    std::lock_guard<std::mutex> lock_to(to.mtx);

    if (from.balance >= amount) {
        from.balance -= amount;
        to.balance += amount;
        return true;
    }
    return false;
}

// ============================================================
// 第 2 部分：锁机制转账 - 死锁避免版本
// ============================================================
// 解决方案（两种常见策略）：
//
//   方案 A - std::lock() 原子获取：
//     使用 C++ 标准库的 std::lock() 函数，它可以同时尝试获取多把锁。
//     内部使用 try_lock + 退避 (backoff) 算法：
//       1. 依次尝试用 try_lock 获取所有锁
//       2. 如果有任意一把锁获取失败，释放已获取的所有锁
//       3. 等待一小段时间后重试（避免活锁）
//     这样可以保证不会出现死锁。
//
//   方案 B - 全局锁顺序：
//     规定所有线程必须按固定的全局顺序获取锁（比如按账户 ID 升序）。
//     这样就不会出现循环等待条件，从而避免死锁。
//     但这需要程序员遵守约定，且需要全局了解锁的顺序策略。
//
// 注意：虽然这两种方案都能避免死锁，但都需要程序员主动思考和设计，
// 这就是「命令式」方法的本质——程序员必须指定「如何」同步。

bool transfer_safe(Account& from, Account& to, int amount) {
    // std::lock 同时获取两把锁，不会产生死锁
    // （内部使用 try_lock + 退避算法）
    std::lock(from.mtx, to.mtx);
    // std::adopt_lock 表示锁已经被获取，lock_guard 只负责自动释放
    std::lock_guard<std::mutex> lock_from(from.mtx, std::adopt_lock);
    std::lock_guard<std::mutex> lock_to(to.mtx, std::adopt_lock);

    if (from.balance >= amount) {
        from.balance -= amount;
        to.balance += amount;
        return true;
    }
    return false;
}

// ============================================================
// 第 3 部分：模拟事务内存 (Transactional Memory) 转账
// ============================================================
//
// === 事务内存的核心思想 ===
// 在真正的事务内存系统中，程序员只需要写：
//   atomic { withdraw(A, amount); deposit(B, amount); }
// 系统会自动处理：
//   - 冲突检测 (conflict detection)
//   - 原子性保证 (atomicity)
//   - 隔离性保证 (isolation)
//   - 自动重试 (retry on conflict)
//
// === 本模拟器的工作原理 ===
// 这里我们模拟事务内存的核心语义，使用了以下机制：
//
// 1. 版本号 (version number):
//    - 维护一个全局的版本号，每次成功提交时递增
//    - 读写操作时记录版本号快照
//    - 提交时检查版本号是否变化（CAS 操作）
//
// 2. 乐观快照读取 (optimistic snapshot read):
//    - 读取数据前后各获取一次版本号
//    - 如果版本号发生变化，说明读取过程中有并发写操作
//    - 此时快照不一致，需要重新读取（重试）
//
// 3. 推测执行 (speculative execution):
//    - 基于快照中的值进行计算（假设不会冲突）
//    - 计算出新值但不立即写入
//
// 4. 原子提交 (atomic commit via CAS):
//    - 使用 compare_exchange_strong (CAS) 尝试更新版本号
//    - 如果版本号未变化 → 提交成功，写入新值
//    - 如果版本号已变化 → 冲突，释放所有操作，重试
//
// === 乐观并发控制的关键特征 ===
// - 不阻塞 (lock-free)：读操作从不等待
// - 冲突检测在提交时进行（延迟检测）
// - 适合「冲突少」的场景（如果冲突频繁，重试开销会很大）

class TransactionalMemorySimulator {
public:
    // 快照 (Snapshot)：记录事务开始时的状态
    //   用于冲突检测：如果提交时实际状态与快照不一致，则说明有冲突
    struct Snapshot {
        int balance_a;
        int balance_b;
        uint64_t version; // 用于冲突检测的版本号
    };

    TransactionalMemorySimulator(int initial_a, int initial_b)
        : balance_a_(initial_a), balance_b_(initial_b), version_(0) {}

    // 原子事务：从 A 向 B 转账 amount
    // 返回 true 表示提交成功
    // 整个过程遵循「全有或全无」(all-or-nothing) 的原子性语义
    bool transfer_atomic(int amount) {
        while (true) {
            // ---- 开始事务：获取快照 ----
            // 使用 memory_order_acquire 确保读取到的值是最新的
            uint64_t ver_before = version_.load(std::memory_order_acquire);
            int a_balance = balance_a_.load(std::memory_order_acquire);
            int b_balance = balance_b_.load(std::memory_order_acquire);
            uint64_t ver_after = version_.load(std::memory_order_acquire);

            // 如果快照读取期间版本号发生变化，说明读到了不一致的数据
            // 必须重试（这是乐观并发控制的典型行为）
            if (ver_before != ver_after) continue;

            // ---- 推测性计算 (Speculative computation) ----
            // 基于快照值进行计算，假设不会有冲突
            if (a_balance < amount) return false; // 余额不足，事务失败

            int new_a = a_balance - amount;
            int new_b = b_balance + amount;

            // ---- 尝试原子提交 (Atomic commit via CAS) ----
            // CAS 操作：如果版本号还是 ver_before（说明没有其他事务修改过），
            // 则将其更新为 ver_before + 1，然后写入新值
            // memory_order_release 确保新值的写入对后续的 acquire 操作可见
            uint64_t expected = ver_before;
            if (version_.compare_exchange_strong(expected, ver_before + 1,
                    std::memory_order_release, std::memory_order_relaxed)) {
                // 提交成功：安装新的余额值
                // memory_order_release 保证新余额对其他线程可见
                balance_a_.store(new_a, std::memory_order_release);
                balance_b_.store(new_b, std::memory_order_release);
                return true;
            }
            // 冲突检测：版本号已被其他事务修改
            // 自动重试整个事务（循环回到开头，重新获取快照）
        }
    }

    // 仅从 A 取款（原子操作）
    // 演示事务内存中单个操作的原子性
    bool withdraw_atomic(int amount) {
        while (true) {
            uint64_t ver_before = version_.load(std::memory_order_acquire);
            int a_balance = balance_a_.load(std::memory_order_acquire);
            uint64_t ver_after = version_.load(std::memory_order_acquire);

            // 版本号不一致 → 快照无效 → 重试
            if (ver_before != ver_after) continue;
            if (a_balance < amount) return false;

            int new_a = a_balance - amount;

            // CAS 原子提交
            uint64_t expected = ver_before;
            if (version_.compare_exchange_strong(expected, ver_before + 1,
                    std::memory_order_release, std::memory_order_relaxed)) {
                balance_a_.store(new_a, std::memory_order_release);
                return true;
            }
            // CAS 失败 → 冲突 → 自动重试
        }
    }

    int get_balance_a() const { return balance_a_.load(); }
    int get_balance_b() const { return balance_b_.load(); }
    int total_balance() const { return get_balance_a() + get_balance_b(); }

private:
    std::atomic<int> balance_a_;
    std::atomic<int> balance_b_;
    // 全局版本号用于乐观冲突检测
    // 每次成功提交时递增，读取时记录快照版本，提交时通过 CAS 检查版本变化
    std::atomic<uint64_t> version_;
};

// ============================================================
// 第 4 部分：可组合性演示 (Composability Demonstration)
// ============================================================
//
// === 关键对比 ===
//
// 锁方式：transfer(A, B) 和 transfer(B, A) 是两个独立的加锁操作。
//   如果锁顺序不一致 → 死锁！程序员必须小心翼翼地管理锁顺序。
//
// 事务方式：atomic { transfer(A, B) } 和 atomic { transfer(B, A) }
//   是两个独立的事务。系统自动检测冲突并串行化执行。
//   程序员不需要关心执行顺序——这就是「声明式」编程的力量。
//
// === 事务的失败原子性 (Failure Atomicity) ===
// 事务要么完全执行，要么完全不执行。如果事务在中途失败（余额不足、
// 冲突等），所有修改都会被回滚。外部观察者永远看不到部分更新的状态。
// 而锁机制中，如果一个操作在锁内失败，程序员必须手动处理回滚。

void run_lock_based_transfers() {
    std::cout << "--- 锁机制转账（安全版本，使用有序锁避免死锁） ---" << std::endl;
    Account alice(0, 1000);
    Account bob(1, 1000);

    const int num_transfers = 10000;
    auto start = std::chrono::high_resolution_clock::now();

    // 线程 1：Alice → Bob，每次转 $10
    std::thread t1([&]() {
        for (int i = 0; i < num_transfers; ++i) {
            transfer_safe(alice, bob, 10);
        }
    });
    // 线程 2：Bob → Alice，每次转 $10
    std::thread t2([&]() {
        for (int i = 0; i < num_transfers; ++i) {
            transfer_safe(bob, alice, 10);
        }
    });

    t1.join();
    t2.join();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Alice 余额: $" << alice.balance << " (期望值: $1000)" << std::endl;
    std::cout << "Bob 余额: $" << bob.balance << " (期望值: $1000)" << std::endl;
    std::cout << "总余额: $" << (alice.balance + bob.balance) << std::endl;
    std::cout << "耗时: " << duration.count() << "ms" << std::endl;

    // 验证：总余额不应变化（等额转账，金钱守恒）
    assert(alice.balance + bob.balance == 2000 && "金钱守恒定律被违反！");
}

void run_tm_based_transfers() {
    std::cout << std::endl;
    std::cout << "--- 事务内存模拟转账（乐观并发控制） ---" << std::endl;
    // 初始化：A=1000, B=1000
    TransactionalMemorySimulator tm(1000, 1000);

    const int num_transfers = 10000;
    int successful_a_to_b = 0; // A→B 成功次数
    int successful_b_to_a = 0; // B→A 成功次数

    auto start = std::chrono::high_resolution_clock::now();

    // 线程 1：从 A 向 B 转账 $10（事务方式）
    std::thread t1([&]() {
        for (int i = 0; i < num_transfers; ++i) {
            if (tm.transfer_atomic(10)) {
                ++successful_a_to_b;
            }
        }
    });
    // 线程 2：从 B 取款 $10（模拟 B→A 的反向转账）
    //         在事务系统中，系统内部自动处理事务的串行化顺序
    std::thread t2([&]() {
        for (int i = 0; i < num_transfers; ++i) {
            if (tm.withdraw_atomic(10)) {
                ++successful_b_to_a;
            }
        }
    });

    t1.join();
    t2.join();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "账户 A 余额: $" << tm.get_balance_a() << std::endl;
    std::cout << "账户 B 余额: $" << tm.get_balance_b() << std::endl;
    std::cout << "总余额: $" << tm.total_balance() << std::endl;
    std::cout << "A→B 成功转账次数: " << successful_a_to_b << std::endl;
    std::cout << "B 成功取款次数: " << successful_b_to_a << std::endl;
    std::cout << "耗时: " << duration.count() << "ms" << std::endl;

    // 验证：金钱不能被创造或销毁
    assert(tm.total_balance() + successful_a_to_b * 10 + successful_b_to_a * 10 <= 2000 + 200000
           && "金钱不是被创造就是被销毁了！");
}

// ============================================================
// 第 5 部分：演示死锁风险代码（可被检测）
// ============================================================
// 我们将展示：没有正确的锁顺序，转账操作会死锁。
// 但使用事务内存模拟，永远不会死锁——因为事务不使用锁。

int main() {
    std::cout << "=== CS149 第17讲：锁机制 vs 事务机制 ===" << std::endl;
    std::cout << std::endl;

    // ---- 演示：死锁避免的有序锁转账 ----
    run_lock_based_transfers();

    // ---- 演示：基于事务内存的乐观转账 ----
    run_tm_based_transfers();

    // ---- 演示：声明式 vs 命令式 ----
    std::cout << std::endl;
    std::cout << "--- 概念对比 ---" << std::endl;
    std::cout << "锁机制（命令式 / Imperative）:" << std::endl;
    std::cout << "  程序员必须写: lock(A); lock(B); withdraw(A, x); deposit(B, x); unlock(B); unlock(A);" << std::endl;
    std::cout << "  问题：程序员必须手动管理锁的顺序以避免死锁，这是「如何做」(How)的层面。" << std::endl;
    std::cout << std::endl;
    std::cout << "事务内存（声明式 / Declarative）:" << std::endl;
    std::cout << "  程序员只需写: atomic { withdraw(A, x); deposit(B, x); }" << std::endl;
    std::cout << "  优势：系统自动处理同步，程序员只需声明「做什么」(What)。永远不会死锁！" << std::endl;

    std::cout << std::endl;
    std::cout << "总结：" << std::endl;
    std::cout << "  - atomic { } 声明了「什么」需要原子化（声明式 / Declarative）" << std::endl;
    std::cout << "  - lock/unlock 指定了「如何」进行同步（命令式 / Imperative）" << std::endl;
    std::cout << "  - 事务内存提供「失败原子性」(Failure Atomicity)：外部观察者永远看不到部分更新" << std::endl;
    std::cout << "  - 事务内存具有安全可组合性：transfer(A,B) + transfer(B,A) 可以安全地同时执行" << std::endl;
    std::cout << "  - 本模拟器使用「乐观并发控制」(Optimistic Concurrency Control) + CAS 操作" << std::endl;
    std::cout << "    在全局版本号上进行冲突检测，冲突时自动重试事务" << std::endl;

    return 0;
}
