/*
 * lecture16_part1.cpp - 锁的实现与对比
 * Stanford CS149, Fall 2025 - Lecture 16（第16讲：并行编程中的同步原语）
 *
 * 本文件详细演示了多种自旋锁（spinlock）的实现方式，每种锁在
 * 公平性、总线流量、延迟等方面有不同的权衡：
 *
 *   1. Test-and-Set 锁（TAS锁）
 *      - 最简单的自旋锁实现，直接使用硬件原子指令 test_and_set
 *      - 优点：实现简单，低竞争下延迟极低
 *      - 缺点：高竞争下产生 O(P^2) 级别的缓存一致性流量
 *        （每个等待线程都在不停地执行原子写操作，导致大量总线争用）
 *
 *   2. Test-and-Test-and-Set 锁（TTAS锁）
 *      - TAS锁的优化版本，采用"先读再写"的两阶段策略
 *      - 第一阶段：只用普通的 load 指令旋转等待（纯读操作，
 *        不产生总线写流量，缓存行保持在 Shared 状态）
 *      - 第二阶段：当锁看起来空闲时，才尝试 exchange 获取
 *      - 优点：大幅降低总线流量，高竞争下性能远优于 TAS
 *
 *   3. Ticket 锁（排队锁/叫号锁）
 *      - 提供 FIFO（先来先服务）公平性保证
 *      - 每个线程获取一个递增的排队号（ticket），然后等待
 *        now_serving 变量轮到自己
 *      - 关键优势：公平性 + O(P) 级别的总线流量
 *        （每次释放锁只需要一次 invalidate，而非 O(P^2)）
 *      - 类似银行/餐厅的叫号系统
 *
 *   4. 基于 CAS 的锁（Compare-And-Swap 锁）
 *      - 使用 compare_exchange_strong 替代 exchange 实现
 *      - 同样采用两阶段策略：先旋转读取，再尝试 CAS
 *      - CAS 的优势在于硬件层面可能对"比较"路径进行优化，
 *        exchange 则总是无条件写入
 *
 *   5. 基于 CAS 构建的原子 fetch-and-op 操作
 *      - 演示如何用 CAS 原语构建更复杂的原子操作
 *      - atomic_min：原子地设置 *addr = min(*addr, x)
 *      - atomic_fetch_add_cas：用 CAS 模拟 fetch_add
 *      - 通用模式：读取旧值 → 计算新值 → CAS 尝试写入 →
 *        如果 CAS 失败（其他人修改了），重试
 *
 * 编译命令：g++ -std=c++17 -pthread lecture16_part1.cpp -o lecture16_part1
 * 运行命令：./lecture16_part1
 */

#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <chrono>
#include <cassert>

// ============================================================
// 第一部分：Test-and-Set 锁（TAS锁）
// ============================================================
// 最简单的自旋锁实现，直接使用 C++ 的 atomic_flag 类型。
// atomic_flag 是 C++ 中唯一的"保证无锁"原子类型，其
// test_and_set 方法直接映射到硬件的 TAS 指令（如 x86 的 XCHG）。
//
// 工作原理：
//   - test_and_set 将 flag 设置为 true，并返回旧值
//   - 如果旧值为 false，说明之前没人持有锁，当前线程成功获取
//   - 如果旧值为 true，说明锁被他人持有，继续循环
//
// 性能特征：
//   - 低竞争：延迟极低（仅一次原子操作即可获取）
//   - 高竞争：每个等待线程在每次循环迭代中都会执行一次原子写操作，
//     导致所有核心的缓存行不断在 Modified/Invalid 状态间切换，
//     产生 O(P^2) 级别的互联网络流量（P 为线程数）
//
// 缓存一致性协议视角：
//   每个等待线程的 TAS 都会将缓存行置为 Modified 状态，
//   迫使其他核心的该缓存行失效。P 个线程在锁上的旋转
//   会产生 P 倍的 invalidate 消息广播。

class TASLock {
public:
    void lock() {
        // test_and_set：将 flag 原子地设为 true，返回旧值
        // memory_order_acquire：保证获取锁后，后续读操作不会
        //   被重排到此操作之前（与 unlock 中的 release 配对）
        // 如果旧值为 false，说明成功获取锁，退出循环
        // 如果旧值为 true，锁被他人持有，继续旋转等待
        while (flag.test_and_set(std::memory_order_acquire)) {
            // 空循环体：持续尝试 TAS 操作
            // 问题：每次 TAS 都是写操作，产生大量总线流量
            // 在 x86 上，这会被编译为 LOCK XCHG 指令
        }
    }

    void unlock() {
        // memory_order_release：保证释放锁之前的所有写操作
        //   对其他线程可见（与 lock 中的 acquire 配对）
        // clear 将 flag 设为 false（原子写操作）
        flag.clear(std::memory_order_release);
    }

private:
    // ATOMIC_FLAG_INIT：将 atomic_flag 初始化为 false（清除状态）
    // atomic_flag 是 C++ 中唯一的无锁保证类型
    std::atomic_flag flag = ATOMIC_FLAG_INIT;
};

// ============================================================
// 第二部分：Test-and-Test-and-Set 锁（TTAS锁）
// ============================================================
// TAS 锁的优化版本，核心思想是"先读后写"的两阶段策略：
//
// 阶段 1（测试阶段 - 纯读旋转）：
//   使用 load（普通读操作）循环等待锁变为 false。
//   纯读操作不会产生总线写流量，多个核心可以同时以
//   Shared 状态持有该缓存行，不会互相 invalidate。
//
// 阶段 2（设置阶段 - 尝试获取）：
//   当 load 看到锁为 false 时，使用 exchange 原子操作
//   尝试获取锁。exchange 会将值设为 true 并返回旧值。
//   如果旧值为 false → 成功获取
//   如果旧值为 true  → 说明另一个线程抢先获取了，返回阶段 1 重试
//
// 为什么 TTAS 比 TAS 好？
//   在高竞争下，N 个等待线程的 TAS 锁产生 N 倍的 invalidate，
//   而 TTAS 锁的读取旋转阶段不产生 invalidate。只有当锁被释放时，
//   所有等待线程才会同时尝试 exchange，产生一轮竞争。
//
// 注意：TTAS 不保证公平性 - 一个等待很久的线程可能在阶段 2
// 被新到达的线程"抢走"锁。

class TTASLock {
public:
    void lock() {
        while (true) {
            // ===== 阶段 1：纯读旋转（低开销等待） =====
            // load 使用 memory_order_relaxed：不需要任何内存序保证，
            //   因为这里只关心"锁是否空闲"这个事实，
            //   不需要与其他内存操作建立 happens-before 关系
            //   使用 relaxed 可以避免不必要的内存屏障开销
            while (flag.load(std::memory_order_relaxed)) {
                // 忙等待：不产生总线写流量
                // 缓存行在此期间保持在 Shared 状态
                // 多个核心可以同时在这个循环中旋转
            }

            // ===== 阶段 2：尝试原子获取 =====
            // exchange 将 flag 设为 true 并返回旧值
            // memory_order_acquire：如果成功获取锁，
            //   保证后续的临界区代码不会被重排到此操作之前
            if (!flag.exchange(true, std::memory_order_acquire)) {
                return; // 锁获取成功！旧值为 false，当前线程是唯一获取者
            }
            // exchange 返回 true：另一个线程抢先获取了锁
            // 回到阶段 1 继续等待（重新开始读取旋转）
        }
    }

    void unlock() {
        // store 使用 memory_order_release：
        //   保证临界区内的所有写操作在锁释放之前对其他线程可见
        flag.store(false, std::memory_order_release);
    }

private:
    // 使用 atomic<bool> 而非 atomic_flag，
    // 因为需要 load（纯读）和 exchange（读-改-写）两种操作
    // atomic_flag 只支持 test_and_set，不支持 load
    std::atomic<bool> flag{false};
};

// ============================================================
// 第三部分：Ticket 锁（排队锁/叫号锁）
// ============================================================
// Ticket 锁提供 FIFO（先进先出）公平性保证，类似银行或
// 餐厅的"取号-叫号"系统。
//
// 工作原理：
//   1. 每个线程到达时，原子地从 next_ticket 获取一个
//      递增的排队号（类似取号机）
//   2. 线程等待 now_serving（当前正在服务的号码）
//      等于自己的排队号
//   3. 释放锁时，原子地递增 now_serving，
//      通知下一个排队的线程
//
// 关键优势：
//   - 公平性：严格按到达顺序服务，不会出现"饥饿"（starvation）
//   - 低总线流量：释放锁只需递增 now_serving（一次原子操作），
//     产生 O(1) 的 invalidate，而非 TAS 的 O(P^2)
//   - 等待线程只读取 now_serving（load 操作），不执行写操作
//
// 缺点：
//   - 需要两个原子变量（next_ticket 和 now_serving），
//     占用更多缓存行
//   - 每次 lock() 都需要一个 fetch_add（写操作），
//     即使锁当前空闲
//
// 缓存一致性流量分析：
//   释放锁时 now_serving 递增，所有等待线程持有的
//   now_serving 缓存行变为 Invalid，它们需要重新获取。
//   但这只是 O(P) 级别的流量（每个等待线程一次），
//   而非 TAS 的 O(P^2)（每个等待线程每次循环都触发的写操作）。

class TicketLock {
public:
    void lock() {
        // fetch_add：原子地将 next_ticket 增加 1，并返回旧值
        // 这个操作就像从取号机上取一个号码
        // memory_order_relaxed：这里不需要与 now_serving 建立顺序关系，
        //   只需要保证"取号"操作本身是原子的
        unsigned int my_ticket = next_ticket.fetch_add(1, std::memory_order_relaxed);

        // 旋转等待：直到 now_serving 等于我的号码
        // load 使用 memory_order_acquire：保证进入临界区后
        //   能看到释放锁的线程在临界区内的所有修改
        while (now_serving.load(std::memory_order_acquire) != my_ticket) {
            // 纯读等待：不产生总线写流量
            // 等待线程只读取 now_serving，不修改任何共享状态
            // 这保证了 O(P) 而非 O(P^2) 的总线流量
        }
    }

    void unlock() {
        // fetch_add 将 now_serving 递增 1：
        //   如果当前服务号是 N，递增后变为 N+1，
        //   持有号码 N+1 的线程会看到 now_serving == 自己的号码
        // memory_order_release：保证临界区内的修改对新进入者可见
        now_serving.fetch_add(1, std::memory_order_release);
    }

private:
    // next_ticket：下一个可用的排队号（类似取号机的计数器）
    // 初始值 0：第一个线程将获取号码 0
    std::atomic<unsigned int> next_ticket{0};

    // now_serving：当前正在服务的号码（类似叫号显示屏）
    // 初始值 0：第一个进入临界区的线程需要号码 0 == now_serving
    std::atomic<unsigned int> now_serving{0};
};

// ============================================================
// 第四部分：基于 CAS 的锁（Compare-And-Swap 锁）
// ============================================================
// 与 TTAS 锁类似，但使用 compare_exchange_strong 替代 exchange
// 来获取锁。两者都是"先读旋转，再尝试原子获取"的模式。
//
// compare_exchange_strong 与 exchange 的区别：
//   - exchange：无条件地将新值写入，返回旧值
//   - compare_exchange_strong：只有当当前值等于期望值时，
//     才写入新值；否则不写入且更新期望值
//
// 为什么 CAS 可能比 exchange 更高效？
//   在现代 x86 处理器上，LOCK CMPXCHG（CAS 的硬件实现）
//   在某些微架构上可能比 LOCK XCHG 略有优势，因为：
//   1. CAS 的"比较"路径在某些情况下可以更早检测到失败
//   2. 如果当前值不等于期望值，CAS 不会写入，
//      减少了不必要的缓存行状态变更
//
// 但实际性能差异很小，TTAS 和 CAS 锁在大多数场景下
// 性能相近，都远优于纯 TAS 锁。

class CASLock {
public:
    void lock() {
        while (true) {
            // ===== 阶段 1：纯读旋转 =====
            // 与 TTAS 锁相同：先用 load 旋转等待，
            // 不产生总线写流量
            while (locked.load(std::memory_order_relaxed)) {
            }

            // ===== 阶段 2：尝试 CAS 获取 =====
            // compare_exchange_strong 的语义：
            //   如果 locked == expected (false)，则将 locked 设为 true，
            //   返回 true（表示交换成功，获取锁）
            //   如果 locked != expected (false)，则将 expected 更新为
            //   locked 的当前值，返回 false（表示交换失败）
            //
            // memory_order_acquire（成功时）：获取语义
            // memory_order_relaxed（失败时）：失败时不需要严格的内存序
            bool expected = false;
            if (locked.compare_exchange_strong(expected, true,
                    std::memory_order_acquire, std::memory_order_relaxed)) {
                return; // CAS 成功：locked 从 false 变成 true，获取锁
            }
            // CAS 失败：另一个线程抢先获取了锁
            // expected 已被更新为当前值（true），继续循环
        }
    }

    void unlock() {
        // 释放锁：简单地写入 false
        locked.store(false, std::memory_order_release);
    }

private:
    std::atomic<bool> locked{false};
};

// ============================================================
// 第五部分：基于 CAS 构建的原子 fetch-and-op 操作
// ============================================================
// 本部分演示了如何使用 CAS（Compare-And-Swap）这个基本原子原语
// 来构建更复杂的原子操作。这是理解"无锁编程"的基础。
//
// 通用模式（CAS 循环模式）：
//   1. 读取当前值 → 2. 计算期望的新值 → 3. CAS 尝试写入 →
//   4. 如果失败（其他线程修改了），重试（回到步骤 2）
//
// 这种模式是构建无锁数据结构（lock-free data structures）的核心技术。

// ============================================================
// 原子最小值操作：原子地设置 *addr = min(*addr, x)
// ============================================================
// 此函数演示了如何用 CAS 实现一个"读-修改-写"原子操作。
// 与直接使用硬件提供的 fetch_add 不同，atomic_min 没有
// 直接的硬件支持（x86 没有"原子最小值"指令），所以必须
// 用 CAS 循环来模拟。
//
// 使用 compare_exchange_weak 而非 _strong 的原因：
//   - _weak 在某些平台上可能更高效（如 ARM 的 LDREX/STREX）
//   - 在 CAS 循环中，_weak 可能发生"虚假失败"（spurious failure），
//     但由于我们在循环中会重试，这不是问题
//   - _weak 的虚假失败反而避免了 _strong 中额外的重试逻辑开销
//
// 参数说明：
//   addr - 要原子修改的地址（引用传递）
//   x    - 要比较的值

void atomic_min(std::atomic<int>& addr, int x) {
    // 步骤 1：读取当前值
    int old_val = addr.load(std::memory_order_relaxed);

    // 步骤 2：计算期望的新值
    int new_val = std::min(old_val, x);

    // 步骤 3-4：CAS 循环
    // compare_exchange_weak 的语义：
    //   如果 addr == old_val，将 addr 设为 new_val，返回 true
    //   如果 addr != old_val（竞争条件），将 old_val 更新为 addr 的当前值，
    //   返回 false，然后循环重试
    //
    // memory_order_release（成功时）：保证写操作对后续读者可见
    // memory_order_relaxed（失败时）：失败路径不需要严格内存序
    while (!addr.compare_exchange_weak(old_val, new_val,
            std::memory_order_release, std::memory_order_relaxed)) {
        // CAS 失败的处理：
        // compare_exchange_weak 在失败时会自动将 old_val
        // 更新为 addr 的当前值，所以不需要额外的 load 操作
        // 这是 C++ 原子库的一个重要便利特性

        // 基于新的 old_val 重新计算 new_val
        new_val = std::min(old_val, x);
    }
    // 循环最多重试 N 次，其中 N = 同时修改 addr 的线程数
}

// ============================================================
// 基于 CAS 实现的原子加法
// ============================================================
// 虽然 C++ 提供了原生的 fetch_add，但这里演示如何用 CAS
// 手动实现，以展示 CAS 循环模式的通用性。
//
// 返回值：加法之前的值（遵循 fetch_add 的语义）

int atomic_fetch_add_cas(std::atomic<int>& addr, int x) {
    // 步骤 1：读取当前值
    int old_val = addr.load(std::memory_order_relaxed);

    // 步骤 2-3：CAS 循环
    // 注意：这里直接在 compare_exchange_weak 的参数中
    // 计算 old_val + x，而不是预先计算 new_val
    while (!addr.compare_exchange_weak(old_val, old_val + x,
            std::memory_order_release, std::memory_order_relaxed)) {
        // CAS 失败：old_val 已被 compare_exchange_weak 自动更新
        // 循环回去继续尝试（此时 old_val 是新的当前值）
        // old_val + x 会基于更新后的旧值重新计算
    }

    // 返回加法前的值（这是 fetch_add 的标准语义）
    return old_val;
}

// ============================================================
// 演示部分：竞争条件下的计数器递增
// ============================================================
// 多线程使用不同类型的锁来递增共享计数器，以验证：
//   1. 正确性：所有锁都能正确保护临界区
//   2. 性能：不同锁在高竞争下的执行时间差异

// 通用计数器递增函数模板
// LockType 必须满足 Lockable 概念（有 lock() 和 unlock() 方法）
// std::lock_guard 会自动在构造时调用 lock()，析构时调用 unlock()
// 这是 RAII（资源获取即初始化）模式在锁管理中的应用
template <typename LockType>
void increment_counter(LockType& lock, int& counter, int iterations) {
    for (int i = 0; i < iterations; ++i) {
        // lock_guard 是作用域锁：进入此作用域时获取锁，
        // 离开作用域时（包括异常退出）自动释放锁
        // 这保证了即使在异常情况下也不会死锁
        std::lock_guard<LockType> guard(lock);
        ++counter;
    }
}

// 所有锁类型（TASLock, TTASLock, TicketLock, CASLock）都实现了
// lock() 和 unlock() 方法，因此都满足 Lockable 概念，
// 可以直接与 std::lock_guard 一起使用。

// 锁性能基准测试模板
// name        - 锁类型的名称（用于输出）
// num_threads - 并发线程数
// per_thread  - 每个线程的递增次数
template <typename LockType>
void run_benchmark(const std::string& name, int num_threads, int per_thread) {
    LockType lock;
    int counter = 0;
    std::vector<std::thread> threads;

    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();

    // 创建 num_threads 个线程，每个线程执行 increment_counter
    // std::ref 用于传递引用（因为 std::thread 默认按值复制参数）
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(increment_counter<LockType>,
                             std::ref(lock), std::ref(counter), per_thread);
    }

    // 等待所有线程完成（join：阻塞直到线程结束）
    for (auto& t : threads) {
        t.join();
    }

    // 记录结束时间并计算耗时
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // 输出基准测试结果
    // 格式：[锁名称] 线程数=X 总操作数=Y 计数器值=Z 耗时=Wms
    std::cout << "[" << name << "] 线程数=" << num_threads
              << " 总操作数=" << (num_threads * per_thread)
              << " 计数器值=" << counter
              << " 耗时=" << duration.count() << "ms" << std::endl;

    // 验证正确性：计数器的最终值应该等于所有线程操作的总和
    // 如果断言失败，说明锁实现有 bug（存在数据竞争）
    assert(counter == num_threads * per_thread);
}

int main() {
    std::cout << "=== CS149 第16讲：锁的实现与对比 ===" << std::endl;
    std::cout << std::endl;

    // ---- 第五部分演示：基于 CAS 的原子 fetch-and-op ----
    std::cout << "--- 基于 CAS 的原子 fetch-and-op 操作 ---" << std::endl;
    {
        std::atomic<int> val{42};
        atomic_min(val, 10);
        std::cout << "atomic_min(42, 10) = " << val.load() << " （期望值：10）" << std::endl;

        val.store(5);
        atomic_min(val, 10);
        std::cout << "atomic_min(5, 10) = " << val.load() << " （期望值：5）" << std::endl;

        val.store(0);
        int old = atomic_fetch_add_cas(val, 5);
        std::cout << "atomic_fetch_add_cas(0, 5)：返回值 " << old
                  << "，新值 = " << val.load() << " （期望：返回值0，新值5）" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "--- 锁性能基准测试（正确性 + 计时） ---" << std::endl;

    // 使用 4 个线程，每个线程递增 100000 次
    // 总共 400000 次临界区进入/退出操作
    const int num_threads = 4;
    const int per_thread = 100000;

    // 运行四种锁的基准测试
    // 注意：在高竞争下，TAS 锁通常是最慢的（O(P^2) 总线流量），
    //       TTAS 和 Ticket 锁应该是最快的（低总线流量）
    run_benchmark<TASLock>("TAS 锁    ", num_threads, per_thread);
    run_benchmark<TTASLock>("TTAS 锁   ", num_threads, per_thread);
    run_benchmark<TicketLock>("Ticket 锁 ", num_threads, per_thread);
    run_benchmark<CASLock>("CAS 锁    ", num_threads, per_thread);

    std::cout << std::endl;
    std::cout << "所有锁实现均已通过正确性验证。" << std::endl;
    std::cout << "注意：TAS 锁在高竞争下通常最慢（高总线流量），" << std::endl;
    std::cout << "TTAS 锁和 Ticket 锁通常最快（低总线流量）。" << std::endl;

    return 0;
}
