// 05_memory_order_seq_cst.cpp - memory_order_seq_cst 顺序一致性（默认）
// 最强内存序：所有线程看到相同的操作全序
// 也是 std::atomic 的默认内存序

#include <atomic>
#include <cassert>
#include <iostream>
#include <thread>
#include <vector>

// ===== Demo 1: seq_cst 保证全局一致顺序 =====
// 两个线程各自写入两个原子变量，seq_cst 保证不存在
// 线程 A 看到 x=true, y=false 同时线程 B 看到 y=true, x=false 的情况

std::atomic<bool> x{false};
std::atomic<bool> y{false};
std::atomic<int>  z{0};

void write_x() {
    x.store(true, std::memory_order_seq_cst);  // (1)
}

void write_y() {
    y.store(true, std::memory_order_seq_cst);  // (2)
}

void read_x_then_y() {
    while (!x.load(std::memory_order_seq_cst)) { }  // (3)
    if (y.load(std::memory_order_seq_cst)) {         // (4)
        z.fetch_add(1, std::memory_order_seq_cst);
    }
}

void read_y_then_x() {
    while (!y.load(std::memory_order_seq_cst)) { }  // (5)
    if (x.load(std::memory_order_seq_cst)) {         // (6)
        z.fetch_add(1, std::memory_order_seq_cst);
    }
}

void demo_sequential_consistency() {
    std::cout << "=== 顺序一致性保证测试 ===\n";
    std::cout << "seq_cst 下，z 永远不会为 0\n";

    // 重置状态
    x.store(false, std::memory_order_seq_cst);
    y.store(false, std::memory_order_seq_cst);
    z.store(0, std::memory_order_seq_cst);

    std::jthread t1(write_x);
    std::jthread t2(write_y);
    std::jthread t3(read_x_then_y);
    std::jthread t4(read_y_then_x);

    t1.join(); t2.join(); t3.join(); t4.join();

    int result = z.load(std::memory_order_seq_cst);
    std::cout << "  z = " << result << " (预期 >= 1，seq_cst 下不可能为 0)\n\n";
}

// ===== Demo 2: 多线程全序递增 =====
void demo_total_order_increment() {
    std::cout << "=== 多线程 seq_cst 递增（全序） ===\n";
    std::atomic<int> counter{0};
    const int        kThreads = 8;
    const int        kIters   = 10000;

    std::vector<std::jthread> threads;
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < kIters; ++j) {
                counter.fetch_add(1, std::memory_order_seq_cst);
            }
        });
    }
    threads.clear();

    std::cout << "  counter = " << counter.load()
              << " (期望 " << kThreads * kIters << ")\n\n";
}

// ===== Demo 3: Peterson 锁（依赖 seq_cst） =====
class PetersonLock {
public:
    void lock(int thread_id) {
        int other = 1 - thread_id;

        flag_[thread_id].store(true, std::memory_order_seq_cst);
        turn_.store(other, std::memory_order_seq_cst);

        // 自旋等待：如果对方有 flag 且轮到自己谦让
        while (flag_[other].load(std::memory_order_seq_cst) &&
               turn_.load(std::memory_order_seq_cst) == other) {
            std::this_thread::yield();
        }
    }

    void unlock(int thread_id) {
        flag_[thread_id].store(false, std::memory_order_seq_cst);
    }

private:
    std::atomic<bool> flag_[2]{false, false};
    std::atomic<int>  turn_{0};
};

void demo_peterson_lock() {
    std::cout << "=== Peterson 锁（需要 seq_cst） ===\n";
    PetersonLock lock;
    int          shared = 0;

    std::jthread t1([&]() {
        for (int i = 0; i < 50000; ++i) {
            lock.lock(0);
            ++shared;
            lock.unlock(0);
        }
    });

    std::jthread t2([&]() {
        for (int i = 0; i < 50000; ++i) {
            lock.lock(1);
            ++shared;
            lock.unlock(1);
        }
    });

    t1.join(); t2.join();
    std::cout << "  shared = " << shared << " (期望 100000)\n\n";
}

// ===== Demo 4: 对比 relaxed —— 可能看到不一致的结果 =====
// (relaxed 版本运行多次可能出现 z=0)

std::atomic<bool> xr{false};
std::atomic<bool> yr{false};
std::atomic<int>  zr{0};

void relaxed_write_x() { xr.store(true, std::memory_order_relaxed); }
void relaxed_write_y() { yr.store(true, std::memory_order_relaxed); }

void relaxed_read_x_then_y() {
    while (!xr.load(std::memory_order_relaxed)) { }
    if (yr.load(std::memory_order_relaxed)) {
        zr.fetch_add(1, std::memory_order_relaxed);
    }
}

void relaxed_read_y_then_x() {
    while (!yr.load(std::memory_order_relaxed)) { }
    if (xr.load(std::memory_order_relaxed)) {
        zr.fetch_add(1, std::memory_order_relaxed);
    }
}

void demo_relaxed_anomaly() {
    std::cout << "=== 运行 relaxed 版本 100 次，可能 z=0 ===\n";
    int zero_count = 0;

    for (int run = 0; run < 100; ++run) {
        xr.store(false, std::memory_order_relaxed);
        yr.store(false, std::memory_order_relaxed);
        zr.store(0, std::memory_order_relaxed);

        std::jthread t1(relaxed_write_x);
        std::jthread t2(relaxed_write_y);
        std::jthread t3(relaxed_read_x_then_y);
        std::jthread t4(relaxed_read_y_then_x);

        t1.join(); t2.join(); t3.join(); t4.join();

        if (zr.load(std::memory_order_relaxed) == 0) ++zero_count;
    }

    std::cout << "  100 次运行中 z=0 出现了 " << zero_count << " 次\n";
    std::cout << "  (relaxed 下两个读者可能都看不到对方的写入)\n";
}

int main() {
    demo_sequential_consistency();
    demo_total_order_increment();
    demo_peterson_lock();
    demo_relaxed_anomaly();

    return 0;
}
