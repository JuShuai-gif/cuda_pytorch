// 04_memory_order_acq_rel.cpp - memory_order_acquire/release 获取-释放序
// 核心：同一原子变量的 release store 与 acquire load 构成同步关系
// release: 之前的所有内存操作不会被重排到之后
// acquire: 之后的所有内存操作不会被重排到之前

#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// ===== 经典生产者-消费者（无锁版本）=====
struct Payload {
    int    id;
    double value;
};

std::atomic<bool>       data_ready{false};  // 同步标志
std::atomic<Payload*>   data_ptr{nullptr};  // 数据指针

// 生产者：准备数据，然后发布
void producer(int id, double value) {
    // release: 保证 data 在 data_ready 之前对其他线程可见
    auto* payload       = new Payload{id, value}; // 注意：仅演示，实际用智能指针
    data_ptr.store(payload, std::memory_order_release);
    data_ready.store(true, std::memory_order_release);

    std::cout << "[Producer] 发布数据: id=" << id
              << ", value=" << value << "\n";
}

// 消费者：等待数据，然后读取
void consumer() {
    // acquire: 保证看到 data_ready 后，也能看到发布前的 data
    while (!data_ready.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    auto* payload = data_ptr.load(std::memory_order_acquire);
    std::cout << "[Consumer] 获取数据: id=" << payload->id
              << ", value=" << payload->value << "\n";

    delete payload; // 清理（配合 new 使用，仅演示）
}

// ===== 引用计数模拟（release/acquire 构建 happens-before）=====
class RefCount {
public:
    void acquire_ref() {
        // relaxed: 当前计数不需要同步
        count_.fetch_add(1, std::memory_order_relaxed);
    }

    bool release_ref() {
        // acq_rel: 既获取上一 release 的可见性，
        //         又发布当前线程的修改给下一 acquire
        if (count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            // 最后一个引用者可以看到之前所有线程写入的数据
            return true; // 可以安全销毁
        }
        return false;
    }

    int value() const { return count_.load(std::memory_order_acquire); }

private:
    std::atomic<int> count_{1};
};

// ===== 自旋锁锁队列（release/acquire 保证临界区顺序）=====
class AcqRelSpinLock {
public:
    void lock() {
        // acquire: 成功获取锁后，临界区代码能看到上一释放者的所有修改
        while (flag_.test_and_set(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }

    void unlock() {
        // release: 释放锁前，临界区所有修改对其他 acquire 可见
        flag_.clear(std::memory_order_release);
    }

private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

int main() {
    // --- 1. 生产者-消费者同步 ---
    {
        std::cout << "=== Release-Acquire 生产者-消费者 ===\n";
        // 必须先启动消费者（它在等待数据）
        std::jthread cons(consumer);
        std::this_thread::sleep_for(50ms); // 确保消费者已就绪
        std::jthread prod([&]() { producer(42, 3.14); });

        prod.join();
        cons.join();
        std::cout << "\n";
    }

    // --- 2. 引用计数 ---
    {
        std::cout << "=== 引用计数 (acq_rel) ===\n";
        RefCount ref;
        ref.acquire_ref(); // 2
        std::cout << "  当前引用: " << ref.value() << "\n";

        bool should_delete = ref.release_ref();
        std::cout << "  释放一次后: " << ref.value()
                  << " (应销毁=" << should_delete << ")\n";

        should_delete = ref.release_ref();
        std::cout << "  再次释放后: " << ref.value()
                  << " (应销毁=" << should_delete << ")\n\n";
    }

    // --- 3. 自旋锁测试 ---
    {
        std::cout << "=== AcqRelSpinLock ===\n";
        AcqRelSpinLock spinlock;
        int            shared = 0;

        std::vector<std::jthread> threads;
        for (int i = 0; i < 4; ++i) {
            threads.emplace_back([&]() {
                for (int j = 0; j < 10000; ++j) {
                    std::lock_guard<AcqRelSpinLock> lock(spinlock);
                    ++shared;
                }
            });
        }
        threads.clear();

        std::cout << "  共享计数器: " << shared << " (期望 40000)\n";
    }

    return 0;
}
