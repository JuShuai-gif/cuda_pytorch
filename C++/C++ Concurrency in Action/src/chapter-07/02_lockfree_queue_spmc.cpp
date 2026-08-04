/**
 * 02_lockfree_queue_spmc.cpp — 单生产者多消费者无锁环形缓冲队列
 *
 * 基于固定大小环形缓冲区 (ring buffer) 的无锁队列。
 * SPSC/SPMC 场景下不需要 CAS, 仅需原子 load/store 即可保证正确性。
 * 技术要点:
 *  - 预分配固定大小缓冲区, 零动态分配
 *  - 原子 head (写指针) 和 tail (读指针) 索引
 *  - 生产者通过 head 写入, 消费者通过 tail 读取
 *  - acquire/release 内存序保证写入可见性
 *
 * 编译: g++ -std=c++20 -O2 -pthread 02_lockfree_queue_spmc.cpp -o spmc_queue
 */

#include <atomic>
#include <iostream>
#include <thread>
#include <vector>
#include <cassert>
#include <cstring>
#include <optional>
#include <chrono>
#include <mutex>

// ============================================================================
// SPMCQueue<T, Capacity> — 单生产者多消费者无锁环形队列
// Capacity 必须为 2 的幂, 以便使用位掩码代替取模运算
// ============================================================================
template <typename T, size_t Capacity>
class SPMCQueue {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity 必须为 2 的幂");
    static_assert(Capacity >= 2, "Capacity 至少为 2");

private:
    static constexpr size_t kMask = Capacity - 1;

    // 元素存储 (使用 aligned_storage 避免默认构造)
    using Storage = typename std::aligned_storage<sizeof(T), alignof(T)>::type;
    Storage buffer_[Capacity];

    // 缓存行对齐: 避免 head/tail 伪共享
    alignas(64) std::atomic<size_t> head_{0}; // 生产者写入位置
    alignas(64) std::atomic<size_t> tail_{0}; // 消费者读取位置

public:
    SPMCQueue() = default;

    ~SPMCQueue() {
        // 销毁所有未消费的元素
        size_t t = tail_.load(std::memory_order_relaxed);
        size_t h = head_.load(std::memory_order_relaxed);
        while (t != h) {
            reinterpret_cast<T*>(&buffer_[t & kMask])->~T();
            ++t;
        }
    }

    SPMCQueue(const SPMCQueue&) = delete;
    SPMCQueue& operator=(const SPMCQueue&) = delete;

    // -----------------------------------------------------------------------
    // try_push — 生产者尝试写入 (仅单生产者安全)
    // -----------------------------------------------------------------------
    bool try_push(const T& item) {
        size_t h = head_.load(std::memory_order_relaxed);
        size_t t = tail_.load(std::memory_order_acquire);

        // 队列满?
        if (h - t >= Capacity) {
            return false;
        }

        // 就地构造
        new (&buffer_[h & kMask]) T(item);

        // release 保证写入对其他线程可见
        head_.store(h + 1, std::memory_order_release);
        return true;
    }

    // -----------------------------------------------------------------------
    // try_push — 移动语义版本
    // -----------------------------------------------------------------------
    bool try_push(T&& item) {
        size_t h = head_.load(std::memory_order_relaxed);
        size_t t = tail_.load(std::memory_order_acquire);

        if (h - t >= Capacity) {
            return false;
        }

        new (&buffer_[h & kMask]) T(std::move(item));
        head_.store(h + 1, std::memory_order_release);
        return true;
    }

    // -----------------------------------------------------------------------
    // try_pop — 消费者尝试读取 (多消费者安全, 通过 CAS 竞争 tail)
    // -----------------------------------------------------------------------
    std::optional<T> try_pop() {
        size_t t = tail_.load(std::memory_order_relaxed);

        while (true) {
            size_t h = head_.load(std::memory_order_acquire);

            // 队列空?
            if (t == h) {
                return std::nullopt;
            }

            T value = std::move(*reinterpret_cast<T*>(&buffer_[t & kMask]));

            // CAS 竞争 tail (多消费者场景下关键)
            if (tail_.compare_exchange_weak(
                    t, t + 1,
                    std::memory_order_release,
                    std::memory_order_relaxed)) {
                // 成功获取, 销毁源对象
                reinterpret_cast<T*>(&buffer_[(t) & kMask])->~T();
                return value;
            }
            // CAS 失败, t 被更新为最新 tail, 继续尝试
        }
    }

    // -----------------------------------------------------------------------
    // 容量信息
    // -----------------------------------------------------------------------
    size_t size() const {
        size_t h = head_.load(std::memory_order_acquire);
        size_t t = tail_.load(std::memory_order_acquire);
        return h - t;
    }

    bool empty() const { return size() == 0; }
    bool full() const { return size() >= Capacity; }
    size_t capacity() const { return Capacity; }
};

// ============================================================================
// 并发正确性测试
// ============================================================================
void correctness_test() {
    std::cout << "=== SPMC 无锁队列正确性测试 ===\n";

    constexpr size_t kCapacity = 1024;
    constexpr int kItems = 1000000;
    SPMCQueue<int, kCapacity> queue;

    std::atomic<long long> producer_sum{0};
    std::atomic<long long> consumer_sum{0};
    std::atomic<int> produced{0};
    std::atomic<int> consumed{0};

    // 单生产者
    std::jthread producer([&]() {
        for (int i = 0; i < kItems; ++i) {
            while (!queue.try_push(i)) {
                std::this_thread::yield();
            }
            producer_sum.fetch_add(i, std::memory_order_relaxed);
            produced.fetch_add(1, std::memory_order_relaxed);
        }
    });

    // 多消费者 (4 个)
    constexpr int kConsumers = 4;
    std::vector<std::jthread> consumers;
    for (int t = 0; t < kConsumers; ++t) {
        consumers.emplace_back([&]() {
            while (consumed.load(std::memory_order_relaxed) < kItems) {
                auto item = queue.try_pop();
                if (item) {
                    consumer_sum.fetch_add(*item, std::memory_order_relaxed);
                    consumed.fetch_add(1, std::memory_order_relaxed);
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }

    producer.join();
    for (auto& c : consumers) c.join();

    std::cout << "  生产者总和: " << producer_sum.load() << "\n";
    std::cout << "  消费者总和: " << consumer_sum.load() << "\n";
    std::cout << "  已生产: " << produced.load() << "\n";
    std::cout << "  已消费: " << consumed.load() << "\n";

    if (producer_sum.load() == consumer_sum.load() &&
        produced.load() == kItems &&
        consumed.load() == kItems) {
        std::cout << "  结果: 通过!\n";
    } else {
        std::cerr << "  结果: 失败!\n";
    }
}

// ============================================================================
// 性能基准测试
// ============================================================================
void benchmark_test() {
    std::cout << "\n=== SPMC 无锁队列性能基准 ===\n";

    constexpr size_t kCapacity = 65536;
    constexpr int kItems = 5000000;
    SPMCQueue<int, kCapacity> queue;

    auto start = std::chrono::high_resolution_clock::now();

    std::jthread producer([&]() {
        for (int i = 0; i < kItems; ++i) {
            while (!queue.try_push(i)) {
                std::this_thread::yield();
            }
        }
    });

    std::vector<std::jthread> consumers;
    std::atomic<int> cons_total{0};
    for (int t = 0; t < 4; ++t) {
        consumers.emplace_back([&]() {
            while (cons_total.load(std::memory_order_relaxed) < kItems) {
                if (queue.try_pop()) {
                    cons_total.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    producer.join();
    for (auto& c : consumers) c.join();

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "  总项数: " << kItems << "\n";
    std::cout << "  容量: " << kCapacity << "\n";
    std::cout << "  耗时: " << elapsed << " ms\n";
    std::cout << "  吞吐量: " << (kItems * 1000.0 / elapsed) << " ops/s\n";
}

// ============================================================================
// main
// ============================================================================
int main() {
    correctness_test();
    benchmark_test();
    return 0;
}
