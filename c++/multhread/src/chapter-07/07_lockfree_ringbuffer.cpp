// 07_lockfree_ringbuffer.cpp — 无锁环形缓冲区 (MPMC Lock-Free Ring Buffer)
//
// Ring Buffer (环形缓冲区) 是无锁编程中最常用的数据结构之一:
//  - 固定容量，预分配内存 (无动态分配)
//  - 生产者和消费者操作不同的索引 (低争用)
//  - 可用于日志系统、事件队列、音频/视频流等
//
// 本实现: MPMC (多生产者-多消费者)，使用独立 head/tail 原子索引

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <syncstream>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// ================================================================
// MPMC Lock-Free Ring Buffer
// ================================================================
template <typename T, size_t Capacity>
class LockFreeRingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "Capacity must be power of 2");

public:
    LockFreeRingBuffer() {
        buffer_ = std::make_unique<Element[]>(Capacity);
    }

    // Producer: 尝试写入一个元素，成功返回 true
    bool try_push(const T& value) {
        size_t tail = tail_.load(std::memory_order_relaxed);

        while (true) {
            size_t head = head_.load(std::memory_order_acquire); // 消费位置
            size_t size = tail - head;

            // 满检查 (tail 和 head 可能被其他线程修改，需 CAS)
            if (size >= Capacity) return false;

            // 尝试锁定 tail 位置
            if (tail_.compare_exchange_weak(
                    tail, tail + 1,
                    std::memory_order_release,
                    std::memory_order_relaxed)) {
                // 写入数据
                size_t idx = tail & (Capacity - 1);
                buffer_[idx].data = value;
                buffer_[idx].ready.store(true, std::memory_order_release);
                return true;
            }
            // CAS 失败: tail 被其他 producer 抢了，重新读取
        }
    }

    // Consumer: 尝试读取一个元素，成功返回 true
    bool try_pop(T& value) {
        size_t head = head_.load(std::memory_order_relaxed);

        while (true) {
            size_t tail = tail_.load(std::memory_order_acquire); // 生产位置
            size_t size = tail - head;

            // 空检查
            if (size == 0) return false;

            // 尝试锁定 head 位置
            if (head_.compare_exchange_weak(
                    head, head + 1,
                    std::memory_order_release,
                    std::memory_order_relaxed)) {
                size_t idx = head & (Capacity - 1);
                // 等待数据就绪 (producer 可能还没写入完成)
                while (!buffer_[idx].ready.load(
                    std::memory_order_acquire)) {
                    // spin wait — rarely needed, producer is fast
                }
                value = buffer_[idx].data;
                buffer_[idx].ready.store(false, std::memory_order_relaxed);
                return true;
            }
            // CAS 失败: head 被其他 consumer 抢了
        }
    }

    size_t approximate_size() const {
        size_t tail = tail_.load(std::memory_order_relaxed);
        size_t head = head_.load(std::memory_order_relaxed);
        return tail - head;
    }

private:
    struct alignas(64) Element {
        T data{};
        std::atomic<bool> ready{false};
        char padding[64 - sizeof(T) - sizeof(std::atomic<bool>)];
    };

    std::unique_ptr<Element[]> buffer_;
    char pad1_[64 - sizeof(std::atomic<size_t>)]{};
    std::atomic<size_t> head_{0}; // consumer index
    char pad2_[64 - sizeof(std::atomic<size_t>)]{};
    std::atomic<size_t> tail_{0}; // producer index

    // head_ 和 tail_ 用 padding 分隔到不同 cache line，避免 producer-consumer 伪共享
    static_assert(sizeof(pad1_) >= 56, "pad1 insufficient");
    static_assert(sizeof(pad2_) >= 56, "pad2 insufficient");
};

// ================================================================
// 性能测试
// ================================================================
void benchmark_ringbuffer() {
    std::cout << "=== MPMC Lock-Free Ring Buffer 性能测试 ===\n";

    constexpr size_t kCapacity = 1024;
    LockFreeRingBuffer<int, kCapacity> rb;

    const int kProducers = 4;
    const int kConsumers = 4;
    const long long kOpsPerThread = 500'000;

    std::atomic<long long> produced{0};
    std::atomic<long long> consumed{0};
    std::atomic<long long> sum_produced{0};
    std::atomic<long long> sum_consumed{0};
    std::atomic<bool> stop{false};

    auto start = std::chrono::high_resolution_clock::now();

    // Producers
    std::vector<std::jthread> producers;
    for (int p = 0; p < kProducers; ++p) {
        producers.emplace_back([&, p]() {
            for (long long i = 0; i < kOpsPerThread; ++i) {
                int value = static_cast<int>(p * kOpsPerThread + i);
                while (!rb.try_push(value)) {
                    // buffer full, spin briefly
                    std::this_thread::yield();
                }
                produced.fetch_add(1);
                sum_produced.fetch_add(value);
            }
        });
    }

    // Consumers
    std::vector<std::jthread> consumers;
    for (int c = 0; c < kConsumers; ++c) {
        consumers.emplace_back([&]() {
            while (!stop.load(std::memory_order_relaxed)) {
                int value;
                if (rb.try_pop(value)) {
                    consumed.fetch_add(1);
                    sum_consumed.fetch_add(value);
                } else {
                    // buffer empty, check if producers done
                    if (produced.load() >=
                        kProducers * kOpsPerThread) {
                        stop.store(true);
                        break;
                    }
                    std::this_thread::yield();
                }
            }
            // drain remaining
            int value;
            while (rb.try_pop(value)) {
                consumed.fetch_add(1);
                sum_consumed.fetch_add(value);
            }
        });
    }

    for (auto& p : producers) p.join();
    stop.store(true);
    for (auto& c : consumers) c.join();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);

    std::cout << "  Producers: " << kProducers
              << " | Consumers: " << kConsumers
              << " | Capacity: " << kCapacity << "\n";
    std::cout << "  生产: " << produced.load()
              << " 次 (期望 " << kProducers * kOpsPerThread << ")\n";
    std::cout << "  消费: " << consumed.load()
              << " 次\n";
    std::cout << "  耗时: " << elapsed.count() << " ms\n";
    std::cout << "  吞吐量: " << std::fixed << std::setprecision(1)
              << produced.load() * 1000.0 / elapsed.count()
              << " ops/s\n";
    std::cout << "  正确性: "
              << ((sum_produced == sum_consumed &&
                   produced == consumed) ? "PASS" : "FAIL")
              << "\n";
}

// ================================================================
// Ring Buffer 设计要点说明
// ================================================================
void design_notes() {
    std::cout << "\n=== Ring Buffer 设计要点 ===\n";
    std::cout << "  1. Capacity 必须为 2 的幂: 用位与 (&) 代替取模 (%)\n";
    std::cout << "  2. head 和 tail 分离到不同 cache line: 消除伪共享\n";
    std::cout << "  3. head 只被 consumer 修改，tail 只被 producer 修改\n";
    std::cout << "  4. 数据槽 ready flag: 确保数据完全写入后再被读取\n";
    std::cout << "  5. 无锁的关键: head/tail 分别用 CAS 保护\n";
    std::cout << "  6. 无需 ABA 保护: 索引单调递增 (size_t 在有限时间内不会回绕)\n";
    std::cout << "  7. 适用: 日志系统、事件循环、音视频流、网络数据包\n";
}

int main() {
    benchmark_ringbuffer();
    design_notes();
    return 0;
}
