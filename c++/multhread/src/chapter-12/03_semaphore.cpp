// 03_semaphore.cpp — C++20 counting_semaphore / binary_semaphore
// 演示: 限制并发数、生产者-消费者、连接池模拟

#include <chrono>
#include <iostream>
#include <semaphore>
#include <syncstream>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// ===== 1. binary_semaphore: 互斥锁替代方案 =====
void demo_binary_semaphore() {
    std::cout << "=== 1. binary_semaphore 互斥 ===\n";
    std::binary_semaphore sem{1};
    int shared_counter = 0;
    const int kThreads = 4;
    const int kIters = 10000;

    std::vector<std::jthread> threads;
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < kIters; ++j) {
                sem.acquire();
                ++shared_counter;
                sem.release();
            }
        });
    }
    threads.clear();

    std::cout << "  shared_counter = " << shared_counter
              << " (期望 " << kThreads * kIters << ")\n";
}

// ===== 2. counting_semaphore: 限制并发访问数 =====
void demo_counting_semaphore() {
    std::cout << "\n=== 2. counting_semaphore 限流 ===\n";

    // 最多 3 个线程同时访问资源
    std::counting_semaphore<3> slots{3};
    std::atomic<int> active{0};
    std::atomic<int> max_active{0};

    auto worker = [&](int id) {
        slots.acquire();
        int current = active.fetch_add(1) + 1;
        // 更新最大并发数
        int expected = max_active.load();
        while (current > expected &&
               !max_active.compare_exchange_weak(expected, current)) {
        }

        std::osyncstream(std::cout)
            << "    线程 " << id << " 进入 (活跃: " << current << ")\n";
        std::this_thread::sleep_for(50ms);

        active.fetch_sub(1);
        std::osyncstream(std::cout)
            << "    线程 " << id << " 退出\n";
        slots.release();
    };

    std::vector<std::jthread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back(worker, i);
    }
    threads.clear();

    std::cout << "  最大并发数: " << max_active.load()
              << " (期望 <= 3)\n";
}

// ===== 3. 连接池模拟 =====
class ConnectionPool {
public:
    ConnectionPool(int size)
        : sem_(size), size_(size) {}

    void acquire_connection() {
        sem_.acquire();
        active_.fetch_add(1);
    }

    void release_connection() {
        active_.fetch_sub(1);
        sem_.release();
    }

    int active() const { return active_.load(); }

private:
    std::counting_semaphore<> sem_;
    int size_;
    std::atomic<int> active_{0};
};

void demo_connection_pool() {
    std::cout << "\n=== 3. 连接池模拟 (最多 4 连接) ===\n";

    ConnectionPool pool{4};
    std::vector<std::jthread> clients;

    for (int i = 0; i < 12; ++i) {
        clients.emplace_back([&, i]() {
            pool.acquire_connection();
            std::osyncstream(std::cout)
                << "    客户端 " << i << " 获得连接 (活跃: "
                << pool.active() << ")\n";
            std::this_thread::sleep_for(30ms);
            pool.release_connection();
        });
    }
    clients.clear();
    std::cout << "  所有客户端完成\n";
}

// ===== 4. 生产者-消费者 (semaphore 版本) =====
void demo_producer_consumer() {
    std::cout << "\n=== 4. 生产者-消费者 (semaphore) ===\n";

    constexpr int kBufSize = 5;
    std::counting_semaphore<kBufSize> empty{kBufSize}; // 空槽位数
    std::counting_semaphore<kBufSize> full{0};          // 满槽位数
    std::mutex mtx;
    std::vector<int> buffer;
    buffer.reserve(kBufSize);

    std::atomic<long long> sum_produced{0};
    std::atomic<long long> sum_consumed{0};

    std::jthread producer([&](std::stop_token stoken) {
        for (int i = 0; i < 50 && !stoken.stop_requested(); ++i) {
            empty.acquire();
            {
                std::lock_guard lock(mtx);
                buffer.push_back(i);
                sum_produced.fetch_add(i);
            }
            full.release();
            std::this_thread::sleep_for(1ms);
        }
    });

    std::jthread consumer([&](std::stop_token stoken) {
        for (int i = 0; i < 50 && !stoken.stop_requested(); ++i) {
            full.acquire();
            int val;
            {
                std::lock_guard lock(mtx);
                val = buffer.back();
                buffer.pop_back();
                sum_consumed.fetch_add(val);
            }
            empty.release();
            std::this_thread::sleep_for(2ms);
        }
    });

    producer.join();
    consumer.join();

    std::cout << "  sum_produced = " << sum_produced.load()
              << " | sum_consumed = " << sum_consumed.load()
              << " | 匹配: " << (sum_produced == sum_consumed ? "OK" : "FAIL")
              << "\n";
}

int main() {
    demo_binary_semaphore();
    demo_counting_semaphore();
    demo_connection_pool();
    demo_producer_consumer();

    std::cout << "\nsemaphore 比 mutex+cv 更简洁，适合计数型同步场景。\n";
    return 0;
}
