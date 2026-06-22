// 10_timed_mutex.cpp — std::timed_mutex / std::recursive_timed_mutex
//
// timed_mutex 在普通 mutex 基础上增加了超时机制:
//  - try_lock_for(duration):   等待一段时间
//  - try_lock_until(timepoint): 等到指定时间点
//
// 解决了普通 mutex 的"无限阻塞"问题，是构建健壮并发系统的关键。

#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <syncstream>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// ===== 1. timed_mutex 基础 =====
void demo_timed_mutex_basic() {
    std::cout << "=== 1. timed_mutex 基础 ===\n";

    std::timed_mutex tmtx;
    auto start = std::chrono::steady_clock::now();

    // 首次加锁: 成功
    {
        std::unique_lock lock(tmtx);
        std::cout << "  主线程获得锁\n";

        // 尝试在另一个线程中超时获取
        std::jthread worker([&]() {
            auto t0 = std::chrono::steady_clock::now();

            // try_lock_for: 等待最多 200ms
            if (tmtx.try_lock_for(200ms)) {
                std::cout << "  Worker: 获得锁 (意外!)\n";
                tmtx.unlock();
            } else {
                auto waited = std::chrono::duration_cast<
                    std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - t0);
                std::cout << "  Worker: 超时! 等待了 "
                          << waited.count() << "ms\n";
            }
        });
    }

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    std::cout << "  总耗时: " << elapsed.count() << "ms\n";
}

// ===== 2. try_lock_until: 等到某个时间点 =====
void demo_try_lock_until() {
    std::cout << "\n=== 2. try_lock_until ===\n";

    std::timed_mutex tmtx;
    std::unique_lock lock(tmtx); // 主线程持有锁

    auto deadline = std::chrono::steady_clock::now() + 300ms;
    std::cout << "  截止时间: 300ms 后\n";

    std::jthread worker([&]() {
        // 等到 deadline 为止
        if (tmtx.try_lock_until(deadline)) {
            std::cout << "  Worker: 在截止前获得锁\n";
            tmtx.unlock();
        } else {
            std::cout << "  Worker: 等到截止时间未获得锁\n";
        }
    });

    // 主线程 500ms 后释放锁 (超过 deadline)
    std::this_thread::sleep_for(500ms);
    lock.unlock();
    std::cout << "  主线程 500ms 后释放锁\n";

    worker.join();
}

// ===== 3. 实际场景: 带超时的资源获取 =====
class TimeoutProtectedResource {
public:
    bool try_acquire(std::chrono::milliseconds timeout) {
        if (mtx_.try_lock_for(timeout)) {
            // 注意: try_lock_for 成功后需要手动 unlock 或使用
            // unique_lock + defer_lock
            return true;
        }
        return false;
    }

    void release() { mtx_.unlock(); }

    void do_work(int id) {
        std::osyncstream(std::cout)
            << "    资源被线程 " << id << " 使用\n";
        std::this_thread::sleep_for(50ms); // 模拟处理
    }

private:
    std::timed_mutex mtx_;
};

void demo_resource_with_timeout() {
    std::cout << "\n=== 3. 带超时的资源获取 ===\n";

    TimeoutProtectedResource resource;
    std::atomic<int> success{0};
    std::atomic<int> timeout{0};

    // 线程 0 先获取资源并持有较长时间
    std::jthread long_holder([&]() {
        if (resource.try_acquire(200ms)) {
            resource.do_work(0);
            std::this_thread::sleep_for(300ms); // 长时间持有
            resource.release();
            success.fetch_add(1);
        }
    });

    // 其他线程尝试获取
    std::vector<std::jthread> contenders;
    for (int i = 1; i <= 4; ++i) {
        contenders.emplace_back([&, i]() {
            std::this_thread::sleep_for(50ms * i); // 错开时间
            if (resource.try_acquire(100ms)) {
                resource.do_work(i);
                resource.release();
                success.fetch_add(1);
            } else {
                std::osyncstream(std::cout)
                    << "    线程 " << i << " 超时放弃\n";
                timeout.fetch_add(1);
            }
        });
    }

    long_holder.join();
    for (auto& t : contenders) t.join();

    std::cout << "  成功: " << success.load()
              << " | 超时: " << timeout.load() << "\n";
}

// ===== 4. recursive_timed_mutex =====
void demo_recursive_timed_mutex() {
    std::cout << "\n=== 4. recursive_timed_mutex ===\n";

    std::recursive_timed_mutex rtmtx;

    auto nested_func = [&](int depth, auto&& self) -> void {
        if (rtmtx.try_lock_for(10ms)) {
            std::cout << "    深度 " << depth << " 获得锁\n";
            if (depth > 0) {
                self(depth - 1, self);
            }
            rtmtx.unlock();
        } else {
            std::cout << "    深度 " << depth << " 超时\n";
        }
    };

    nested_func(2, nested_func);
    std::cout << "  recursive_timed_mutex = recursive + timed\n";
}

// ===== 5. timed_mutex 的设计决策 =====
void demo_design_decisions() {
    std::cout << "\n=== 5. 设计决策 ===\n";

    std::cout << "  何时使用 timed_mutex:\n";
    std::cout << "    ✅ 需要超时放弃而非无限等待\n";
    std::cout << "    ✅ 需要 try_lock_for 实现 try_get 语义\n";
    std::cout << "    ✅ 实时系统: 不能接受无限阻塞\n";
    std::cout << "    ✅ 可降级服务: 获取不到资源就走 fallback 逻辑\n\n";

    std::cout << "  何时使用普通 mutex:\n";
    std::cout << "    ✅ 临界区极短 (微秒级)\n";
    std::cout << "    ✅ 必须成功 (无法降级)\n";
    std::cout << "    ✅ 对性能极度敏感 (timed_mutex 有轻微开销)\n\n";

    std::cout << "  mutex 类型对比:\n";
    std::cout << "  ┌───────────────────────┬──────┬──────────┬──────────┐\n";
    std::cout << "  │ 类型                   │ 超时  │ 递归     │ 共享     │\n";
    std::cout << "  ├───────────────────────┼──────┼──────────┼──────────┤\n";
    std::cout << "  │ mutex                  │ 否   │ 否       │ 否       │\n";
    std::cout << "  │ recursive_mutex        │ 否   │ 是       │ 否       │\n";
    std::cout << "  │ timed_mutex            │ 是   │ 否       │ 否       │\n";
    std::cout << "  │ recursive_timed_mutex  │ 是   │ 是       │ 否       │\n";
    std::cout << "  │ shared_mutex           │ 否*  │ 否       │ 是       │\n";
    std::cout << "  │ shared_timed_mutex     │ 是*  │ 否       │ 是       │\n";
    std::cout << "  └───────────────────────┴──────┴──────────┴──────────┘\n";
    std::cout << "  * shared_mutex 也有 try_lock_shared_for/until 变体\n";
}

int main() {
    demo_timed_mutex_basic();
    demo_try_lock_until();
    demo_resource_with_timeout();
    demo_recursive_timed_mutex();
    demo_design_decisions();

    std::cout << "\ntimed_mutex 是构建优雅降级并发系统的基石。\n";
    return 0;
}
