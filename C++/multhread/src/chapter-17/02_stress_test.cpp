// 02_stress_test.cpp — 并发压力测试框架
// 演示: 持续高并发操作、随机负载、正确性不变式验证

#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <mutex>
#include <random>
#include <syncstream>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// ===== 被测组件: 简单的线程安全栈 =====
template <typename T>
class SimpleThreadSafeStack {
public:
    void push(T value) {
        std::lock_guard lock(mtx_);
        data_.push_back(std::move(value));
        size_.fetch_add(1);
    }

    bool try_pop(T& value) {
        std::lock_guard lock(mtx_);
        if (data_.empty()) return false;
        value = std::move(data_.back());
        data_.pop_back();
        size_.fetch_sub(1);
        return true;
    }

    size_t size() const { return size_.load(); }

private:
    std::mutex mtx_;
    std::vector<T> data_;
    std::atomic<size_t> size_{0};
};

// ===== 不变式验证 =====
template <typename T>
bool verify_invariant(SimpleThreadSafeStack<T>& stack,
                       std::atomic<long long>& pushed,
                       std::atomic<long long>& popped) {
    long long p = pushed.load();
    long long q = popped.load();
    size_t s = stack.size();

    // 不变式: pushed = popped + size
    bool ok = (p == q + static_cast<long long>(s));

    if (!ok) {
        std::osyncstream(std::cerr)
            << "INVARIANT VIOLATION: pushed=" << p
            << ", popped=" << q << ", size=" << s
            << " (expected " << q + s << ")\n";
    }
    return ok;
}

// ===== 压力测试执行 =====
template <typename StackType>
bool run_stress_test(std::chrono::seconds duration,
                      int num_threads) {
    StackType stack;
    std::atomic<long long> pushed{0};
    std::atomic<long long> popped{0};
    std::atomic<bool> stop{false};
    std::atomic<bool> invariant_violated{false};

    std::cout << "  启动 " << num_threads << " 个线程，运行 "
              << duration.count() << " 秒...\n";

    auto start = std::chrono::steady_clock::now();

    std::vector<std::jthread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            std::mt19937 rng(t + 42);
            std::uniform_int_distribution<int> dist(0, 99);

            while (!stop.load(std::memory_order_relaxed)) {
                if (dist(rng) < 50) {
                    // push
                    int value = dist(rng);
                    stack.push(value);
                    pushed.fetch_add(1);
                } else {
                    // pop
                    int value;
                    if (stack.try_pop(value)) {
                        popped.fetch_add(1);
                    }
                }
            }
        });
    }

    // 主线程: 定期检查不变式
    auto next_check = std::chrono::steady_clock::now() + 500ms;
    int checks = 0;
    int violations = 0;

    while (std::chrono::steady_clock::now() - start < duration) {
        std::this_thread::sleep_until(next_check);
        next_check += 500ms;

        if (!verify_invariant(stack, pushed, popped)) {
            ++violations;
            invariant_violated.store(true);
        }
        ++checks;
    }

    stop.store(true);
    for (auto& t : threads) t.join();

    // 清空栈
    {
        int value;
        while (stack.try_pop(value)) {
            popped.fetch_add(1);
        }
    }

    // 最终验证
    bool final_ok = verify_invariant(stack, pushed, popped);

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);

    std::cout << "  结果:\n";
    std::cout << "    总操作: push=" << pushed.load()
              << ", pop=" << popped.load() << "\n";
    std::cout << "    不变式检查: " << checks << " 次, "
              << violations << " 次违规\n";
    std::cout << "    最终验证: " << (final_ok ? "PASS" : "FAIL") << "\n";
    std::cout << "    耗时: " << elapsed.count() << " ms\n\n";

    return final_ok && violations == 0;
}

// ===== 压力测试入口 =====
int main() {
    bool all_pass = true;

    // 短测试(快速验证)
    all_pass &= run_stress_test<SimpleThreadSafeStack<int>>(2s, 4);

    // 长测试(充分覆盖时序)
    all_pass &= run_stress_test<SimpleThreadSafeStack<int>>(5s, 8);

    std::cout << "============================\n";
    std::cout << "所有压力测试: " << (all_pass ? "PASS" : "FAIL") << "\n";
    std::cout << "============================\n";

    return all_pass ? 0 : 1;
}
