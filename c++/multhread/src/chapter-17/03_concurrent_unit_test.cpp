// 03_concurrent_unit_test.cpp — 并发单元测试框架 (无外部依赖)
// 演示: 纯头文件测试宏、断言、测试套件注册

#include <atomic>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// ===== 微型测试框架 =====
class Minitest {
public:
    using TestFunc = std::function<void()>;

    struct TestCase {
        std::string name;
        TestFunc func;
    };

    static Minitest& instance() {
        static Minitest inst;
        return inst;
    }

    void register_test(const std::string& name, TestFunc func) {
        tests_.push_back({name, std::move(func)});
    }

    int run_all() {
        int passed = 0, failed = 0;
        std::cout << "Running " << tests_.size() << " tests...\n\n";

        for (const auto& test : tests_) {
            std::cout << "  [" << test.name << "] ";
            try {
                test.func();
                std::cout << "PASSED\n";
                ++passed;
            } catch (const std::exception& e) {
                std::cout << "FAILED: " << e.what() << "\n";
                ++failed;
            } catch (...) {
                std::cout << "FAILED: unknown exception\n";
                ++failed;
            }
        }

        std::cout << "\n" << std::string(50, '=') << "\n";
        std::cout << "Results: " << passed << " passed, "
                  << failed << " failed, "
                  << tests_.size() << " total\n";
        return failed > 0 ? 1 : 0;
    }

private:
    std::vector<TestCase> tests_;
};

// 断言宏
void expect_true(bool condition, const char* expr, const char* file, int line) {
    if (!condition) {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) +
            ": EXPECT_TRUE(" + expr + ") failed");
    }
}

void expect_eq_int(int expected, int actual,
                    const char* file, int line) {
    if (expected != actual) {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) +
            ": EXPECT_EQ(expected=" + std::to_string(expected) +
            ", actual=" + std::to_string(actual) + ")");
    }
}

#define EXPECT_TRUE(cond) \
    expect_true((cond), #cond, __FILE__, __LINE__)
#define EXPECT_EQ(expected, actual) \
    expect_eq_int((expected), (actual), __FILE__, __LINE__)

// 测试注册宏
#define TEST(name) \
    static void test_##name(); \
    static bool registered_##name = []() { \
        Minitest::instance().register_test(#name, test_##name); \
        return true; \
    }(); \
    static void test_##name()

// ===== 测试用例 =====

TEST(atomic_increment) {
    std::atomic<int> counter{0};
    const int kThreads = 4;
    const int kIters = 1000;

    std::vector<std::jthread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < kIters; ++i) {
                counter.fetch_add(1);
            }
        });
    }
    threads.clear();

    EXPECT_EQ(kThreads * kIters, counter.load());
}

TEST(mutex_protected_counter) {
    std::mutex mtx;
    int counter = 0;
    const int kThreads = 4;
    const int kIters = 1000;

    std::vector<std::jthread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < kIters; ++i) {
                std::lock_guard lock(mtx);
                ++counter;
            }
        });
    }
    threads.clear();

    EXPECT_EQ(kThreads * kIters, counter);
}

TEST(spinlock_exclusion) {
    std::atomic_flag spinlock = ATOMIC_FLAG_INIT;
    std::atomic<int> active_count{0};
    std::atomic<bool> violation{false};
    const int kThreads = 4;
    const int kIters = 500;

    auto critical_section = [&]() -> bool {
        // test_and_set: if was false, I got the lock
        while (spinlock.test_and_set(std::memory_order_acquire)) {
            // spin
        }
        // Critical section
        int count = active_count.fetch_add(1) + 1;
        if (count > 1) {
            violation.store(true); // 多个线程同时在临界区 → 互斥失败
        }
        active_count.fetch_sub(1);
        spinlock.clear(std::memory_order_release);
        return !violation.load();
    };

    std::vector<std::jthread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < kIters; ++i) {
                critical_section();
            }
        });
    }
    threads.clear();

    EXPECT_TRUE(!violation.load());
}

TEST(stop_flag) {
    std::atomic<bool> stop{false};
    std::atomic<int> count{0};

    std::jthread worker([&](std::stop_token stoken) {
        while (!stop.load() && !stoken.stop_requested()) {
            count.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop.store(true);
    worker.join();

    EXPECT_TRUE(count.load() > 0);
}

TEST(deadlock_detection_timeout) {
    // 测试: 确保操作在超时内完成 (不会死锁)
    std::mutex m1, m2;
    std::atomic<bool> done{false};

    std::jthread t([&]() {
        std::scoped_lock lock(m1, m2); // scoped_lock 同时获取两个锁
        done.store(true);
    });

    // 等待完成 (最多 1 秒)
    auto deadline = std::chrono::steady_clock::now() + 1s;
    while (!done.load() &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(1ms);
    }

    EXPECT_TRUE(done.load());
}

int main() {
    return Minitest::instance().run_all();
}
