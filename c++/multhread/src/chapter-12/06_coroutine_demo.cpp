// 06_coroutine_demo.cpp — C++20 协程基础
// 演示: 简单协程、生成器模式、异步任务框架
// 注意: C++20 只提供协程基础设施，标准库类型需自实现

#include <coroutine>
#include <exception>
#include <iostream>
#include <optional>
#include <thread>
#include <utility>

// ================================================================
// 1. 最简单的协程——只打印 Hello World
// ================================================================
struct SimpleTask {
    struct promise_type {
        SimpleTask get_return_object() { return {}; }
        std::suspend_never initial_suspend() { return {}; }
        std::suspend_never final_suspend() noexcept { return {}; }
        void return_void() {}
        void unhandled_exception() { std::terminate(); }
    };
};

SimpleTask hello_coroutine() {
    std::cout << "  Hello, ";
    co_await std::suspend_always{}; // 挂起点
    std::cout << "C++20 Coroutine!\n";
    co_return;
}

void demo_simple_coroutine() {
    std::cout << "=== 1. 简单协程 ===\n";
    auto task = hello_coroutine();
    std::cout << "  (协程在 first co_await 处挂起，已打印 \"Hello, \")\n";
    std::cout << "  (协程在 co_return 后自动销毁)\n\n";
}

// ================================================================
// 2. Generator — 惰性生成序列
// ================================================================
template <typename T>
struct Generator {
    struct promise_type {
        T current_value;

        Generator get_return_object() {
            return Generator{
                std::coroutine_handle<promise_type>::from_promise(*this)
            };
        }
        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        std::suspend_always yield_value(T value) {
            current_value = value;
            return {};
        }
        void return_void() {}
        void unhandled_exception() { std::terminate(); }
    };

    using handle_type = std::coroutine_handle<promise_type>;
    handle_type handle;

    explicit Generator(handle_type h) : handle(h) {}
    ~Generator() { if (handle) handle.destroy(); }
    Generator(const Generator&) = delete;
    Generator& operator=(const Generator&) = delete;
    Generator(Generator&& other) noexcept
        : handle(std::exchange(other.handle, nullptr)) {}
    Generator& operator=(Generator&& other) noexcept {
        if (this != &other) {
            if (handle) handle.destroy();
            handle = std::exchange(other.handle, nullptr);
        }
        return *this;
    }

    // 移动到下一个值
    bool next() {
        if (!handle || handle.done()) return false;
        handle.resume();
        return !handle.done();
    }

    T value() const { return handle.promise().current_value; }

    // 迭代器支持
    struct iterator {
        handle_type handle;
        bool operator!=(std::default_sentinel_t) const {
            return handle && !handle.done();
        }
        iterator& operator++() {
            handle.resume();
            return *this;
        }
        T operator*() const { return handle.promise().current_value; }
    };

    iterator begin() {
        if (handle) handle.resume();
        return {handle};
    }
    std::default_sentinel_t end() { return {}; }
};

// 斐波那契生成器
Generator<int> fibonacci(int n) {
    int a = 0, b = 1;
    for (int i = 0; i < n; ++i) {
        co_yield a;
        int next = a + b;
        a = b;
        b = next;
    }
}

void demo_generator() {
    std::cout << "=== 2. 生成器: 斐波那契 ===\n";
    std::cout << "  前 10 个斐波那契数: ";
    for (int v : fibonacci(10)) {
        std::cout << v << " ";
    }
    std::cout << "\n\n";
}

// ================================================================
// 3. 可等待的异步任务
// ================================================================
template <typename T>
struct Task {
    struct promise_type {
        T result;
        std::exception_ptr exception;

        Task get_return_object() {
            return Task{
                std::coroutine_handle<promise_type>::from_promise(*this)
            };
        }
        std::suspend_never initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void return_value(T value) { result = std::move(value); }
        void unhandled_exception() { exception = std::current_exception(); }
    };

    using handle_type = std::coroutine_handle<promise_type>;
    handle_type handle;

    explicit Task(handle_type h) : handle(h) {}
    ~Task() { if (handle) handle.destroy(); }
    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;
    Task(Task&& other) noexcept
        : handle(std::exchange(other.handle, nullptr)) {}
    Task& operator=(Task&& other) noexcept {
        if (this != &other) {
            if (handle) handle.destroy();
            handle = std::exchange(other.handle, nullptr);
        }
        return *this;
    }

    bool is_ready() const { return handle.done(); }

    T get() {
        if (!handle.done()) handle.resume();
        auto& p = handle.promise();
        if (p.exception) std::rethrow_exception(p.exception);
        return std::move(p.result);
    }
};

// 模拟异步计算
Task<int> async_compute(int base, int mul) {
    // 模拟耗时操作
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    co_return base * mul;
}

Task<int> combined_compute() {
    // 顺序等待两个异步任务
    auto task1 = async_compute(10, 2); // 返回 20
    int r1 = task1.get();

    auto task2 = async_compute(r1, 3); // 返回 60
    int r2 = task2.get();

    co_return r2;
}

void demo_async_task() {
    std::cout << "=== 3. 异步任务 ===\n";
    auto task = combined_compute();
    int result = task.get();
    std::cout << "  组合计算: 10*2*3 = " << result << "\n\n";
}

// ================================================================
// main
// ================================================================
int main() {
    std::cout << "C++20 协程基础演示\n\n";

    demo_simple_coroutine();
    demo_generator();
    demo_async_task();

    std::cout << "注意: C++20 只提供协程的语言基础设施。\n";
    std::cout << "生产环境推荐使用 cppcoro 或 boost::asio 的协程支持。\n";
    return 0;
}
