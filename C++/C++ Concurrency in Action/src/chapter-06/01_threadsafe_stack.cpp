// 01_threadsafe_stack.cpp - 线程安全栈 ThreadSafeStack<T>
// 关键设计：消除 empty() + top() + pop() 的竞态条件
// 方案：合并为一粒操作的 pop()，返回 std::shared_ptr 或通过参数返回

#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <stack>
#include <thread>
#include <vector>

// ===== 方案 A：pop() 返回 shared_ptr（无异常安全问题） =====
template <typename T>
class ThreadSafeStack {
public:
    ThreadSafeStack() = default;

    // 禁止拷贝（mutex 不可拷贝）
    ThreadSafeStack(const ThreadSafeStack&) = delete;
    ThreadSafeStack& operator=(const ThreadSafeStack&) = delete;

    void push(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        data_.push(std::make_shared<T>(std::move(value)));
    }

    // pop() 返回 shared_ptr：一次性完成 pop + top
    // 即使栈为空也不会抛异常，而是返回 nullptr
    std::shared_ptr<T> pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (data_.empty()) {
            return nullptr; // 或抛出异常，视语义而定
        }
        std::shared_ptr<T> result = data_.top();
        data_.pop();
        return result;
    }

    // 方案 B：pop 时通过引用返回（可能抛异常，但常用）
    void pop(T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (data_.empty()) {
            throw std::out_of_range("ThreadSafeStack: pop from empty stack");
        }
        value = *data_.top();
        data_.pop();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_.size();
    }

private:
    mutable std::mutex                            mutex_;
    std::stack<std::shared_ptr<T>>                data_;
};

// ===== 测试 =====
int main() {
    ThreadSafeStack<int> stack;

    std::cout << "=== ThreadSafeStack ===\n";

    // 生产者：压入 0..19
    const int kNumItems    = 20;
    const int kNumProducers = 2;
    const int kNumConsumers = 4;

    {
        std::vector<std::jthread> producers;
        for (int p = 0; p < kNumProducers; ++p) {
            producers.emplace_back([&, p]() {
                int start = p * (kNumItems / kNumProducers);
                int end   = start + (kNumItems / kNumProducers);
                for (int i = start; i < end; ++i) {
                    stack.push(i);
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
            });
        }

        std::vector<std::jthread> consumers;
        std::atomic<int>          total_popped{0};

        for (int c = 0; c < kNumConsumers; ++c) {
            consumers.emplace_back([&]() {
                int popped = 0;
                while (popped < kNumItems / kNumConsumers) {
                    auto val = stack.pop();
                    if (val) {
                        std::cout << "[C] pop: " << *val << "\n";
                        ++popped;
                        total_popped.fetch_add(1);
                    } else {
                        std::this_thread::yield();
                    }
                }
            });
        }

        // jthread 自动 join
    }

    std::cout << "[Main] 栈最终大小: " << stack.size() << "\n";
    return 0;
}
