// 08_threadsafe_stack.cpp
// 知识点: 线程安全栈 - 接口设计与竞争条件
// 演示: 实现线程安全栈，展示接口层面的竞争条件问题
// 对应书中 3.14 节

#include <chrono>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stack>
#include <string>
#include <thread>
#include <vector>

// =============================================================================
// ThreadSafeStack v1: 基本实现
//
// 问题: top() 和 pop() 之间存在竞争条件
// 场景:
//   线程A: if (!stack.empty()) { value = stack.top(); }  ← 线程B在此时pop
//   线程B: stack.pop();
//   线程A: stack.pop();  ← 弹出了错误的元素!
// =============================================================================
template <typename T>
class ThreadSafeStackV1 {
public:
    void push(T value) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stack.push(std::move(value));
    }

    // 有问题的接口: top() 返回引用，调用者和pop()之间存在窗口
    [[nodiscard]] T& top() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_stack.top();  // 返回引用 → 锁释放 → 引用悬挂!
    }

    [[nodiscard]] const T& top() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_stack.top();
    }

    [[nodiscard]] bool empty() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_stack.empty();
    }

    [[nodiscard]] size_t size() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_stack.size();
    }

    void pop() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_stack.empty()) {
            m_stack.pop();
        }
    }

    // 问题: 即使这样组合，top() 和 pop() 也不是原子的
    // 正确做法: 合并 top() 和 pop()

private:
    std::stack<T>   m_stack;
    mutable std::mutex m_mutex;
};

// =============================================================================
// ThreadSafeStack v2: 工业级实现
//
// 改进:
//   1. 合并 top() 和 pop() 为原子操作
//   2. 使用 shared_ptr 避免拷贝异常
//   3. 提供两种pop方式: 返回值和引用参数
// =============================================================================
template <typename T>
class ThreadSafeStack {
public:
    ThreadSafeStack() = default;

    // 拷贝整个栈 (在锁保护下)
    ThreadSafeStack(const ThreadSafeStack& other) {
        std::lock_guard<std::mutex> lock(other.m_mutex);
        m_stack = other.m_stack;  // 内部 stack 的拷贝
    }

    ThreadSafeStack& operator=(const ThreadSafeStack&) = delete;

    // 压入元素
    void push(T value) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stack.push(std::move(value));
    }

    // pop 方式1: 通过 shared_ptr 返回 (推荐)
    // 返回 nullptr 表示栈为空
    [[nodiscard]] std::shared_ptr<T> pop() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_stack.empty()) {
            return std::shared_ptr<T>();  // 空指针
        }
        // 注意: 先构造 shared_ptr，再 pop
        // 如果 T 的拷贝构造抛异常，栈状态不变
        auto result = std::make_shared<T>(std::move(m_stack.top()));
        m_stack.pop();
        return result;
    }

    // pop 方式2: 通过引用参数接收 (避免动态内存分配)
    // 返回 true 表示成功
    bool pop(T& value) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_stack.empty()) {
            return false;
        }
        value = std::move(m_stack.top());
        m_stack.pop();
        return true;
    }

    [[nodiscard]] bool empty() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_stack.empty();
    }

    [[nodiscard]] size_t size() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_stack.size();
    }

    // 非破坏性查看栈顶 (谨慎使用)
    [[nodiscard]] std::shared_ptr<T> peek() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_stack.empty()) {
            return std::shared_ptr<T>();
        }
        return std::make_shared<T>(m_stack.top());
    }

private:
    std::stack<T>       m_stack;
    mutable std::mutex  m_mutex;
};

// =============================================================================
// 生产者-消费者 演示
// =============================================================================

void fct_producer(ThreadSafeStack<int>& stack, int start, int count) {
    for (int i = 0; i < count; ++i) {
        stack.push(start + i);
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

void fct_consumer(ThreadSafeStack<int>& stack, int id, int& total) {
    int local_count = 0;
    for (int i = 0; i < 1000; ++i) {
        auto item = stack.pop();
        if (item) {
            ++local_count;
        } else {
            // 栈为空，稍等
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }
    total += local_count;
    std::cout << "  [Consumer " << id << "] 弹出 " << local_count
              << " 个元素\n";
}

int main() {
    std::cout << "=== 线程安全栈: 完整实现 ===\n\n";

    // --- 测试1: 基本操作 ---
    std::cout << "--- 测试1: 基本 push/pop ---\n";
    {
        ThreadSafeStack<std::string> stack;

        stack.push("first");
        stack.push("second");
        stack.push("third");

        std::cout << "  栈大小: " << stack.size() << "\n";

        std::string value;
        while (stack.pop(value)) {
            std::cout << "  弹出: " << value << "\n";
        }

        std::cout << "  弹出后大小: " << stack.size() << "\n";
    }

    // --- 测试2: shared_ptr pop ---
    std::cout << "\n--- 测试2: shared_ptr 返回 ---\n";
    {
        ThreadSafeStack<int> stack;
        stack.push(10);
        stack.push(20);
        stack.push(30);

        auto item = stack.pop();
        if (item) {
            std::cout << "  弹出: " << *item << " (期望: 30)\n";
        }

        item = stack.pop();
        if (item) {
            std::cout << "  弹出: " << *item << " (期望: 20)\n";
        }
    }

    // --- 测试3: peek (非破坏性查看) ---
    std::cout << "\n--- 测试3: peek 查看栈顶 ---\n";
    {
        ThreadSafeStack<int> stack;
        stack.push(42);
        stack.push(99);

        auto top = stack.peek();
        if (top) {
            std::cout << "  栈顶: " << *top << " (期望: 99)\n";
        }
        std::cout << "  peek后大小: " << stack.size() << " (应该仍是2)\n";
    }

    // --- 测试4: 生产者-消费者并发 ---
    std::cout << "\n--- 测试4: 生产者-消费者并发 ---\n";
    {
        ThreadSafeStack<int> stack;

        const int num_producers = 3;
        const int num_consumers = 3;
        const int items_per_prod = 500;

        std::vector<int> consumer_totals(num_consumers, 0);

        std::vector<std::thread> producers;
        std::vector<std::thread> consumers;

        producers.reserve(num_producers);
        consumers.reserve(num_consumers);

        // 启动生产者
        for (int i = 0; i < num_producers; ++i) {
            producers.emplace_back(fct_producer, std::ref(stack),
                                   i * items_per_prod, items_per_prod);
        }

        // 给生产者一点先发优势
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // 启动消费者
        for (int i = 0; i < num_consumers; ++i) {
            consumers.emplace_back(fct_consumer, std::ref(stack), i,
                                   std::ref(consumer_totals[i]));
        }

        // 等待所有线程完成
        for (auto& t : producers) {
            t.join();
        }
        for (auto& t : consumers) {
            t.join();
        }

        int total_consumed = 0;
        for (int v : consumer_totals) {
            total_consumed += v;
        }

        std::cout << "  生产总数: " << num_producers * items_per_prod << "\n";
        std::cout << "  消费总数: " << total_consumed << "\n";
        std::cout << "  栈剩余:   " << stack.size() << "\n";
        std::cout << "  一致性: "
                  << (total_consumed + stack.size() ==
                              num_producers * items_per_prod
                          ? "✓"
                          : "✗")
                  << "\n";
    }

    // --- 测试5: 空栈处理 ---
    std::cout << "\n--- 测试5: 空栈操作 ---\n";
    {
        ThreadSafeStack<int> stack;

        auto item = stack.pop();
        std::cout << "  空栈 pop: " << (item ? "有值" : "nullptr") << "\n";

        int value = 999;
        bool ok   = stack.pop(value);
        std::cout << "  空栈 pop(ref): ok=" << ok << " value=" << value
                  << "\n";

        auto top = stack.peek();
        std::cout << "  空栈 peek: " << (top ? "有值" : "nullptr") << "\n";
    }

    // --- 测试6: 拷贝构造 ---
    std::cout << "\n--- 测试6: 栈拷贝 ---\n";
    {
        ThreadSafeStack<int> original;
        original.push(1);
        original.push(2);
        original.push(3);

        ThreadSafeStack<int> copy(original);

        std::cout << "  原始栈大小: " << original.size() << "\n";
        std::cout << "  拷贝栈大小: " << copy.size() << "\n";

        // 验证拷贝独立
        original.push(4);
        std::cout << "  修改后: 原始=" << original.size()
                  << " 拷贝=" << copy.size() << "\n";
    }

    std::cout << "\n=== 线程安全栈设计要点 ===\n";
    std::cout << "1. top() 和 pop() 必须合并为原子操作\n";
    std::cout << "2. 合并方案: shared_ptr 返回 或 引用参数\n";
    std::cout << "3. 异常安全: 先构造返回值再 pop\n";
    std::cout << "4. 接口最小化: 不暴露指针/引用给外部\n";
    std::cout << "5. 不提供迭代器: 迭代期间无法保证一致性\n";
    std::cout << "6. 考虑使用 lock-free 数据结构 (后续章节)\n";

    return 0;
}
