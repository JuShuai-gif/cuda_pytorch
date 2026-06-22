// 03_joining_thread.cpp
// 知识点: joining_thread - 析构时自动 join 的线程类
// 演示: 手动实现类似 std::jthread (C++20) 的 RAII 线程
// 这是书中 2.1.3 节的扩展实现

#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <utility>

// =============================================================================
// JoiningThread: 拥有线程所有权，析构时自动 join
// 与 ThreadGuard 的区别: 直接拥有线程对象（值语义），而非持有引用
// 等价于 C++20 的 std::jthread 简化版
// =============================================================================
class JoiningThread {
public:
    // 默认构造 - 空线程
    JoiningThread() noexcept = default;

    // 从可调用对象构造 (万能引用 + 完美转发)
    template <typename Callable, typename... Args>
    explicit JoiningThread(Callable&& func, Args&&... args)
        : m_thread(std::forward<Callable>(func),
                   std::forward<Args>(args)...) {}

    // 从 std::thread 移动构造 (接管所有权)
    explicit JoiningThread(std::thread t) noexcept
        : m_thread(std::move(t)) {}

    // 禁止拷贝
    JoiningThread(const JoiningThread&)            = delete;
    JoiningThread& operator=(const JoiningThread&) = delete;

    // 移动构造
    JoiningThread(JoiningThread&& other) noexcept
        : m_thread(std::move(other.m_thread)) {}

    // 移动赋值
    JoiningThread& operator=(JoiningThread&& other) noexcept {
        if (this != &other) {
            join();  // 先 join 当前线程
            m_thread = std::move(other.m_thread);
        }
        return *this;
    }

    // 移动赋值 std::thread
    JoiningThread& operator=(std::thread t) noexcept {
        join();
        m_thread = std::move(t);
        return *this;
    }

    // 析构: 自动 join
    ~JoiningThread() { join(); }

    // 检查是否有线程
    [[nodiscard]] bool joinable() const noexcept {
        return m_thread.joinable();
    }

    // 手动 join
    void join() {
        if (m_thread.joinable()) {
            m_thread.join();
        }
    }

    // 手动 detach (谨慎使用)
    void detach() {
        if (m_thread.joinable()) {
            m_thread.detach();
        }
    }

    // 获取底层 std::thread ID
    [[nodiscard]] std::thread::id get_id() const noexcept {
        return m_thread.get_id();
    }

    // 获取底层 std::thread 引用 (危险，谨慎使用)
    [[nodiscard]] std::thread& native_handle() noexcept { return m_thread; }
    [[nodiscard]] const std::thread& native_handle() const noexcept {
        return m_thread;
    }

    // 交换
    void swap(JoiningThread& other) noexcept {
        m_thread.swap(other.m_thread);
    }

private:
    std::thread m_thread;
};

// 非成员 swap
inline void swap(JoiningThread& a, JoiningThread& b) noexcept {
    a.swap(b);
}

// =============================================================================
// 演示
// =============================================================================

void fct_worker(int id, int delay_ms) {
    std::cout << "[JoinThread-" << id << "] 开始工作\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    std::cout << "[JoinThread-" << id << "] 工作完成\n";
}

int main() {
    std::cout << "=== JoiningThread: 自动 join 的线程 ===\n\n";

    // --- 测试1: 正常退出时自动 join ---
    std::cout << "--- 测试1: 自动 join ---\n";
    {
        JoiningThread jt(fct_worker, 1, 100);
        std::cout << "  线程 joinable: " << std::boolalpha << jt.joinable()
                  << "\n";
        std::cout << "  线程 ID: " << jt.get_id() << "\n";
    }  // jt 析构时自动 join
    std::cout << "  jt 已析构\n";

    // --- 测试2: 使用 lambda ---
    std::cout << "\n--- 测试2: lambda 表达式 ---\n";
    {
        const std::string msg = "Hello from JoiningThread!";
        JoiningThread      jt([&msg]() {
            std::cout << "  消息: " << msg << "\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        });
    }

    // --- 测试3: 移动语义 ---
    std::cout << "\n--- 测试3: 移动语义 ---\n";
    {
        JoiningThread jt1(fct_worker, 10, 100);
        std::cout << "  jt1 线程 ID: " << jt1.get_id() << "\n";

        // 移动构造
        JoiningThread jt2(std::move(jt1));
        std::cout << "  移动后 jt1 joinable: " << jt1.joinable() << "\n";
        std::cout << "  移动后 jt2 joinable: " << jt2.joinable() << "\n";
        std::cout << "  jt2 线程 ID: " << jt2.get_id() << "\n";
    }

    // --- 测试4: 容器存储 ---
    std::cout << "\n--- 测试4: 容器中存储 JoiningThread ---\n";
    {
        std::vector<JoiningThread> threads;
        threads.reserve(3);

        for (int i = 0; i < 3; ++i) {
            threads.emplace_back(fct_worker, i + 20, 100);
        }

        std::cout << "  容器中有 " << threads.size() << " 个线程\n";
        // 容器析构时，每个 JoiningThread 自动 join
    }

    // --- 测试5: swap ---
    std::cout << "\n--- 测试5: swap ---\n";
    {
        JoiningThread jt1([]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        });
        JoiningThread jt2;  // 空

        std::cout << "  swap前 - jt1 joinable: " << jt1.joinable()
                  << ", jt2 joinable: " << jt2.joinable() << "\n";

        jt1.swap(jt2);

        std::cout << "  swap后 - jt1 joinable: " << jt1.joinable()
                  << ", jt2 joinable: " << jt2.joinable() << "\n";
    }

    std::cout << "\n=== JoiningThread vs std::jthread ===\n";
    std::cout << "1. JoiningThread: 手动实现，用于理解原理\n";
    std::cout << "2. std::jthread (C++20): 标准库版本，功能更完整\n";
    std::cout << "3. std::jthread 额外支持: stop_token 协作取消\n";
    std::cout << "4. 生产环境优先使用 std::jthread\n";

    return 0;
}
