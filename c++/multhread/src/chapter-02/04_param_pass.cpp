// 04_param_pass.cpp
// 知识点: 向线程函数传递参数的多种方式
// 演示: 传值、传引用(std::ref)、成员函数指针、以及常见陷阱
// 对应书中 2.2 节

#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// =============================================================================
// 辅助类: 用于演示参数传递
// =============================================================================

class DataProcessor {
public:
    explicit DataProcessor(std::string name) : m_name(std::move(name)) {
        std::cout << "[DataProcessor] 构造: " << m_name << "\n";
    }

    ~DataProcessor() {
        std::cout << "[DataProcessor] 析构: " << m_name << "\n";
    }

    // 禁止拷贝 (演示移动语义)
    DataProcessor(const DataProcessor&)            = delete;
    DataProcessor& operator=(const DataProcessor&) = delete;

    DataProcessor(DataProcessor&& other) noexcept
        : m_name(std::move(other.m_name)) {
        std::cout << "[DataProcessor] 移动构造: " << m_name << "\n";
    }

    void process(int id) const {
        std::cout << "  [Processor " << m_name << "] 处理任务 #" << id
                  << " 在线程 " << std::this_thread::get_id() << "\n";
    }

    void batch_process(int start, int count) const {
        for (int i = 0; i < count; ++i) {
            process(start + i);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

    [[nodiscard]] const std::string& name() const { return m_name; }

private:
    std::string m_name;
};

// 普通函数: 演示参数传递
void fct_show_message(const std::string& msg, int repeat) {
    for (int i = 0; i < repeat; ++i) {
        std::cout << "  [消息] " << msg << " (" << i + 1 << "/" << repeat
                  << ")\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
}

// 可调用对象: 函数对象(functor)
class TaskFunctor {
public:
    explicit TaskFunctor(int id) : m_id(id) {}
    void operator()() const {
        std::cout << "  [TaskFunctor #" << m_id << "] 执行\n";
    }

private:
    int m_id;
};

int main() {
    std::cout << "=== 线程参数传递 ===\n\n";

    // --- 方式1: 传值 ---
    std::cout << "--- 方式1: 传值 (值拷贝到新线程) ---\n";
    {
        std::string msg   = "Hello Thread";
        int         count = 2;

        // 参数被拷贝(或移动)到新线程的存储中
        // 即使 msg, count 在主线程中销毁，新线程仍持有副本
        std::thread t(fct_show_message, msg, count);
        t.join();
    }

    // --- 方式2: std::ref 传引用 ---
    std::cout << "\n--- 方式2: std::ref 传引用 ---\n";
    {
        int shared_counter = 0;

        // 使用 std::ref 包装引用参数
        // 注意: 必须确保被引用对象在线程执行期间保持有效!
        std::thread t([](int& counter) {
            for (int i = 0; i < 5; ++i) {
                ++counter;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }, std::ref(shared_counter));

        t.join();
        std::cout << "  shared_counter = " << shared_counter
                  << " (期望: 5)\n";
    }

    // --- 方式3: 成员函数指针 ---
    std::cout << "\n--- 方式3: 成员函数指针 ---\n";
    {
        DataProcessor proc("MainProcessor");

        // 语法: &ClassName::method, &object/pointer, args...
        std::thread t(&DataProcessor::batch_process, &proc, 100, 3);
        t.join();
    }

    // --- 方式4: 可调用对象 ---
    std::cout << "\n--- 方式4: 可调用对象 (函数对象/lambda) ---\n";
    {
        // 4a: 函数对象
        TaskFunctor task(42);
        std::thread t1(task);  // 拷贝 task 到新线程
        t1.join();

        // 4b: lambda (最常用)
        int  local_var = 100;
        auto lambda    = [&local_var]() {
            local_var += 10;
            std::cout << "  [Lambda] local_var = " << local_var << "\n";
        };
        std::thread t2(lambda);
        t2.join();
        std::cout << "  join后 local_var = " << local_var << "\n";
    }

    // --- 方式5: std::move 移动语义 ---
    std::cout << "\n--- 方式5: std::move 移动大对象 ---\n";
    {
        DataProcessor big_proc("BigDataProcessor");

        // 使用 std::move 转移所有权，避免拷贝
        std::thread t(
            [](DataProcessor proc) {
                proc.process(1);
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            },
            std::move(big_proc));

        t.join();
        std::cout << "  big_proc 已被移动到线程中\n";
    }

    // --- 常见陷阱 ---
    std::cout << "\n--- 常见陷阱演示 ---\n";

    // 陷阱1: 临时对象问题
    std::cout << "陷阱1: 指针/引用悬挂\n";
    {
        std::thread t;
        {
            std::string local_str = "local string";
            // 危险: local_str 在作用域结束后销毁
            // t = std::thread(fct_show_message, std::ref(local_str), 1);
            // 禁止使用! 线程执行时 local_str 可能已被销毁

            // 安全做法: 传值(拷贝)
            t = std::thread(fct_show_message, local_str, 1);
        }  // local_str 销毁，但线程中持有的是副本
        t.join();
        std::cout << "  (传值安全，线程持有副本)\n";
    }

    // 陷阱2: 隐式类型转换的时机
    std::cout << "\n陷阱2: 隐式转换发生在子线程\n";
    {
        char buffer[] = "hello";
        // buffer 隐式转换为 std::string
        // 这个转换发生在新线程中，可能太晚!
        // 如果 buffer 是局部变量的指针，可能已被销毁
        // 安全做法: 显式构造 std::string 后再传递
        std::thread t(fct_show_message, std::string(buffer), 1);
        t.join();
        std::cout << "  (显式构造 std::string，安全)\n";
    }

    std::cout << "\n=== 参数传递最佳实践 ===\n";
    std::cout << "1. 默认传值: 线程持有独立副本，最安全\n";
    std::cout << "2. std::ref: 需要共享状态时使用，确保生命周期\n";
    std::cout << "3. 成员函数: &Class::method, &obj, args...\n";
    std::cout << "4. std::move: 转移大对象所有权，避免拷贝\n";
    std::cout << "5. 避免隐式转换: 显式构造参数后再传递\n";
    std::cout << "6. 指针/引用: 必须确保指向的对象在线程运行期间有效\n";

    return 0;
}
