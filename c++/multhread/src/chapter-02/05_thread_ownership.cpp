// 05_thread_ownership.cpp
// 知识点: std::thread 所有权转移 (移动语义)
// 演示: 线程不可拷贝但可移动，线程容器管理，各种所有权转移场景
// 对应书中 2.3 节

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

void fct_named_task(const std::string& name) {
    std::cout << "  [任务 " << name << "] 在线程 " << std::this_thread::get_id()
              << " 中执行\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// 工厂函数: 返回线程 (移动语义)
std::thread fct_spawn_worker(int id) {
    return std::thread([id]() {
        std::cout << "  [工厂线程 #" << id << "] 开始\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::cout << "  [工厂线程 #" << id << "] 完成\n";
    });
}

// 接收线程所有权作为参数
void fct_accept_thread(std::thread t, const std::string& owner) {
    std::cout << "  [" << owner << "] 接管了线程 " << t.get_id() << "\n";
    if (t.joinable()) {
        t.join();
    }
    std::cout << "  [" << owner << "] 线程已 join\n";
}

int main() {
    std::cout << "=== std::thread 所有权转移 ===\n\n";

    // --- 场景1: 显式移动 ---
    std::cout << "--- 场景1: std::move 转移所有权 ---\n";
    {
        std::thread t1(fct_named_task, "Task-A");
        std::cout << "  t1 持有线程: " << t1.get_id() << "\n";

        // 移动: t1 失去所有权，t2 获得所有权
        std::thread t2 = std::move(t1);
        std::cout << "  移动后 t1 joinable: " << t1.joinable() << "\n";
        std::cout << "  移动后 t2 joinable: " << t2.joinable() << "\n";
        std::cout << "  t2 持有线程: " << t2.get_id() << "\n";

        t2.join();
    }

    // --- 场景2: 工厂函数返回线程 ---
    std::cout << "\n--- 场景2: 工厂函数返回线程 ---\n";
    {
        std::thread t = fct_spawn_worker(1);
        std::cout << "  主线程接管了工厂线程: " << t.get_id() << "\n";
        t.join();
    }

    // --- 场景3: 将线程所有权传递给函数 ---
    std::cout << "\n--- 场景3: 传入线程所有权 ---\n";
    {
        std::thread t(fct_named_task, "Task-B");
        fct_accept_thread(std::move(t), "Receiver");
        // 此时 t 不再持有线程
        std::cout << "  传入后 t joinable: " << t.joinable() << "\n";
    }

    // --- 场景4: 线程容器 (批量管理) ---
    std::cout << "\n--- 场景4: std::vector<std::thread> ---\n";
    {
        const int                  num_threads = 4;
        std::vector<std::thread>   threads;
        threads.reserve(num_threads);

        // emplace_back 直接构造 (无需移动)
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back(fct_named_task,
                                 "Vector-" + std::to_string(i));
        }

        std::cout << "  容器中有 " << threads.size() << " 个线程\n";

        // 批量 join
        for (auto& t : threads) {
            if (t.joinable()) {
                t.join();
            }
        }
        std::cout << "  所有线程已 join\n";
    }

    // --- 场景5: 使用 std::generate_n 填充容器 ---
    std::cout << "\n--- 场景5: std::generate_n 填充线程 ---\n";
    {
        std::vector<std::thread> threads;
        threads.reserve(3);

        int counter = 0;
        std::generate_n(std::back_inserter(threads), 3,
                        [&counter]() -> std::thread {
                            int id = ++counter;
                            return std::thread([id]() {
                                std::cout << "  [生成线程 #" << id
                                          << "] 执行\n";
                                std::this_thread::sleep_for(
                                    std::chrono::milliseconds(50));
                            });
                        });

        for (auto& t : threads) {
            t.join();
        }
    }

    // --- 场景6: 移动赋值 ---
    std::cout << "\n--- 场景6: 移动赋值 ---\n";
    {
        std::thread t1(fct_named_task, "Original");

        // 移动赋值: t1 的原线程被无缝替换
        // 注意: 如果 t1 的当前线程是 joinable 状态
        // 且没有 join/detach，会导致 std::terminate()
        // 所以必须先确保 t1 已 join 或为空
        t1 = fct_spawn_worker(99);  // OK: t1 已经移走过，为空
        std::cout << "  t1 现在持有线程: " << t1.get_id() << "\n";
        t1.join();
    }

    std::cout << "\n=== 所有权转移要点 ===\n";
    std::cout << "1. std::thread 不可拷贝，只能移动\n";
    std::cout << "2. 工厂函数可返回 std::thread (隐式移动)\n";
    std::cout << "3. std::vector<std::thread> 管理线程池\n";
    std::cout << "4. 移动前确保目标线程不是 joinable 状态\n";
    std::cout << "5. C++20 std::jthread 同样支持移动语义\n";

    return 0;
}
