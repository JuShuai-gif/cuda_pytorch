// 01_basic_thread.cpp
// 知识点: std::thread 基础 - 创建、join、detach
// 演示: join 与 detach 的区别，以及 detach 的使用场景和注意事项

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

// 一个可以被 detach 的线程函数
// detach 后: 线程在后台运行，主线程不再等待它
void fct_background_task(int id, int duration_ms) {
    std::cout << "[后台任务 " << id << "] 启动，预计运行 " << duration_ms
              << "ms\n";

    for (int i = 0; i < 5; ++i) {
        std::cout << "[后台任务 " << id << "] 第 " << i + 1 << "/5 步\n";
        std::this_thread::sleep_for(
            std::chrono::milliseconds(duration_ms / 5));
    }

    std::cout << "[后台任务 " << id << "] 完成\n";
}

// joinable 检测: 判断线程是否可以被 join 或 detach
void fct_demonstrate_joinability() {
    std::thread t;  // 默认构造: 不代表任何线程，不可 joinable
    std::cout << "默认构造的线程是否 joinable: " << std::boolalpha
              << t.joinable() << "\n";

    t = std::thread([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    });
    std::cout << "赋值后是否 joinable: " << t.joinable() << "\n";

    t.join();
    std::cout << "join后是否 joinable: " << t.joinable() << "\n";
}

int main() {
    std::cout << "=== std::thread 基础 ===\n\n";

    // --- 场景1: join - 等待线程完成 ---
    std::cout << "--- 场景1: join (等待线程完成) ---\n";
    {
        std::thread t([]() {
            std::cout << "  线程 " << std::this_thread::get_id()
                      << " 开始工作\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            std::cout << "  线程 " << std::this_thread::get_id()
                      << " 工作完成\n";
        });

        std::cout << "  主线程: 等待子线程完成...\n";
        t.join();  // 阻塞直到 t 完成
        std::cout << "  主线程: 子线程已 join\n";
    }

    // --- 场景2: detach - 分离线程 ---
    std::cout << "\n--- 场景2: detach (分离线程) ---\n";
    {
        std::thread t(fct_background_task, 1, 300);

        if (t.joinable()) {
            t.detach();  // 分离: 线程在后台独立运行
            std::cout << "  主线程: 线程已 detach，不再等待\n";
        }
        // 注意: detach 后不能再 join

        // 主线程继续执行
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
        std::cout << "  主线程: 完成自己的工作\n";
    }

    // --- 场景3: joinable 检测 ---
    std::cout << "\n--- 场景3: joinable 状态检测 ---\n";
    fct_demonstrate_joinability();

    // --- 场景4: 忘记 join/detach 的后果 ---
    std::cout << "\n--- 场景4: 安全实践 ---\n";
    {
        // 错误示范(已被注释掉):
        // std::thread t([](){ });
        // // 如果 t 是 joinable 状态且在析构前既没有 join 也没有 detach
        // // 程序会调用 std::terminate()!

        // 正确做法: 总是确保 join 或 detach
        std::thread t([]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        });

        // 方式1: 明确 join
        t.join();

        // 方式2: 使用 RAII 包装 (见 02_thread_guard.cpp)
        // 方式3: 使用 std::jthread (C++20, 自动 join)
    }

    std::cout << "\n=== 要点总结 ===\n";
    std::cout << "1. 线程析构前必须是 joinable() == false\n";
    std::cout << "2. join(): 阻塞等待线程完成，回收资源\n";
    std::cout << "3. detach(): 分离线程，后台运行，失去控制\n";
    std::cout << "4. detach 风险: 线程可能访问已销毁的局部变量\n";
    std::cout << "5. 优先使用 join + RAII 或 std::jthread\n";

    return 0;
}
