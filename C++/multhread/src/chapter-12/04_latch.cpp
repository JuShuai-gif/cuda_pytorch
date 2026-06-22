// 04_latch.cpp — std::latch 一次性同步点
// 演示: 等待所有线程就绪、多阶段启动、超时模拟

#include <chrono>
#include <iostream>
#include <latch>
#include <syncstream>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// ===== 1. 基础: 等待所有线程就绪后同时开始 =====
void demo_basic_latch() {
    std::cout << "=== 1. 起跑门控 ===\n";

    const int kRunners = 5;
    std::latch start_gate{1};  // 裁判控制，初始 1
    std::latch finish_gate{kRunners}; // 等所有运动员完成

    auto runner = [&](int id) {
        std::osyncstream(std::cout)
            << "  运动员 " << id << " 就位，等待发令...\n";
        start_gate.wait(); // 等待发令枪

        std::osyncstream(std::cout)
            << "  运动员 " << id << " 起跑！\n";
        std::this_thread::sleep_for((kRunners - id) * 20ms);

        finish_gate.count_down(); // 到达终点
    };

    std::vector<std::jthread> runners;
    for (int i = 0; i < kRunners; ++i) {
        runners.emplace_back(runner, i);
    }

    std::this_thread::sleep_for(100ms);
    std::cout << "  发令枪响！\n";
    start_gate.count_down(); // 所有运动员同时起跑

    finish_gate.wait(); // 等待全部完赛
    std::cout << "  所有运动员完赛！\n";
}

// ===== 2. 等待多任务初始化完成 =====
void demo_init_latch() {
    std::cout << "\n=== 2. 初始化同步 ===\n";

    const int kServices = 4;
    std::latch init_done{kServices};

    std::vector<std::string> services{"Database", "Cache", "Queue", "Config"};
    std::vector<std::jthread> threads;

    for (int i = 0; i < kServices; ++i) {
        threads.emplace_back([&, i]() {
            // 模拟初始化时间不同
            std::this_thread::sleep_for((i + 1) * 30ms);
            std::osyncstream(std::cout)
                << "  " << services[i] << " 初始化完成\n";
            init_done.count_down();
        });
    }

    std::cout << "  等待所有服务初始化...\n";
    init_done.wait();
    std::cout << "  所有服务就绪，开始提供服务！\n";
}

// ===== 3. try_wait 非阻塞检查 =====
void demo_try_wait() {
    std::cout << "\n=== 3. try_wait 非阻塞检查 ===\n";

    std::latch done{2};
    bool ready = done.try_wait();
    std::cout << "  初始 try_wait: " << std::boolalpha << ready << "\n";

    done.count_down();
    ready = done.try_wait();
    std::cout << "  一次 count_down 后: " << ready << "\n";

    done.count_down();
    ready = done.try_wait();
    std::cout << "  两次 count_down 后: " << ready << "\n";
}

// ===== 4. max() 静态上限 =====
void demo_latch_limits() {
    std::cout << "\n=== 4. latch 限制 ===\n";
    std::cout << "  std::latch::max() = "
              << std::latch::max() << " (平台上限)\n";

    // latch 不可复制、不可移动
    std::latch small{1};
    // std::latch copy = small;  // 编译错误

    std::cout << "  注意: latch 不可复制/移动，用完即弃\n";
}

int main() {
    demo_basic_latch();
    demo_init_latch();
    demo_try_wait();
    demo_latch_limits();

    std::cout << "\nlatch 适合\"一次性门控\"场景，比条件变量简单得多。\n";
    return 0;
}
