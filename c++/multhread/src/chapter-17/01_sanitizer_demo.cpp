// 01_sanitizer_demo.cpp — Sanitizer 演示
// 演示: 数据竞争、use-after-free、deadlock 场景
// 编译: g++ -std=c++20 -fsanitize=thread -g -O1 01_sanitizer_demo.cpp -o sanitizer_demo

#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// ===== 场景 1: 数据竞争 (会被 TSan 检测) =====
void data_race_demo() {
    std::cout << "=== 场景 1: 数据竞争 (TSan 应检测到) ===\n";

    int shared = 0; // 非原子变量，无锁保护

    std::jthread t1([&]() {
        for (int i = 0; i < 100000; ++i) {
            ++shared; // 写操作
        }
    });

    std::jthread t2([&]() {
        for (int i = 0; i < 100000; ++i) {
            ++shared; // 并发写 → 数据竞争
        }
    });

    t1.join();
    t2.join();

    std::cout << "  shared = " << shared
              << " (期望 200000, "
              << "TSan 会报告 data race)\n";
}

// ===== 场景 2: 正确的原子操作 (TSan 无警告) =====
void correct_atomic_demo() {
    std::cout << "\n=== 场景 2: 原子操作 (TSan 应无警告) ===\n";

    std::atomic<int> counter{0};

    std::jthread t1([&]() {
        for (int i = 0; i < 100000; ++i) {
            counter.fetch_add(1, std::memory_order_relaxed);
        }
    });

    std::jthread t2([&]() {
        for (int i = 0; i < 100000; ++i) {
            counter.fetch_add(1, std::memory_order_relaxed);
        }
    });

    t1.join();
    t2.join();

    std::cout << "  counter = " << counter.load()
              << " (期望 200000, TSan 无警告)\n";
}

// ===== 场景 3: 锁保护的共享数据 (TSan 无警告) =====
void protected_shared_data_demo() {
    std::cout << "\n=== 场景 3: mutex 保护 (TSan 应无警告) ===\n";

    std::mutex mtx;
    int shared = 0;

    std::jthread t1([&]() {
        for (int i = 0; i < 100000; ++i) {
            std::lock_guard lock(mtx);
            ++shared;
        }
    });

    std::jthread t2([&]() {
        for (int i = 0; i < 100000; ++i) {
            std::lock_guard lock(mtx);
            ++shared;
        }
    });

    t1.join();
    t2.join();

    std::cout << "  shared = " << shared
              << " (期望 200000, TSan 无警告)\n";
}

// ===== 场景 4: 检测到的无用锁 (常见的伪锁错误) =====
void fake_lock_demo() {
    std::cout << "\n=== 场景 4: 伪锁错误 (不同锁保护同一数据) ===\n";

    std::mutex mtx1, mtx2;
    int shared = 0; // 被两个不同的锁"保护"

    std::jthread t1([&]() {
        for (int i = 0; i < 100000; ++i) {
            std::lock_guard lock(mtx1); // 用 mtx1
            ++shared;
        }
    });

    std::jthread t2([&]() {
        for (int i = 0; i < 100000; ++i) {
            std::lock_guard lock(mtx2); // 用 mtx2 — 不同的锁!
            ++shared;
        }
    });

    t1.join();
    t2.join();

    std::cout << "  shared = " << shared
              << " (TSan 应报告: 不同锁保护同一数据 = data race)\n";
}

// ===== 场景 5: 正确的锁用法 =====
void correct_lock_demo() {
    std::cout << "\n=== 场景 5: 正确的锁 (同一锁保护同一数据) ===\n";

    std::mutex mtx;
    int shared = 0;

    std::jthread t1([&]() {
        for (int i = 0; i < 100000; ++i) {
            std::lock_guard lock(mtx);
            ++shared;
        }
    });

    std::jthread t2([&]() {
        for (int i = 0; i < 100000; ++i) {
            std::lock_guard lock(mtx);
            ++shared;
        }
    });

    t1.join();
    t2.join();

    std::cout << "  shared = " << shared
              << " (期望 200000, 正确)\n";
}

// ===== Sanitizer 使用说明 =====
void print_sanitizer_guide() {
    std::cout << "\n============================================\n";
    std::cout << "Sanitizer 使用指南\n";
    std::cout << "============================================\n\n";

    std::cout << "ThreadSanitizer (数据竞争检测):\n";
    std::cout << "  g++ -fsanitize=thread -g -O1 file.cpp -o file\n";
    std::cout << "  ./file\n\n";

    std::cout << "AddressSanitizer (内存错误检测):\n";
    std::cout << "  g++ -fsanitize=address -g -O1 file.cpp -o file\n\n";

    std::cout << "UndefinedBehaviorSanitizer (未定义行为):\n";
    std::cout << "  g++ -fsanitize=undefined -g file.cpp -o file\n\n";

    std::cout << "组合使用:\n";
    std::cout << "  g++ -fsanitize=thread,undefined -g -O1 "
              << "file.cpp -o file\n\n";

    std::cout << "CMake 集成:\n";
    std::cout << "  set(CMAKE_CXX_FLAGS_DEBUG "
              << "\"-fsanitize=thread -g -O1\")\n";
}

int main() {
    data_race_demo();
    correct_atomic_demo();
    protected_shared_data_demo();
    fake_lock_demo();
    correct_lock_demo();
    print_sanitizer_guide();

    std::cout << "\n提示: 不带 -fsanitize=thread 编译看不到检测信息。\n";
    std::cout << "重新编译: g++ -std=c++20 -fsanitize=thread -g -O1 "
              << "01_sanitizer_demo.cpp -o sanitizer_demo\n";
    return 0;
}
