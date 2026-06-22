/**
 * 02_race_detector_demo.cpp — ThreadSanitizer (TSan) 使用演示
 *
 * TSan 是 Google 开发的动态数据竞争检测工具, 集成在 GCC/Clang 中。
 *
 * 编译并运行:
 *   g++ -std=c++20 -g -O1 -fsanitize=thread -pthread 02_race_detector_demo.cpp -o tsan_demo
 *   ./tsan_demo 2>&1 | head -50
 *
 * 注意: TSan 在运行时检测, 不是编译期检测。
 *       需要实际触发竞争路径才能检测到。
 *
 * 不想要 TSan 时只需去掉 -fsanitize=thread 即可正常编译运行。
 */

#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <atomic>
#include <chrono>
#include <string>
#include <cstring>

// ============================================================================
// 场景1: 简单数据竞争 (TSan 可检测)
// ============================================================================
namespace race1 {

int g_counter = 0; // ⚠️ 无保护的全局变量

void unsafe_increment() {
    for (int i = 0; i < 100000; ++i) {
        ++g_counter; // 数据竞争! 读-改-写 不是原子的
    }
}

void demo() {
    std::cout << "=== 场景1: 简单数据竞争 ===\n";
    std::cout << "  (用 TSan 运行: g++ -fsanitize=thread ...)\n\n";

    g_counter = 0;
    std::jthread t1(unsafe_increment);
    std::jthread t2(unsafe_increment);
    t1.join();
    t2.join();

    std::cout << "  期望: 200000, 实际: " << g_counter;
    if (g_counter != 200000) {
        std::cout << " ← 数据竞争导致丢失更新!\n";
    } else {
        std::cout << " (恰好正确, 但仍有数据竞争)\n";
    }
    std::cout << "\n";
}

} // namespace race1

// ============================================================================
// 场景2: 安全的修复方案
// ============================================================================
namespace race2 {

int g_counter = 0;
std::mutex g_mutex; // ✅ 加锁保护

void safe_increment() {
    for (int i = 0; i < 100000; ++i) {
        std::lock_guard<std::mutex> lock(g_mutex);
        ++g_counter;
    }
}

// 或者使用原子变量
std::atomic<int> g_atomic_counter{0};

void atomic_increment() {
    for (int i = 0; i < 100000; ++i) {
        g_atomic_counter.fetch_add(1, std::memory_order_relaxed);
    }
}

void demo() {
    std::cout << "=== 场景2: 正确同步 ===\n\n";

    // 方案A: 互斥锁
    {
        g_counter = 0;
        std::jthread t1(safe_increment);
        std::jthread t2(safe_increment);
        t1.join();
        t2.join();
        std::cout << "  mutex 版本: 期望=200000, 实际=" << g_counter
                  << (g_counter == 200000 ? " ✓" : " ✗") << "\n";
    }

    // 方案B: 原子操作
    {
        g_atomic_counter.store(0);
        std::jthread t1(atomic_increment);
        std::jthread t2(atomic_increment);
        t1.join();
        t2.join();
        std::cout << "  atomic 版本: 期望=200000, 实际=" << g_atomic_counter.load()
                  << (g_atomic_counter.load() == 200000 ? " ✓" : " ✗") << "\n";
    }

    std::cout << "\n";
}

} // namespace race2

// ============================================================================
// 场景3: 隐蔽的数据竞争 — vtable 指针竞争 (TSan 可检测)
// ============================================================================
namespace race3 {

struct Base {
    int data_{0};
    virtual ~Base() = default;
    virtual void do_work() { data_ += 1; }
};

Base* g_obj = nullptr;
std::mutex g_mutex;

void writer_thread() {
    std::lock_guard<std::mutex> lock(g_mutex);
    delete g_obj;
    g_obj = new Base();
}

void reader_thread() {
    // ⚠️ 没有加锁! 可能读取到半构造对象或已销毁对象
    if (g_obj) {
        g_obj->do_work(); // 数据竞争! vtable 指针可能正在被修改
    }
}

void demo() {
    std::cout << "=== 场景3: vtable 指针竞争 ===\n";
    std::cout << "  (TSan 可能检测到对象构造/析构期间的竞争)\n\n";

    g_obj = new Base();

    std::jthread writer([&]() {
        for (int i = 0; i < 10000; ++i) {
            writer_thread();
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    });

    std::jthread reader([&]() {
        for (int i = 0; i < 10000; ++i) {
            reader_thread();
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    });

    writer.join();
    reader.join();

    std::lock_guard<std::mutex> lock(g_mutex);
    delete g_obj;
    g_obj = nullptr;

    std::cout << "  运行完成 (可能有隐蔽的数据竞争)\n\n";
}

} // namespace race3

// ============================================================================
// 场景4: false positive vs true positive
// ============================================================================
namespace race4 {

// 某些看似竞争的代码实际上是安全的
std::atomic<int> g_flag{0};
int g_shared_data = 0; // 被 g_flag 保护

void producer() {
    g_shared_data = 42;
    g_flag.store(1, std::memory_order_release);
}

void consumer() {
    while (g_flag.load(std::memory_order_acquire) == 0) {
        std::this_thread::yield();
    }
    // 到这里, g_shared_data 保证可见
    int val = g_shared_data;
    std::cout << "  消费者读到: " << val << " (期望 42)\n";
}

void demo() {
    std::cout << "=== 场景4: 正确的 release/acquire 同步 ===\n";
    std::cout << "  TSan 应感知 release/acquire 语义, 不会误报\n\n";

    g_flag.store(0);
    g_shared_data = 0;

    std::jthread t1(producer);
    std::jthread t2(consumer);
    t1.join();
    t2.join();

    std::cout << "\n";
}

} // namespace race4

// ============================================================================
// TSan 使用指南
// ============================================================================
void tsan_guide() {
    std::cout << "=== ThreadSanitizer 使用指南 ===\n\n";
    std::cout << "  [编译]\n";
    std::cout << "  g++ -std=c++20 -g -O1 -fsanitize=thread -pthread source.cpp -o output\n\n";
    std::cout << "  [运行]\n";
    std::cout << "  ./output     # TSan 会在检测到竞争时打印报告\n\n";
    std::cout << "  [常用环境变量]\n";
    std::cout << "  TSAN_OPTIONS=\"history_size=7\" ./output\n";
    std::cout << "  TSAN_OPTIONS=\"second_deadlock_stack=1\" ./output\n";
    std::cout << "  TSAN_OPTIONS=\"suppressions=tsan_suppressions.txt\" ./output\n\n";
    std::cout << "  [抑制文件示例] (tsan_suppressions.txt):\n";
    std::cout << "  # 忽略第三方库的已知竞争\n";
    std::cout << "  race:some_library_function\n\n";
    std::cout << "  [CMake 集成]\n";
    std::cout << "  set(CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} -fsanitize=thread\")\n";
    std::cout << "  set(CMAKE_LINKER_FLAGS \"${CMAKE_LINKER_FLAGS} -fsanitize=thread\")\n\n";
    std::cout << "  [注意]\n";
    std::cout << "  - TSan 无法与 ASan (AddressSanitizer) 同时使用\n";
    std::cout << "  - TSan 会显著降低运行速度 (5-15x) 和增加内存 (5-10x)\n";
    std::cout << "  - 仅在 Debug/Test 构建中使用, 不要用于生产环境\n";
}

// ============================================================================
// main
// ============================================================================
int main() {
    std::cout << "╔══════════════════════════════════════════════╗\n";
    std::cout << "║  ThreadSanitizer 数据竞争检测演示             ║\n";
    std::cout << "║  请使用 -fsanitize=thread 编译以启用 TSan     ║\n";
    std::cout << "╚══════════════════════════════════════════════╝\n\n";

    race1::demo();
    race2::demo();
    race3::demo();
    race4::demo();
    tsan_guide();

    return 0;
}
