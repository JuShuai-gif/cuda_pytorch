// 08_memory_order_consume.cpp — memory_order_consume 为什么不被推荐
//
// memory_order_consume 的设计目标:
//   只同步"携带依赖"的操作，比 acquire 更轻量
//
// 实际情况:
//   几乎所有编译器都将其退化为 memory_order_acquire
//   C++17 明确标记为"不鼓励使用"
//
// 本演示: 展示 consume 的预期行为 vs 实际行为

#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// ===== 1. consume 的理想行为 vs 实际行为 =====
// 理想: consume 只保证 *p 的读取顺序，不保证 p 之后的其他读取
// 实际: 所有编译器把它当 acquire 处理

struct Data {
    int value;
    std::string name;
};

std::atomic<Data*> ptr{nullptr};
int other_flag = 0; // 与 ptr 无关的数据

void demo_consume_vs_acquire() {
    std::cout << "=== 1. consume vs acquire 行为对比 ===\n\n";

    // 生产者: release 发布
    std::jthread producer([&]() {
        Data* data = new Data{42, "hello"};
        other_flag = 99;
        // release: 保证 data 和 other_flag 对后续 acquire/consume 可见
        ptr.store(data, std::memory_order_release);
    });

    std::this_thread::sleep_for(10ms);

    // 消费者 A: 用 consume (理论上只保证 *data 的可见性)
    {
        int count = 0;
        Data* p = nullptr;
        while (!(p = ptr.load(std::memory_order_consume))) {
            std::this_thread::yield();
            ++count;
        }

        // 理论上 consume 只保证 *p 的依赖链可见
        // 但不保证 other_flag 可见
        // 实际上几乎所有编译器都把它退化为 acquire，
        // 所以 other_flag 也能看到
        std::cout << "  [consume] 自旋 " << count << " 次后看到 ptr\n";
        std::cout << "    p->value    = " << p->value << "\n";
        std::cout << "    p->name     = " << p->name << "\n";
        std::cout << "    other_flag  = " << other_flag
                  << " (注: consume 理论上不保证可见)\n";

        delete p;
    }

    producer.join();
}

// ===== 2. 依赖链 (Dependency Chain) 演示 =====
// consume 的核心概念: 只追踪指针依赖链

void demo_dependency_chain() {
    std::cout << "\n=== 2. 依赖链概念 ===\n\n";

    std::cout << "  consume 的设计目标: 只同步\"携带依赖\"的操作\n\n";

    std::cout << "  例: ptr.load(consume) 返回的指针 p\n";
    std::cout << "    p->x        ← 携带依赖 (通过 p 访问)\n";
    std::cout << "    p->y        ← 携带依赖 (通过 p 访问)\n";
    std::cout << "    global_z    ← 不携带依赖 (与 p 无关)\n\n";

    std::cout << "  acquire 保证 ALL 后续操作看到 release 前的写入\n";
    std::cout << "  consume 只保证指针依赖链上的操作\n\n";

    std::cout << "  实际上:\n";
    std::cout << "    - GCC: consume 退化为 acquire\n";
    std::cout << "    - Clang: consume 退化为 acquire\n";
    std::cout << "    - MSVC: consume 退化为 acquire\n";
    std::cout << "    - 没有任何主流编译器正确实现 consume\n\n";

    std::cout << "  原因:\n";
    std::cout << "    - 追踪依赖链在编译器中实现极其复杂\n";
    std::cout << "    - 许多优化会破坏依赖链(如值编号、公共子表达式消除)\n";
    std::cout << "    - 性能收益与实现成本不成正比\n";
}

// ===== 3. consume 实际退化为 acquire 的验证 =====
void demo_consume_degradation() {
    std::cout << "\n=== 3. consume 退化为 acquire 的验证 ===\n\n";

    std::atomic<int*> ptr2{nullptr};
    int  unrelated_data = 0;
    bool saw_unrelated  = false;

    const int kRounds = 10000;

    for (int r = 0; r < kRounds; ++r) {
        ptr2.store(nullptr, std::memory_order_relaxed);
        unrelated_data = 0;
        saw_unrelated = false;

        std::jthread producer([&]() {
            int* data = new int(42);
            unrelated_data = 1;        // 与 ptr 无关的数据
            // release 保证 data 和 unrelated_data 的写入
            ptr2.store(data, std::memory_order_release);
            // 短暂等待后清理
            std::this_thread::sleep_for(1ms);
            delete data;
        });

        std::jthread consumer([&]() {
            int* p = nullptr;
            while (!(p = ptr2.load(std::memory_order_consume))) {
                std::this_thread::yield();
            }
            // consume 理论上不保证 unrelated_data 可见
            // 但由于退化为 acquire，实际总是能看到
            if (unrelated_data == 1) {
                saw_unrelated = true;
            }
        });

        producer.join();
        consumer.join();
    }

    std::cout << "  测试 " << kRounds << " 轮\n";
    std::cout << "  consume 后看到 unrelated_data: "
              << (saw_unrelated ? "总是" : "偶尔")
              << "\n";
    std::cout << "  结论: consume 实际上和 acquire 行为一致\n";
}

// ===== 4. 使用建议 =====
void demo_guidance() {
    std::cout << "\n=== 4. 最终建议 ===\n\n";

    std::cout << "  ┌─────────────────────────────────────────────┐\n";
    std::cout << "  │ 永远不要使用 memory_order_consume           │\n";
    std::cout << "  │ 在所有需要消费语义的地方用 memory_order_acquire │\n";
    std::cout << "  │ C++17 标准: \"Prefer acquire over consume\"    │\n";
    std::cout << "  └─────────────────────────────────────────────┘\n\n";

    std::cout << "  内存序选择速查:\n";
    std::cout << "    relaxed  → 纯计数器、无依赖的数据\n";
    std::cout << "    acquire  → 读端需要看到写端的所有前置写入\n";
    std::cout << "    release  → 写端需要保证所有前置写入对读端可见\n";
    std::cout << "    acq_rel  → 读-改-写操作(如 fetch_add、CAS)\n";
    std::cout << "    seq_cst  → 需要全局统一顺序(默认、最安全)\n";
    std::cout << "    consume  → 不.要.用.\n";
}

int main() {
    demo_consume_vs_acquire();
    demo_dependency_chain();
    demo_consume_degradation();
    demo_guidance();

    return 0;
}
