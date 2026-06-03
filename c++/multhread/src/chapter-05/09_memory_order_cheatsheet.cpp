// 09_memory_order_cheatsheet.cpp — 六种内存序全面对比速查
//
// 用一个统一场景演示所有内存序的语义差异:
//   - 单一线程内: 无意义(一律按程序顺序)
//   - 跨线程: 通过生产者-消费者模型展示每种序的行为边界
//
// 注意: 此文件用于教学理解，实际性能测量见 ch15

#include <atomic>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// ================================================================
// 1. 概念速查表 (直接输出)
// ================================================================
void print_cheatsheet() {
    std::cout << "=== 六种内存序速查表 ===\n\n";

    std::cout << "┌────────────────────┬────────────────────────────────┬──────────┐\n";
    std::cout << "│ 内存序              │ 语义                           │ 代价     │\n";
    std::cout << "├────────────────────┼────────────────────────────────┼──────────┤\n";
    std::cout << "│ memory_order_relaxed│ 仅保证原子性，无顺序保证       │ 最低     │\n";
    std::cout << "│ memory_order_consume│ 仅同步依赖链(实际退化为acquire)│ 理论上低 │\n";
    std::cout << "│ memory_order_acquire│ 后续操作不会重排到此操作之前   │ 中       │\n";
    std::cout << "│ memory_order_release│ 之前操作不会重排到此操作之后   │ 中       │\n";
    std::cout << "│ memory_order_acq_rel│ acquire + release (RMW操作)    │ 中高     │\n";
    std::cout << "│ memory_order_seq_cst│ 全局统一全序，最严格           │ 最高     │\n";
    std::cout << "└────────────────────┴────────────────────────────────┴──────────┘\n\n";
}

// ================================================================
// 2. 统一的生产者-消费者测试框架
// ================================================================

// 用模板参数指定不同的内存序，观察正确性
template <std::memory_order StoreOrder, std::memory_order LoadOrder>
struct SyncPair {
    static constexpr auto store_order = StoreOrder;
    static constexpr auto load_order  = LoadOrder;
};

using RelaxedPair   = SyncPair<std::memory_order_relaxed,   std::memory_order_relaxed>;
using AcqRelPair    = SyncPair<std::memory_order_release,   std::memory_order_acquire>;
using SeqCstPair    = SyncPair<std::memory_order_seq_cst,   std::memory_order_seq_cst>;
using ConsumePair   = SyncPair<std::memory_order_release,   std::memory_order_consume>;
using RelaxedStoreOnly = SyncPair<std::memory_order_release, std::memory_order_relaxed>; // 错误用法演示

// 测试函数: 给定内存序对，检查消费者能否看到生产者的数据
template <typename SyncPairType>
int test_sync_pair(int rounds) {
    std::atomic<bool> ready{false};
    int data = 0;
    int errors = 0;

    for (int r = 0; r < rounds; ++r) {
        ready.store(false, std::memory_order_relaxed);
        data = 0;

        std::jthread producer([&]() {
            data = 42;
            ready.store(true, SyncPairType::store_order);
        });

        std::jthread consumer([&]() {
            while (!ready.load(SyncPairType::load_order)) {
                std::this_thread::yield();
            }
            if (data != 42) {
                ++errors;
            }
        });

        producer.join();
        consumer.join();
    }
    return errors;
}

// ================================================================
// 3. release/acquire 配对演示 (核心模式)
// ================================================================
void demo_release_acquire_pairing() {
    std::cout << "=== release + acquire 配对演示 ===\n\n";

    std::cout << "  规则: release-store + acquire-load 构成同步关系\n";
    std::cout << "  效果: acquire 之后的所有操作都能看到 release 之前的写入\n\n";

    const int kRounds = 5000;

    auto errors_relaxed  = test_sync_pair<RelaxedPair>(kRounds);
    auto errors_acqrel   = test_sync_pair<AcqRelPair>(kRounds);
    auto errors_seqcst   = test_sync_pair<SeqCstPair>(kRounds);
    auto errors_consume  = test_sync_pair<ConsumePair>(kRounds);
    // 错误用法: release 存但 relaxed 读 (没有配对的 acquire)
    auto errors_wrong    = test_sync_pair<RelaxedStoreOnly>(kRounds);

    std::cout << "  每轮测试 " << kRounds << " 次，期望 data=42:\n\n";

    std::cout << std::left << std::setw(40) << "  内存序对"
              << std::right << std::setw(10) << "错误次数" << "\n";
    std::cout << "  " << std::string(50, '-') << "\n";

    auto print_result = [](const std::string& label, int errors) {
        std::cout << std::left << std::setw(40) << ("  " + label)
                  << std::right << std::setw(10) << errors;
        if (errors == 0)
            std::cout << "  ✓ 正确";
        else
            std::cout << "  ✗ 可能失败";
        std::cout << "\n";
    };

    print_result("relaxed + relaxed (无同步)", errors_relaxed);
    print_result("release + acquire (正确)", errors_acqrel);
    print_result("seq_cst + seq_cst (正确)", errors_seqcst);
    print_result("release + consume (退化为acquire)", errors_consume);
    print_result("release + relaxed (错误用法)", errors_wrong);

    std::cout << "\n  关键发现:\n";
    std::cout << "    1. release 必须配 acquire 或 seq_cst 读才有效\n";
    std::cout << "    2. release + relaxed 是没有同步的 (不保证可见性)\n";
    std::cout << "    3. seq_cst 不仅同步，还保证全局顺序\n";
}

// ================================================================
// 4. acq_rel 演示 (RMW 操作)
// ================================================================
void demo_acq_rel() {
    std::cout << "\n=== acq_rel (Read-Modify-Write 操作的内存序) ===\n\n";

    std::cout << "  acq_rel 用于 RMW 操作 (fetch_add, exchange, CAS):\n";
    std::cout << "    - 读部分使用 acquire 语义\n";
    std::cout << "    - 写部分使用 release 语义\n";
    std::cout << "    - 保证整个 RMW 操作的原子性\n\n";

    // 演示: 用 fetch_add(acq_rel) 实现引用计数
    std::atomic<int> refcount{1};
    std::atomic<bool> released{false};
    int resource_data = 100;

    std::jthread t2([&]() {
        // acquire 部分: 能看到 resource_data
        int old = refcount.fetch_sub(1, std::memory_order_acq_rel);
        if (old == 1) {
            // 最后一个引用, 释放资源
            resource_data = 0; // release 部分保证之前的操作完成
            released.store(true, std::memory_order_release);
        }
    });

    std::jthread t3([&]() {
        // 另一个持有者: 安全读取
        while (!released.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        // acq_rel 保证能看到 resource_data 的最终状态
        assert(resource_data == 0 && "acq_rel 保证可见性");
    });

    t2.join();
    t3.join();

    std::cout << "  引用计数测试通过! resource_data = "
              << resource_data << " (期望 0)\n";
}

// ================================================================
// 5. 可视化: 每种内存序的"栅栏"范围
// ================================================================
void demo_barrier_visualization() {
    std::cout << "\n=== 内存屏障可视化 ===\n\n";

    auto print_barrier = [](const std::string& name,
                             const std::string& before,
                             const std::string& after) {
        std::cout << "  " << name << ":\n";
        std::cout << "    前操作 " << before << " [屏障] " << after << " 后操作\n\n";
    };

    print_barrier("relaxed",   "可跨越 --->", "<--- 可跨越");
    print_barrier("acquire",   "可跨越 --->", "<--- 不可跨越");
    print_barrier("release",   "不可跨越 -->", "<--- 可跨越");
    print_barrier("acq_rel",   "不可跨越 ->", "<- 不可跨越");
    print_barrier("seq_cst",   "← 不可跨越 →", "← 不可跨越 →");

    std::cout << "  记忆口诀:\n";
    std::cout << "    relaxed:  \"随你便，我不管顺序\"\n";
    std::cout << "    acquire:  \"后面的别往前跑\" (读屏障)\n";
    std::cout << "    release:  \"前面的别往后跑\" (写屏障)\n";
    std::cout << "    acq_rel:  \"前后都不能跑\" (读写屏障)\n";
    std::cout << "    seq_cst:  \"所有人都按我说的顺序来\" (全局屏障)\n";
}

// ================================================================
// main
// ================================================================
int main() {
    print_cheatsheet();
    demo_release_acquire_pairing();
    demo_acq_rel();
    demo_barrier_visualization();

    std::cout << "\n================================================\n";
    std::cout << "选择指南:\n";
    std::cout << "  计数器/统计     → relaxed\n";
    std::cout << "  生产者-消费者   → release + acquire\n";
    std::cout << "  读-改-写操作    → acq_rel\n";
    std::cout << "  不确定用什么    → seq_cst (默认，最安全)\n";
    std::cout << "  (永远不要用)    → consume\n";
    std::cout << "================================================\n";

    return 0;
}
