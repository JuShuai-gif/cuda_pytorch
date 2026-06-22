// 05_barrier.cpp — std::barrier 可重用同步屏障
// 演示: 多阶段同步、完成回调、arrive_and_drop

#include <barrier>
#include <chrono>
#include <iostream>
#include <syncstream>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// ===== 1. 基础: 多线程多阶段同步 =====
void demo_basic_barrier() {
    std::cout << "=== 1. 多阶段同步 ===\n";

    const int kWorkers = 4;
    const int kPhases = 3;

    // 完成回调: 每个阶段结束时自动调用
    int phase_counter = 0;
    std::barrier sync(kWorkers, [&]() noexcept {
        ++phase_counter;
        std::osyncstream(std::cout)
            << "  >>> 阶段 " << phase_counter << " 完成 <<<\n";
    });

    auto worker = [&](int id) {
        for (int p = 0; p < kPhases; ++p) {
            // 模拟该阶段的工作
            std::this_thread::sleep_for(10ms * (id + 1));
            std::osyncstream(std::cout)
                << "    Worker " << id << " 完成阶段 " << p << "\n";

            sync.arrive_and_wait(); // 等待所有 worker 完成当前阶段
        }
    };

    std::vector<std::jthread> threads;
    for (int i = 0; i < kWorkers; ++i) {
        threads.emplace_back(worker, i);
    }
    threads.clear();

    std::cout << "  完成 " << phase_counter << " 个阶段\n";
}

// ===== 2. arrive_and_drop: 减少参与线程数 =====
void demo_arrive_and_drop() {
    std::cout << "\n=== 2. arrive_and_drop ===\n";

    const int kInitial = 4;
    std::barrier sync(kInitial, []() noexcept {
        std::osyncstream(std::cout) << "  >>> 屏障打开 <<<\n";
    });

    auto worker = [&](int id, bool drop_early) {
        std::osyncstream(std::cout) << "    Worker " << id << " 到达\n";

        if (drop_early) {
            sync.arrive_and_drop(); // 永久离开
            std::osyncstream(std::cout)
                << "    Worker " << id << " 永久离开 barrier\n";
        } else {
            sync.arrive_and_wait();
            std::osyncstream(std::cout)
                << "    Worker " << id << " 通过屏障\n";
        }
    };

    std::vector<std::jthread> threads;
    threads.emplace_back(worker, 0, true);  // 提前离开
    threads.emplace_back(worker, 1, true);  // 提前离开
    // worker 2 和 3 成为剩下的参与者
    threads.emplace_back(worker, 2, false);
    threads.emplace_back(worker, 3, false);
    threads.clear();
}

// ===== 3. 并行归约模拟 (分阶段) =====
void demo_parallel_reduce_phases() {
    std::cout << "\n=== 3. 并行归约 (barrier 同步) ===\n";

    const int kThreads = 4;
    const int kDataSize = 16;
    std::vector<int> data(kDataSize);
    for (int i = 0; i < kDataSize; ++i) data[i] = i + 1;

    // 每线程的局部和
    std::vector<int> local_sums(kThreads, 0);

    int phase = 0;
    std::barrier sync(kThreads, [&]() noexcept {
        ++phase;
        if (phase == 1) {
            std::osyncstream(std::cout) << "  阶段1: 局部求和完成\n";
        }
    });

    auto worker = [&](int tid) {
        // 阶段1: 局部求和
        int chunk = kDataSize / kThreads;
        int start = tid * chunk;
        int end = (tid == kThreads - 1) ? kDataSize : start + chunk;
        for (int i = start; i < end; ++i) {
            local_sums[tid] += data[i];
        }
        sync.arrive_and_wait();

        // 阶段2: 主线程归约 (简化: 所有线程打印后 barrier)
        if (tid == 0) {
            int total = 0;
            for (int s : local_sums) total += s;
            std::osyncstream(std::cout)
                << "  阶段2: 总和 = " << total
                << " (期望 " << kDataSize * (kDataSize + 1) / 2 << ")\n";
        }
    };

    std::vector<std::jthread> threads;
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back(worker, i);
    }
    threads.clear();
}

int main() {
    demo_basic_barrier();
    demo_arrive_and_drop();
    demo_parallel_reduce_phases();

    std::cout << "\nbarrier 是多阶段并行算法的核心同步工具。\n";
    return 0;
}
