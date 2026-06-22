// lecture15_part2.cpp — CS149 第15讲：C++11 原子操作和内存排序
// ============================================================================
// 【课程核心概念】
// 本文件深入演示 C++11 内存模型中的各种 memory_order 选项及其应用场景。
// C++11 引入了一套精确定义的内存排序语义，使得程序员可以在正确性和性能之间
// 做出精细的权衡。
//
// 【C++11 五种 memory_order 详解】
//
// memory_order_relaxed（最弱，最快）:
//   仅保证操作的原子性（不可分割），不提供任何跨线程的排序保证。
//   编译器/CPU 可以自由重排序周围的非原子和原子操作。
//   适用场景：简单的原子计数器、统计信息收集（不需要与其他变量同步）。
//
// memory_order_acquire（加载专用）:
//   保证此加载之后的任何读/写操作不会被重排序到此加载之前。
//   效果："获取"了 release 线程在此之前发布的所有写入的可见性。
//   必须与 release 配对使用。
//
// memory_order_release（存储专用）:
//   保证此存储之前的任何读/写操作不会被重排序到此存储之后。
//   效果："释放"了当前线程之前的所有写入，使其对后续的 acquire 可见。
//   必须与 acquire 配对使用。
//
// memory_order_acq_rel（RMW 专用）:
//   同时具有 acquire 和 release 语义。
//   仅用于 Read-Modify-Write 操作（fetch_add, compare_exchange 等）。
//   RMW 操作的 load 部分有 acquire 语义，store 部分有 release 语义。
//
// memory_order_seq_cst（最强，默认）:
//   顺序一致性（Sequential Consistency）。
//   所有 seq_cst 操作在所有线程中有一个单一的全局全序（total order）。
//   这是 std::atomic 的默认内存序，也是最强（最昂贵）的保证。
//   在 ARM 上需要显式的 DMB（Data Memory Barrier）指令，开销较大。
//
// 【核心概念：synchronizes-with（同步关系）】
//   当线程 A 执行 release 存储，线程 B 执行 acquire 加载（同一原子变量），
//   且 B 的加载读到了 A 的存储值时：
//     → A 的 release store synchronizes-with B 的 acquire load
//     → A 在 release 之前的所有内存写入对 B 在 acquire 之后的所有操作立即可见
//   这是无锁编程（lock-free programming）的基础构建块。
// ============================================================================
// 编译：g++ -std=c++17 -O2 -pthread lecture15_part2.cpp -o lecture15_part2
// 运行：./lecture15_part2

#include <iostream>
#include <thread>
#include <atomic>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cassert>
#include <mutex>

// ============================================================================
// C++11 内存排序参考（已在文件头详述，此处省略注释中的重复内容）
// ============================================================================

// ============================================================================
// 第一部分：Relaxed 排序 —— 原子计数器
//
// relaxed 适用于只需要原子性（操作不可分割）而不需要与其他内存操作
// 排序的场景。典型应用：共享计数器的增减、统计信息收集。
//
// 【为什么 relaxed 足够用于计数器？】
// 计数器只需要保证每个 fetch_add 是原子的（不会丢失更新），
// 而不需要关心 fetch_add 与程序中的其他读写之间的相对顺序。
// 这意味着编译器/CPU 可以自由地重排序计数器的更新以最大化性能。
// ============================================================================

void demo_relaxed_counter() {
    std::cout << "=== 第一部分：Relaxed 原子计数器 ===" << std::endl;
    std::cout << std::endl;
    std::cout << "relaxed 排序: 仅保证操作的原子性（不可分割）。" << std::endl;
    std::cout << "不提供与其他内存访问之间的任何排序保证。" << std::endl;
    std::cout << "非常适合简单的计数器、统计信息等场景。" << std::endl;
    std::cout << std::endl;

    std::atomic<long long> counter{0};
    const int N_THREADS = 4;
    const long long N_INC = 25'000'000;   // 每个线程 2500 万次自增

    // 工作函数：每个线程执行 N_INC 次 fetch_add
    auto worker = [&]() {
        for (long long i = 0; i < N_INC; ++i)
            counter.fetch_add(1, std::memory_order_relaxed);
    };

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> threads;
    for (int i = 0; i < N_THREADS; ++i)
        threads.emplace_back(worker);
    for (auto& th : threads) th.join();
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "线程数: " << N_THREADS << ", 每个线程自增次数: " << N_INC << std::endl;
    std::cout << "期望结果: " << (N_THREADS * N_INC) << std::endl;
    std::cout << "实际结果: " << counter.load() << std::endl;
    std::cout << "耗时:     " << std::fixed << std::setprecision(1) << ms << " ms" << std::endl;
}

// ============================================================================
// 第二部分：Acquire-Release —— 消息传递
//
// 【Message Passing（消息传递）范式】
// 这是 acquire/release 最经典的用例：一个线程"生产"数据并设置"就绪"标志，
// 另一个线程自旋等待标志后"消费"数据。
//
// 【关键保证】
// release store（flag=true）保证在其之前的所有内存写入（data=42）
// 对执行 acquire load 的线程立即可见。这建立了"happens-before"关系：
//   data=42  happens-before  flag.store(release)
//   flag.store(release)  synchronizes-with  flag.load(acquire)
//   flag.load(acquire)  happens-before  x=data
//   因此 data=42  happens-before  x=data（传递闭包）
//
// 【为什么比 mutex 轻量？】
// acquire/release 不涉及内核调用、调度、上下文切换。
// 消费者在 flag 为 false 时自旋（busy-waiting）—— 适合等待时间很短的场景。
// 对于长时间等待，应使用 mutex + condition_variable。
// ============================================================================

void demo_acquire_release_message() {
    std::cout << std::endl;
    std::cout << "=== 第二部分：Acquire-Release 消息传递 ===" << std::endl;
    std::cout << std::endl;

    // 共享变量
    // data 是普通非原子变量！但通过 flag 的 acquire/release 语义获得安全保证
    int data = 0;
    std::atomic<bool> flag{false};

    // 生产者线程：写入数据后设置就绪标志
    std::thread producer([&]() {
        data = 42;                                    // (A) 非原子写 —— 被 flag 的 release 保护
        flag.store(true, std::memory_order_release);   // (B) release: 使 (A) 对消费者可见
    });

    // 消费者线程：自旋等待就绪标志后读取数据
    std::thread consumer([&]() {
        while (!flag.load(std::memory_order_acquire))  // (C) acquire: 与 (B) 建立同步关系
            ;                                           // 忙等自旋（适合耗时短的等待）
        int x = data;                                  // (D) 保证读到 42！
        std::cout << "消费者读取到的 data = " << x << "（保证是 42）" << std::endl;
    });

    producer.join();
    consumer.join();

    std::cout << "release store 与 acquire load 之间建立 synchronizes-with 关系:" << std::endl;
    std::cout << "  release 之前的所有写操作对 acquire 之后的所有代码立即可见。" << std::endl;
    std::cout << "  这是无锁编程（lock-free programming）的基本构建块。" << std::endl;
}

// ============================================================================
// 第三部分：Sequential Consistency —— 最强保证
//
// seq_cst 操作在所有线程中有一个单一的全局全序（total order）。
// 这是 std::atomic 的默认内存序（如 atomic<int> x; x.store(1); 等价于 seq_cst）。
// 比 acquire/release 更昂贵，但更容易理解和推理。
//
// 【seq_cst 的额外开销】
// 在 x86 上，大部分内存操作已经是强排序的，但 seq_cst store 可能需要 mfence
// （全屏障）指令，而 acquire release 不需要。ARM 上差别更大，
// seq_cst 需要显式的 DMB（Data Memory Barrier）指令。
// ============================================================================

void demo_seq_cst_ordering() {
    std::cout << std::endl;
    std::cout << "=== 第三部分：Sequential Consistency (seq_cst) ===" << std::endl;
    std::cout << std::endl;

    // 用 seq_cst 跑 Dekker 模式 —— (0,0) 应不可能
    std::atomic<int> X{0}, Y{0};
    std::atomic<int> Z{0};  // 见证变量

    int zero_zero = 0;
    const int N = 500'000;    // 50 万次试验

    for (int trial = 0; trial < N; ++trial) {
        X.store(0, std::memory_order_seq_cst);
        Y.store(0, std::memory_order_seq_cst);

        int r1 = -1, r2 = -1;
        std::thread t0([&]() {
            X.store(1, std::memory_order_seq_cst);      // seq_cst 写
            r1 = Y.load(std::memory_order_seq_cst);      // seq_cst 读
        });
        std::thread t1([&]() {
            Y.store(1, std::memory_order_seq_cst);
            r2 = X.load(std::memory_order_seq_cst);
        });
        t0.join(); t1.join();

        if (r1 == 0 && r2 == 0) zero_zero++;   // 在 seq_cst 下应永不发生
    }

    std::cout << "Dekker 模式测试（seq_cst 内存序, N=" << N << "）" << std::endl;
    std::cout << "  r1=0 && r2=0 出现次数: " << zero_zero;
    if (zero_zero == 0)
        std::cout << " — 在 SC 下不可能观察到，与预期一致。" << std::endl;
    else
        std::cout << " — 在 SC 下出现此结果是异常的！" << std::endl;

    std::cout << std::endl;
    std::cout << "seq_cst 是 std::atomic 的默认内存序。" << std::endl;
    std::cout << "它提供最强的一致性保证，但可能带来更高的开销" << std::endl;
    std::cout << "（尤其在 ARM 架构上，需要显式的 DMB（Data Memory Barrier）指令）。" << std::endl;
}

// ============================================================================
// 第四部分：acq_rel 用于 Read-Modify-Write 操作
//
// Read-Modify-Write（RMW）操作如 compare_exchange、fetch_add 等，
// 在逻辑上包含两个步骤：先读取（load）后写入（store）。
// acq_rel 为 load 阶段提供 acquire 语义，为 store 阶段提供 release 语义。
// 这使得 RMW 操作可以同时参与两个方向的同步关系。
//
// 【常用 RMW 操作】:
//   fetch_add / fetch_sub  : 原子加法 / 减法
//   exchange               : 原子交换
//   compare_exchange_weak  : 弱比较交换（可能伪失败，适合循环中）
//   compare_exchange_strong: 强比较交换（不会伪失败）
// ============================================================================

void demo_rmw_acq_rel() {
    std::cout << std::endl;
    std::cout << "=== 第四部分：acq_rel 用于 Read-Modify-Write 操作 ===" << std::endl;
    std::cout << std::endl;

    std::atomic<int> counter{0};
    std::atomic<bool> ready{false};
    int observed = 0;

    // 生产者：使用 fetch_add 配合 acq_rel
    std::thread producer([&]() {
        // fetch_add 的 store 部分有 release 语义 —— 使消费者能看到递增
        counter.fetch_add(1, std::memory_order_acq_rel);   // 0→1
        counter.fetch_add(1, std::memory_order_acq_rel);   // 1→2
        counter.fetch_add(1, std::memory_order_acq_rel);   // 2→3
    });

    // 消费者：自旋等待计数器达到 3
    std::thread consumer([&]() {
        int prev = counter.load(std::memory_order_acquire);
        while (prev < 3) {
            // compare_exchange_weak 是 RMW —— 用 acq_rel 同时获得两边语义
            // 如果当前值不等于 prev，则更新 prev 为实际值（weak 可能伪失败）
            counter.compare_exchange_weak(prev, prev, std::memory_order_acq_rel);
            prev = counter.load(std::memory_order_acquire);
        }
        observed = counter.load(std::memory_order_acquire);
    });

    producer.join();
    consumer.join();

    std::cout << "fetch_add 配合 acq_rel: 同时具有 acquire 和 release 语义" << std::endl;
    std::cout << "  消费者观察到的 counter = " << observed << "（期望 3）" << std::endl;
    std::cout << "  acq_rel 是 RMW 操作最自然的内存序选择。" << std::endl;
}

// ============================================================================
// 第五部分：内存排序的性能开销对比
//
// 【平台差异】
// x86 架构本身提供较强的排序保证：
//   - 大部分 load 自带 acquire 语义，大部分 store 自带 release 语义
//   - 因此 relaxed/acquire/release/acq_rel 在 x86 上开销相似
//   - seq_cst 可能需要 mfence 指令（全屏障），代价较高
//
// ARM 架构提供较弱的排序保证：
//   - relaxed 不生成任何屏障指令（最快）
//   - acquire 需要 DMB LD 屏障
//   - release 需要 DMB ST 屏障
//   - seq_cst 需要 DMB SY（全屏障），与 acquire/release 差异较大
// ============================================================================

void demo_ordering_cost() {
    std::cout << std::endl;
    std::cout << "=== 第五部分：内存排序的性能开销对比 ===" << std::endl;
    std::cout << std::endl;

    const long long N = 50'000'000;   // 5000 万次操作
    std::atomic<long long> sum{0};

    // 基准测试 lambda：使用指定内存序执行 N 次 fetch_add
    auto bench = [&](std::memory_order mo, const char* name) {
        sum.store(0, std::memory_order_relaxed);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (long long i = 0; i < N; ++i)
            sum.fetch_add(1, mo);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "  " << std::left << std::setw(20) << name
                  << std::fixed << std::setprecision(1) << std::setw(10) << ms << " ms";
        if (mo == std::memory_order_relaxed)
            std::cout << " (基准对照)";
        std::cout << std::endl;
    };

    std::cout << "fetch_add 性能对比 (" << (N / 1'000'000) << "M 次操作, 单线程):" << std::endl;
    bench(std::memory_order_relaxed, "relaxed");
    bench(std::memory_order_acquire, "acquire");   // 对于 RMW 不理想，但合法
    bench(std::memory_order_release, "release");   // 对于 RMW 不理想，但合法
    bench(std::memory_order_acq_rel, "acq_rel");
    bench(std::memory_order_seq_cst, "seq_cst");

    std::cout << std::endl;
    std::cout << "在 x86 上: relaxed、acquire、release 和 acq_rel 的开销相似" << std::endl;
    std::cout << "（因为 x86 本身就提供较强的排序保证）。" << std::endl;
    std::cout << "seq_cst 在 x86 上可能需要 mfence（全屏障）指令，开销较大。" << std::endl;
    std::cout << "在 ARM 上: acquire/release 需要显式屏障指令，各内存序之间差异更大。" << std::endl;
}

// ============================================================================
// 第六部分：C++11 内存排序速查表
// ============================================================================

void print_cheat_sheet() {
    std::cout << std::endl;
    std::cout << "=== C++11 内存排序速查表 ===" << std::endl;
    std::cout << std::endl;
    std::cout << std::left
              << std::setw(18) << "内存序"
              << std::setw(12) << "Load 可用"
              << std::setw(12) << "Store 可用"
              << std::setw(51) << "适用场景" << std::endl;
    std::cout << std::string(93, '-') << std::endl;
    std::cout << std::left
              << std::setw(18) << "relaxed"
              << std::setw(12) << "✓"
              << std::setw(12) << "✓"
              << "计数器、统计信息（不需要排序保证）" << std::endl;
    std::cout << std::left
              << std::setw(18) << "acquire"
              << std::setw(12) << "✓"
              << std::setw(12) << "—"
              << "消费者读取端；与 release store 配对使用" << std::endl;
    std::cout << std::left
              << std::setw(18) << "release"
              << std::setw(12) << "—"
              << std::setw(12) << "✓"
              << "生产者写入端；与 acquire load 配对使用" << std::endl;
    std::cout << std::left
              << std::setw(18) << "acq_rel"
              << std::setw(12) << "—"
              << std::setw(12) << "—"
              << "仅用于 RMW 操作（fetch_add, compare_exchange）" << std::endl;
    std::cout << std::left
              << std::setw(18) << "seq_cst"
              << std::setw(12) << "✓"
              << std::setw(12) << "✓"
              << "顺序一致性（默认值，提供最强保证）" << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    demo_relaxed_counter();
    demo_acquire_release_message();
    demo_seq_cst_ordering();
    demo_rmw_acq_rel();
    demo_ordering_cost();
    print_cheat_sheet();

    std::cout << std::endl;
    std::cout << "=== 核心要点 ===" << std::endl;
    std::cout << "开发建议: 从 seq_cst（默认值）入手，确保正确性。" << std::endl;
    std::cout << "只有在性能分析（profiling）证明内存序是瓶颈后，" << std::endl;
    std::cout << "才将 seq_cst 优化为 acquire/release（或 relaxed）。" << std::endl;
    std::cout << "除非能够证明不需要任何排序，否则不要使用 relaxed。" << std::endl;

    return 0;
}
