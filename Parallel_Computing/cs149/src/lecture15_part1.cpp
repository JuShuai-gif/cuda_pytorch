// lecture15_part1.cpp — CS149 第15讲：SC vs TSO vs PSO 内存一致性模型
// ============================================================================
// 【课程核心概念】
// 本文件通过 Dekker 模式的 litmus test（一致性检验测试）来演示不同内存一致性模型
// 的行为差异。内存一致性模型定义了多处理器系统中内存操作的可见性顺序。
//
// 【Coherence（一致性） vs Consistency（一致性模型）】
// 这两个概念经常被混淆，但有明确区别：
//   - Cache Coherence（缓存一致性）：保证对同一地址的读写在所有处理器中有统一的顺序。
//     例如：如果 P0 写 X=1 然后 P1 读 X，P1 必须看到 X=1（不会看到过期的旧值）。
//     这是"单地址"的保证，由 MSI/MESI 等协议实现。
//   - Memory Consistency（内存一致性模型）：定义不同地址上的读写操作在所有处理器中
//     如何交织（interleaving）。例如：P0 写 X=1 后写 Y=1，P1 可能在看到 Y=1 后
//     读到的 X 仍是旧值 0。这是"跨地址"的保证，由硬件内存模型定义。
//
// 【四种内存操作排序约束】
// 内存一致性模型通过对以下四种排序提供不同程度的保证来区分：
//   W→R：写必须在后续读之前完成（Write-to-Read ordering）
//   R→R：读必须在后续读之前完成（Read-to-Read ordering）
//   R→W：读必须在后续写之前完成（Read-to-Write ordering）
//   W→W：写必须在后续写之前完成（Write-to-Write ordering）
//
// 【四种主要一致性模型】
//   SC (Sequential Consistency): 所有四种排序都保证 —— 最直观但最慢
//   TSO (Total Store Order):     放松 W→R（允许读超越缓存中的写）—— x86 模型
//   PSO (Partial Store Order):   放松 W→R 和 W→W —— SPARC PSO
//   WO/RC (Weak/Release):        放松全部四种 —— ARM, PowerPC
// ============================================================================
// 编译：g++ -std=c++17 -O2 -pthread lecture15_part1.cpp -o lecture15_part1
// 运行：./lecture15_part1

#include <iostream>
#include <thread>
#include <atomic>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cassert>

// ============================================================================
// 第一部分：内存一致性模型的概念解释
// ============================================================================

void explain_consistency_models() {
    std::cout << "=== CS149 第15讲：内存一致性模型 ===" << std::endl;
    std::cout << std::endl;
    std::cout << "缓存一致性（Coherence, 单地址约束）:" << std::endl;
    std::cout << "  所有处理器对同"一"地址的读写顺序达成一致。" << std::endl;
    std::cout << "  例如：P0 写 X=5 后，P1 读 X 一定看到 5（而非旧值）。" << std::endl;
    std::cout << std::endl;
    std::cout << "内存一致性模型（Consistency, 跨地址约束）:" << std::endl;
    std::cout << "  对地址 X 的写，何时对地址 Y 的读/写变得可见？" << std::endl;
    std::cout << "  例如：P0 先写 X=5 再写 Y=1，P1 看到 Y=1 后读 X，该看到 5 吗？" << std::endl;
    std::cout << std::endl;
    std::cout << "四种内存操作排序约束:" << std::endl;
    std::cout << "  W→R: 写操作必须在后续读操作之前完成" << std::endl;
    std::cout << "  R→R: 读操作必须在后续读操作之前完成" << std::endl;
    std::cout << "  R→W: 读操作必须在后续写操作之前完成" << std::endl;
    std::cout << "  W→W: 写操作必须在后续写操作之前完成" << std::endl;
    std::cout << std::endl;
    std::cout << "各一致性模型放松的排序约束:" << std::endl;
    std::cout << std::endl;
    std::cout << std::left
              << std::setw(24) << "模型"
              << std::setw(12) << "W→R"
              << std::setw(12) << "R→R"
              << std::setw(12) << "R→W"
              << std::setw(12) << "W→W"
              << "典型平台" << std::endl;
    std::cout << std::string(76, '-') << std::endl;
    std::cout << std::left
              << std::setw(24) << "SC (Sequential-顺序一致性)"
              << std::setw(12) << "✓"
              << std::setw(12) << "✓"
              << std::setw(12) << "✓"
              << std::setw(12) << "✓"
              << "理论模型; 最慢" << std::endl;
    std::cout << std::left
              << std::setw(24) << "TSO (Total Store Order)"
              << std::setw(12) << "✗"
              << std::setw(12) << "✓"
              << std::setw(12) << "✓"
              << std::setw(12) << "✓"
              << "x86, SPARC" << std::endl;
    std::cout << std::left
              << std::setw(24) << "PSO (Partial Store Order)"
              << std::setw(12) << "✗"
              << std::setw(12) << "✓"
              << std::setw(12) << "✓"
              << std::setw(12) << "✗"
              << "SPARC PSO 模式" << std::endl;
    std::cout << std::left
              << std::setw(24) << "WO/RC (Weak/Release)"
              << std::setw(12) << "✗"
              << std::setw(12) << "✗"
              << std::setw(12) << "✗"
              << std::setw(12) << "✗"
              << "ARM, PowerPC" << std::endl;
}

// ============================================================================
// 第二部分：Dekker 模式 —— 经典的一致性检验测试
//
// 【Dekker 模式（又称 Store Buffering / SB）】
// 这是验证内存一致性模型的最经典 litmus test（一致性检验测试）。
//
// 初始状态: A = 0, B = 0
//
// 线程 0:              线程 1:
//   A = 1;              B = 1;
//   r1 = B;             r2 = A;
//
// 【可能的结果】:
//   SC:     (r1,r2) = (0,1) 或 (1,0) 或 (1,1) — 绝不可能 (0,0)
//   TSO/x86:(r1,r2) 可以出现 (0,0) — 每个线程的读可能超越自己 write buffer 中的写
//
// 【为什么 SC 下 (r1=0, r2=0) 不可能？（happens-before 证明）】
//   假设 r1=0，意味着线程 0 对 B 的读发生在线程 1 对 B 的写之前（coherence 顺序）。
//   由程序顺序（program order），线程 0 对 A 的写在读 B 之前 → 线程 0 的 A 写
//   发生在线程 1 的 B 写之前。线程 1 的程序顺序要求写 B 在读 A 之前。
//   通过传递闭包：线程 0 的 A 写 → 线程 1 的 B 写 → 线程 1 的 A 读。
//   因此线程 1 的读 A 应该看到 1 → r2=1，与 r2=0 矛盾！
//   happens-before 图中形成了环 → (0,0) 不可达。
//
// 【为什么 TSO/x86 下 (0,0) 可能？】
//   x86 的每个核心有自己的 write buffer（存储缓冲区）。
//   线程 0 的 A=1 被放入 write buffer，然后立即执行 r1=B（不等待 A=1 完成）。
//   同时线程 1 的 B=1 也被放入 write buffer，然后立即执行 r2=A。
//   两者都读到了对方变量在 write buffer 之外的旧值 0 → (0,0)！
// ============================================================================

void dekker_sc(int num_trials) {
    // 使用 seq_cst 内存序的 atomic —— 保证顺序一致性
    std::atomic<int> A{0}, B{0};
    int sc_violations = 0;  // SC 违规计数（理论上应为 0）

    for (int trial = 0; trial < num_trials; ++trial) {
        A.store(0, std::memory_order_seq_cst);
        B.store(0, std::memory_order_seq_cst);

        int r1 = -1, r2 = -1;
        std::thread t0([&]() {
            A.store(1, std::memory_order_seq_cst);        // (1) 写 A=1
            r1 = B.load(std::memory_order_seq_cst);       // (2) 读 B
        });
        std::thread t1([&]() {
            B.store(1, std::memory_order_seq_cst);        // (3) 写 B=1
            r2 = A.load(std::memory_order_seq_cst);       // (4) 读 A
        });
        t0.join(); t1.join();

        if (r1 == 0 && r2 == 0)    // 检测 (0,0) —— 在 SC 下应不可能
            sc_violations++;
    }

    std::cout << "Dekker 模式测试（seq_cst 原子操作 — 顺序一致性）:" << std::endl;
    std::cout << "  测试次数: " << num_trials << std::endl;
    std::cout << "  r1=0 && r2=0 (SC 违规) 出现次数: " << sc_violations;
    if (sc_violations == 0)
        std::cout << " — SC 保证成立（从未观察到(0,0)）" << std::endl;
    else
        std::cout << " — 异常！理论上 SC 不应出现此结果！" << std::endl;
}

void dekker_relaxed(int num_trials) {
    // 使用 relaxed 内存序的 atomic —— 允许重排序
    std::atomic<int> A{0}, B{0};
    int zero_zero = 0;

    for (int trial = 0; trial < num_trials; ++trial) {
        A.store(0, std::memory_order_relaxed);
        B.store(0, std::memory_order_relaxed);

        int r1 = -1, r2 = -1;
        std::thread t0([&]() {
            A.store(1, std::memory_order_relaxed);        // (1) 写 A=1（可能停留在 write buffer）
            r1 = B.load(std::memory_order_relaxed);       // (2) 读 B（可能发生在 A=1 可见之前）
        });
        std::thread t1([&]() {
            B.store(1, std::memory_order_relaxed);        // (3) 写 B=1
            r2 = A.load(std::memory_order_relaxed);       // (4) 读 A
        });
        t0.join(); t1.join();

        if (r1 == 0 && r2 == 0)    // 在 relaxed 下此结果常见！
            zero_zero++;
    }

    std::cout << "Dekker 模式测试（relaxed 原子操作 — 放松排序）:" << std::endl;
    std::cout << "  测试次数: " << num_trials << std::endl;
    std::cout << "  r1=0 && r2=0 (SC 违规，但 TSO/relaxed 允许) 出现次数: " << zero_zero
              << " (" << std::fixed << std::setprecision(1)
              << (100.0 * zero_zero / num_trials) << "%)" << std::endl;
    std::cout << "  在 SC 下此结果不可能，但在 TSO/relaxed 下可以观察到！" << std::endl;
}

// ============================================================================
// 第三部分：Write Buffer 模拟（概念层面）
//
// 【TSO（Total Store Order）动机：write buffer】
// 在 SC 下，一次写操作可能需要数百个周期（缓存缺失、一致性流量）。
// SC 要求处理器必须"停止等待"直到该写操作全局可见后，才能发出下一条指令。
// 这严重限制了流水线效率和 ILP（指令级并行）。
//
// 【Write Buffer 解决方案】
// 每个处理器核心配备一个 write buffer（存储缓冲区，FIFO 队列）：
//   1. 写操作被放入 write buffer（快速，约 1 周期）
//   2. 处理器可以立即执行下一条指令（包括读操作）
//   3. write buffer 在后台异步地将数据排空到缓存/主存
//
// 【为什么这导致 (r1=0, r2=0)？】
// 每个处理器的写仍在 write buffer 中（尚未全局可见）时，
// 读操作访问的是缓存（而非 write buffer 中其他核心的写）。
// 因此线程 0 读 B 时看到旧值 0，线程 1 读 A 时也看到旧值 0。
//
// 【Store-to-Load Forwarding（存储到加载转发）】
// 每个核心自己的读会检查 write buffer，所以线程 0 读自己的 A 会看到 1。
// x86 对所有核心提供 TSO-like 模型（规范不完全严格）。
// ============================================================================

void explain_write_buffer() {
    std::cout << std::endl;
    std::cout << "=== Write Buffer（TSO 机制的动机） ===" << std::endl;
    std::cout << std::endl;
    std::cout << "SC（顺序一致性）的问题: 一次写入可能需要数百个周期" << std::endl;
    std::cout << "（缓存缺失、一致性流量等开销）。在 SC 下，处理器必须" << std::endl;
    std::cout << "暂停（stall）直到该写入完成，才能发出下一条指令。" << std::endl;
    std::cout << std::endl;
    std::cout << "Write Buffer（存储缓冲区）的解决方案:" << std::endl;
    std::cout << "  store A=1 → 放入 write buffer（快速，约 1 个周期）" << std::endl;
    std::cout << "  load B    → 从缓存读取（不需要等 A 的写完成！）" << std::endl;
    std::cout << "  write buffer 在后台异步排空到缓存/主存" << std::endl;
    std::cout << std::endl;
    std::cout << "这就是为什么 r1=r2=0 可能发生: 每个处理器在对方的写" << std::endl;
    std::cout << "离开 write buffer 之前就读取了对方的变量。" << std::endl;
    std::cout << std::endl;
    std::cout << "所有现代处理器都使用 write buffer！" << std::endl;
    std::cout << "  x86: 接近 TSO 模型（规范不完全严格定义）" << std::endl;
    std::cout << "  ARM: 非常放松的模型（比 TSO 更弱）" << std::endl;
}

// ============================================================================
// 第四部分：Store Buffer — 用普通（非原子）变量做 Dekker 测试（x86 行为）
//
// 在 x86 上，即使是普通（非 atomic）变量也会表现出 TSO 行为：
// 编译器可以做重排序，而硬件层面也一定重排序 W→R。
// 使用 volatile 防止编译器重排序，但不能防止硬件重排序。
// ============================================================================

void dekker_plain_vars(int num_trials) {
    // volatile 防止编译器将变量优化到寄存器中或重排序编译后的指令
    // 但无法防止 CPU 硬件层面的重排序（如 write buffer）
    volatile int A = 0, B = 0;
    int zero_zero = 0;

    for (int trial = 0; trial < num_trials; ++trial) {
        A = 0; B = 0;

        int r1 = -1, r2 = -1;
        std::thread t0([&]() {
            A = 1;       // 写入 write buffer（x86 TSO 行为）
            r1 = B;      // 可能读到旧值 0（A=1 仍在 buffer 中）
        });
        std::thread t1([&]() {
            B = 1;       // 写入 write buffer
            r2 = A;      // 可能读到旧值 0
        });
        t0.join(); t1.join();

        if (r1 == 0 && r2 == 0)
            zero_zero++;
    }

    std::cout << "Dekker 模式测试（普通 volatile int，x86 TSO 行为）:" << std::endl;
    std::cout << "  测试次数: " << num_trials << std::endl;
    std::cout << "  r1=0 && r2=0 出现次数: " << zero_zero
              << " (" << std::fixed << std::setprecision(1)
              << (100.0 * zero_zero / num_trials) << "%)" << std::endl;
    std::cout << "  由于 store buffer / TSO 机制，(0,0) 在 x86 上是可以发生的！" << std::endl;
}

// ============================================================================
// 第五部分：Store Forwarding（存储转发，TSO 的重要细节）
//
// 【Store-to-Load Forwarding（存储到加载转发）】
// 在 TSO 下，处理器自己的读操作可以"看穿"write buffer：
// 如果一个读操作的地址在 write buffer 中有匹配的写入，
// 硬件会直接从 write buffer 返回该值（而非从缓存读取）。
// 这保证了同一线程内"先写后读"的直觉语义（写对自身立即可见）。
//
// 示例:
//   线程 0:                    线程 1:
//     X = 1;   (在 write buffer 中)  r1 = X;  (可能看到旧值 0)
//     r2 = X;  (从 write buffer 读到 1)   r2 = Y;
//
//   线程 0 立即看到自己的写（store forwarding），
//   但线程 1 可能还没看到（该写仍在 write buffer 中）。
// ============================================================================

void demo_store_forwarding() {
    std::cout << std::endl;
    std::cout << "=== Store-to-Load Forwarding（存储转发, TSO 特性） ===" << std::endl;
    std::cout << std::endl;

    std::atomic<int> X{0}, Y{0};
    int seen_stale = 0;
    int trials = 100000;

    for (int t = 0; t < trials; ++t) {
        X.store(0, std::memory_order_relaxed);
        Y.store(0, std::memory_order_relaxed);

        int r1 = -1, r2 = -1, r3 = -1;

        std::thread t0([&]() {
            X.store(1, std::memory_order_relaxed);   // 写入到 write buffer
            r1 = X.load(std::memory_order_relaxed);   // store forwarding → 读到 1
            r2 = Y.load(std::memory_order_relaxed);   // 读取 Y
        });
        std::thread t1([&]() {
            Y.store(1, std::memory_order_relaxed);
            r3 = X.load(std::memory_order_relaxed);   // 可能读到 0 (仍在 t0 的 buffer 中)
        });
        t0.join(); t1.join();

        // r1==1 表示 store forwarding 生效（读到了自己的写）
        // r3==0 表示线程 1 读到了旧值（t0 的写尚未全局可见）
        if (r3 == 0 && r1 == 1)
            seen_stale++;
    }

    std::cout << "线程 0 写 X=1，然后立即读 X（通过 store forwarding 读到 1）" << std::endl;
    std::cout << "线程 1 读 X —— 可能读到过期的 0（仍在 t0 的 write buffer 中）" << std::endl;
    std::cout << "  测试次数: " << trials << std::endl;
    std::cout << "  T1 读到过期值 X=0 的次数: " << seen_stale
              << " (" << std::fixed << std::setprecision(1)
              << (100.0 * seen_stale / trials) << "%)" << std::endl;
    std::cout << "  这说明：写对自己的线程立即可见（store forwarding），" << std::endl;
    std::cout << "  但对其他线程的可见有延迟（write buffer 未排空）。" << std::endl;
}

// ============================================================================
// 第六部分：PSO 示例 —— flag-before-data 危害
//
// 在 PSO（Partial Store Order）下，写-写（W→W）排序约束也被放松了。
// 这意味着一个"标志"变量可能在"数据"变量之前变得可见。
//
//   线程 0 (P0):              线程 1 (P1):
//     A = 1;   (数据)          while (flag == 0);
//     flag = 1; (标志)          print A;  ← 可能打印 0！
//
// 在 PSO 下，flag 的写可能比 A 的写先从 write buffer 排空。
// P1 看到 flag=1 时，A 可能仍是 0 —— 这是典型的"flag-before-data"危害。
//
// 【修复方法】使用 release/acquire 语义：
//   flag.store(1, memory_order_release)  —— 保证 A=1 在 flag=1 之前可见
//   flag.load(memory_order_acquire)       —— 保证读 A 在 flag 之后发生
// ============================================================================

void demo_pso_flag_data() {
    std::cout << std::endl;
    std::cout << "=== PSO（Partial Store Order）: Flag-Before-Data（标志先于数据）危害 ===" << std::endl;
    std::cout << std::endl;

    std::atomic<int> A{0}, flag{0};
    int stale_reads = 0;
    int trials = 100000;

    for (int t = 0; t < trials; ++t) {
        A.store(0, std::memory_order_relaxed);
        flag.store(0, std::memory_order_relaxed);

        std::thread t0([&]() {
            // PSO 下这两个写可能被重排序！
            A.store(1, std::memory_order_relaxed);       // 数据写入
            flag.store(1, std::memory_order_relaxed);     // 标志写入（可能先完成！）
        });
        std::thread t1([&]() {
            while (flag.load(std::memory_order_relaxed) == 0)
                ;  // 自旋等待 flag 变为 1
            if (A.load(std::memory_order_relaxed) == 0)
                stale_reads++;  // 看到了 flag=1 但数据仍是 0！
        });
        t0.join(); t1.join();
    }

    std::cout << "线程 0: 先写 A=1 (数据)，再写 flag=1 (标志)。" << std::endl;
    std::cout << "线程 1: 自旋等待 flag，然后读 A。" << std::endl;
    std::cout << "  测试次数: " << trials << std::endl;
    std::cout << "  P1 看到 flag=1 但 A=0 的次数: " << stale_reads
              << " (" << std::fixed << std::setprecision(1)
              << (100.0 * stale_reads / trials) << "%)" << std::endl;
    std::cout << "  修复方法: 对 flag 使用 release store（写端）和 acquire load（读端）。" << std::endl;
    std::cout << "  即 flag.store(1, release) 和 flag.load(acquire)，保证正确的 happens-before。" << std::endl;
}

// ============================================================================
// 第七部分：Synchronized = Data-Race-Free = SC
//
// 【DRF (Data-Race-Free) 程序的 SC 保证】
// C++11 和 Java 内存模型的一个核心定理：
//   如果程序是 data-race-free（无数据竞争），即所有共享内存的冲突访问
//   都通过同步操作（锁、原子变量、屏障等）排序，那么该程序的执行效果
//   等价于在顺序一致性（SC）下运行！
//
// 这意味着：应用程序员不需要手动处理复杂的内存排序问题，
// 只需正确使用同步原语（mutex, atomic, barrier），
// 编译器/运行时/硬件会自动保证正确的内存可见性顺序。
//
// 【数据竞争（Data Race）的定义】
// 满足以下三个条件即构成数据竞争：
//   1. 两个（或更多）操作访问同一内存位置
//   2. 其中至少一个是写操作
//   3. 这些操作之间没有通过同步（fence、lock、原子操作等）建立 happens-before 关系
//
// 在 C++11 中，存在数据竞争的程序的行为是未定义的（undefined behavior）。
// ============================================================================

void demo_synchronized_is_sc() {
    std::cout << std::endl;
    std::cout << "=== DRF（Data-Race-Free, 无数据竞争）程序的 SC 保证 ===" << std::endl;
    std::cout << std::endl;
    std::cout << "C++11 / Java 内存模型的核心保证:" << std::endl;
    std::cout << "  如果程序是 data-race-free（所有对共享内存的冲突访问" << std::endl;
    std::cout << "  都通过同步操作排序），那么程序的行为等价于在" << std::endl;
    std::cout << "  顺序一致性（SC）下执行。" << std::endl;
    std::cout << std::endl;
    std::cout << "这意味着库代码（锁、屏障、原子操作）帮你处理了" << std::endl;
    std::cout << "所有复杂的内存排序问题！应用程序员只需正确使用同步原语。" << std::endl;
    std::cout << std::endl;
    std::cout << "数据竞争（Data Race）的定义:" << std::endl;
    std::cout << "  两个操作访问同一内存位置," << std::endl;
    std::cout << "  其中至少一个是写操作," << std::endl;
    std::cout << "  且这些操作没有通过同步（fence、lock 等）建立 happens-before 关系。" << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    explain_consistency_models();
    std::cout << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::endl;

    // Dekker 模式实验
    int n = 1000000;    // 100 万次试验

    dekker_sc(n);       // SC 下 (0,0) 应不可能
    std::cout << std::endl;
    dekker_relaxed(n);  // relaxed 下 (0,0) 可能出现
    std::cout << std::endl;

    explain_write_buffer();
    std::cout << std::endl;

    dekker_plain_vars(n / 100);   // volatile 测试次数更少（较慢）
    std::cout << std::endl;

    demo_store_forwarding();      // 存储转发演示
    std::cout << std::endl;

    demo_pso_flag_data();         // PSO flag-before-data 危害
    std::cout << std::endl;

    demo_synchronized_is_sc();    // DRF=SC 定理

    std::cout << std::endl;
    std::cout << "=== 总结 ===" << std::endl;
    std::cout << "1. SC（顺序一致性）: 保持全部 4 种排序 —— 直观理解但性能代价大。" << std::endl;
    std::cout << "2. TSO（完全存储排序）: 放松 W→R（允许 write buffer）—— x86 采用的模型。" << std::endl;
    std::cout << "3. PSO（部分存储排序）: 还放松 W→W —— 可能导致 flag-before-data 危害。" << std::endl;
    std::cout << "4. DRF 程序自动获得 SC 等价行为 —— 正确使用同步原语即可！" << std::endl;

    return 0;
}
