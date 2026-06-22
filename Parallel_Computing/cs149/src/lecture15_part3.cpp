// lecture15_part3.cpp — CS149 第15讲：内存屏障（Fence）与数据竞争（Data Race）
// ============================================================================
// 【课程核心概念】
// 本文件深入探讨三个紧密相关的主题：
//
// 1. 内存屏障（Memory Fences / Memory Barriers）:
//    当一致性模型太松（relaxed）而无法满足程序的正确性要求时，
//    "内存屏障"是程序员可以用来强行约束内存访问顺序的工具（escape hatch）。
//    它在指定的程序点上阻止编译器/CPU 对读写操作进行某些类型的重排序。
//
//    x86 硬件屏障指令：
//      mfence: 全屏障 — 所有之前的 load/store 必须在后续的 load/store 之前完成
//      lfence: 加载屏障 — 所有之前的 load 必须在后续的 load 之前完成
//      sfence: 存储屏障 — 所有之前的 store 必须在后续的 store 之前完成
//
//    C++11 等价屏障（std::atomic_thread_fence）：
//      seq_cst fence   ≈ mfence（全屏障）
//      acquire fence    ≈ "load fence"（读端屏障）
//      release fence    ≈ "store fence"（写端屏障）
//
//    【重要】fence 与带 memory_order 的原子操作不同：
//      - atomic.store(1, release): 只约束这个 store 与其他操作的排序
//      - atomic_thread_fence(release): 约束该 fence 前后的所有操作的排序
//      fence 是全局性的约束，而单次原子操作的 memory_order 是局部的。
//
// 2. Happens-Before 分析:
//    happens-before 关系是推理并发程序结果的基本工具。
//    如果 happens-before 图中存在环路，那么对应的执行结果是不可达的。
//
//    三种 happens-before 边:
//      a) 程序顺序（program order, po）: 同一线程中的操作按代码顺序发生
//      b) 同步边（synchronizes-with, sw）: release store → acquire load
//      c) 传递闭包: 如果 A hb B 且 B hb C，则 A hb C
//
// 3. 数据竞争（Data Race）:
//    在 C++11 中，存在数据竞争的程序行为是未定义的（undefined behavior）。
//    数据竞争的三要素:
//      a) 两个（或多个）操作访问同一内存位置
//      b) 其中至少一个是写操作
//      c) 操作之间没有通过同步建立 happens-before 关系
//
// 【DRF（Data-Race-Free）定理】
// 如果程序是 data-race-free，则其行为等价于在顺序一致性（SC）下执行。
// 这就是"SC for DRF"—— C++11 语言标准与程序员之间的契约：
// 程序员保证无数据竞争 → 语言保证 SC 行为。
// ============================================================================
// 编译：g++ -std=c++17 -O2 -pthread lecture15_part3.cpp -o lecture15_part3
// 运行：./lecture15_part3

#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cassert>

// ============================================================================
// 第一部分：内存屏障（Fence/Barrier）详解
// ============================================================================

void explain_fences() {
    std::cout << "=== CS149 第15讲：内存屏障（Fence）与数据竞争（Data Race） ===" << std::endl;
    std::cout << std::endl;
    std::cout << "内存屏障（memory fence/barrier）阻止内存操作的重排序。" << std::endl;
    std::cout << "当一致性模型太松（relaxed）时，内存屏障是程序员的" << std::endl;
    std::cout << ""逃生舱"（escape hatch）—— 在特定位置强制约束内存顺序。" << std::endl;
    std::cout << std::endl;
    std::cout << "x86 硬件屏障指令:" << std::endl;
    std::cout << "  mfence: 全屏障 —— 所有之前的 load+store 在后续 load/store 之前完成" << std::endl;
    std::cout << "  lfence: 加载屏障 —— 所有之前的 load 在后续 load 之前完成" << std::endl;
    std::cout << "  sfence: 存储屏障 —— 所有之前的 store 在后续 store 之前完成" << std::endl;
    std::cout << std::endl;
    std::cout << "C++11 等价屏障（std::atomic_thread_fence）:" << std::endl;
    std::cout << "  std::atomic_thread_fence(std::memory_order_seq_cst);   // 对应 mfence（全屏障）" << std::endl;
    std::cout << "  std::atomic_thread_fence(std::memory_order_acquire);   // 对应"加载屏障"" << std::endl;
    std::cout << "  std::atomic_thread_fence(std::memory_order_release);   // 对应"存储屏障"" << std::endl;
    std::cout << std::endl;
    std::cout << "【fence vs 带 memory_order 的原子操作】的区別:" << std::endl;
    std::cout << "  atomic.store(1, release) 只约束该 store 与周围操作的排序。" << std::endl;
    std::cout << "  atomic_thread_fence(release) 约束 fence 前后所有操作的排序。" << std::endl;
    std::cout << "  fence 是全局性约束，单次操作的 memory_order 是局部约束。" << std::endl;
}

// ============================================================================
// 第二部分：使用 std::atomic_thread_fence 修复 Dekker 模式
//
// 不加屏障：relaxed 原子操作允许重排序 → (r1=0, r2=0) 可能出现。
// 添加 seq_cst 屏障：在 store 和 load 之间插入全屏障，
// 保证 store 在 load 之前全局完成 → 阻止 (0,0) 出现。
//
// 这种用法等价于使用 seq_cst 内存序，但允许更精细的控制——
// 可以选择只对关键点施加屏障约束，其他操作保持 relaxed 以获得更好性能。
// ============================================================================

void demo_fence_fixing_dekker() {
    std::cout << std::endl;
    std::cout << "=== 用内存屏障修复 Dekker 模式 ===" << std::endl;
    std::cout << std::endl;

    std::atomic<int> A{0}, B{0};
    int zero_zero = 0;
    const int N = 500'000;    // 50 万次试验

    for (int trial = 0; trial < N; ++trial) {
        A.store(0, std::memory_order_relaxed);
        B.store(0, std::memory_order_relaxed);

        int r1 = -1, r2 = -1;
        std::thread t0([&]() {
            A.store(1, std::memory_order_relaxed);              // (1) 存储到 A（relaxed）
            // 屏障: 保证 (1) 的所有 store 在后续 load 之前全部完成
            std::atomic_thread_fence(std::memory_order_seq_cst);
            r1 = B.load(std::memory_order_relaxed);              // (2) 从 B 加载（relaxed）
        });
        std::thread t1([&]() {
            B.store(1, std::memory_order_relaxed);              // (3) 存储到 B
            std::atomic_thread_fence(std::memory_order_seq_cst);
            r2 = A.load(std::memory_order_relaxed);              // (4) 从 A 加载
        });
        t0.join(); t1.join();

        if (r1 == 0 && r2 == 0) zero_zero++;   // 加了屏障后应不可能出现
    }

    std::cout << "Dekker 测试: relaxed 原子 + store 与 load 之间插入 seq_cst 屏障:" << std::endl;
    std::cout << "  测试次数: " << N << std::endl;
    std::cout << "  r1=0 && r2=0 出现次数: " << zero_zero;
    if (zero_zero == 0)
        std::cout << " — 屏障成功阻止了重排序！实现了 SC 级别的行为。" << std::endl;
    else
        std::cout << " — 重排序仍然发生了（屏障未能完全阻止）。" << std::endl;
}

// ============================================================================
// 第三部分：Happens-Before 分析
//
// 【Happens-Before 形式化推理】
// 这是分析并发程序可能输出结果的形式化方法。
//
// Dekker 模式示例:
//   线程 0:      线程 1:
//     (1) A = 1    (3) B = 1
//     (2) r1 = B   (4) r2 = A
//
// 要得到 r1=0, r2=0, 需要:
//   r1=0 → (2) 读取了 B 的初始值 → 在 coherence 顺序中 (2) 在 (3) 之前
//   r2=0 → (4) 读取了 A 的初始值 → 在 coherence 顺序中 (4) 在 (1) 之前
//
// happens-before 边:
//   程序顺序:  (1) → (2)   和   (3) → (4)
//   coherence: (2) → (3)   (因为 r1=0, B 的读发生在 B 的写之前)
//   coherence: (4) → (1)   (因为 r2=0, A 的读发生在 A 的写之前)
//
// 环: (1) → (2) → (3) → (4) → (1)  ← 不可达！
// 因此 (r1=0, r2=0) 在 SC 下不可能。
// ============================================================================

void explain_happens_before() {
    std::cout << std::endl;
    std::cout << "=== Happens-Before 分析 ===" << std::endl;
    std::cout << std::endl;
    std::cout << "Dekker 模式: r1=0, r2=0 — 为什么在 SC 下不可能？" << std::endl;
    std::cout << std::endl;
    std::cout << "  线程 0:         线程 1:" << std::endl;
    std::cout << "    (1) A = 1       (3) B = 1" << std::endl;
    std::cout << "    (2) r1 = B      (4) r2 = A" << std::endl;
    std::cout << std::endl;
    std::cout << "要得到 r1=0, r2=0, 必须满足:" << std::endl;
    std::cout << "  (2) 读到 0 → 在 coherence 顺序中，(2) 发生在 (3) 之前" << std::endl;
    std::cout << "  (4) 读到 0 → 在 coherence 顺序中，(4) 发生在 (1) 之前" << std::endl;
    std::cout << std::endl;
    std::cout << "Happens-before 边:" << std::endl;
    std::cout << "  程序顺序 (program order):  (1) → (2)  和  (3) → (4)" << std::endl;
    std::cout << "  coherence 顺序:            (2) → (3)  (r1=0, B 的读在 B 的写之前)" << std::endl;
    std::cout << "  coherence 顺序:            (4) → (1)  (r2=0, A 的读在 A 的写之前)" << std::endl;
    std::cout << std::endl;
    std::cout << "形成环: (1) → (2) → (3) → (4) → (1)  ← 不可达的循环！" << std::endl;
    std::cout << "因此 (r1=0, r2=0) 在 SC 下不可能出现。" << std::endl;
}

// ============================================================================
// 第四部分：数据竞争（Data Race）演示
//
// 一个数据竞争的例子：两个线程并发地对同一个普通 int 变量执行 ++ 操作。
// 由于 ++ 不是原子操作（包含读-改-写三个步骤），
// 两个线程可能同时读、各自加 1、再写回 → 丢失了一次更新。
//
// 修复方法：使用 std::atomic<int> 配合 fetch_add。
// 即使使用 memory_order_relaxed，fetch_add 也是原子的 —— 无丢失更新。
// ============================================================================

void demonstrate_data_race() {
    std::cout << std::endl;
    std::cout << "=== 数据竞争（Data Race） ===" << std::endl;
    std::cout << std::endl;

    // ---- 不安全版本：对普通 int 存在数据竞争 ----
    int shared_counter = 0;
    const int N_INC = 1'000'000;   // 每个线程 100 万次自增
    std::atomic<bool> done{false};

    std::thread t0([&]() {
        for (int i = 0; i < N_INC; ++i)
            shared_counter++;  // 数据竞争: 非原子写 + 并发写
        done.store(true, std::memory_order_release);
    });
    std::thread t1([&]() {
        for (int i = 0; i < N_INC; ++i)
            shared_counter++;  // 数据竞争！
    });
    t0.join(); t1.join();

    std::cout << "数据竞争示例（未同步的普通 int）:" << std::endl;
    std::cout << "  期望值: " << (2 * N_INC) << std::endl;
    std::cout << "  实际值: " << shared_counter << std::endl;
    std::cout << "  由于数据竞争丢失了更新！（在 C++ 中属于未定义行为）" << std::endl;

    // ---- 安全版本：使用 atomic 配合 relaxed —— 无竞争，计数正确 ----
    std::atomic<int> safe_counter{0};
    std::thread t2([&]() {
        for (int i = 0; i < N_INC; ++i)
            safe_counter.fetch_add(1, std::memory_order_relaxed);
    });
    std::thread t3([&]() {
        for (int i = 0; i < N_INC; ++i)
            safe_counter.fetch_add(1, std::memory_order_relaxed);
    });
    t2.join(); t3.join();

    std::cout << std::endl;
    std::cout << "无竞争示例（atomic<int> + relaxed 内存序）:" << std::endl;
    std::cout << "  期望值: " << (2 * N_INC) << std::endl;
    std::cout << "  实际值: " << safe_counter.load() << std::endl;
    std::cout << "  结果正确！原子操作不可分割 —— 不会丢失更新。" << std::endl;
}

// ============================================================================
// 第五部分：冲突访问的分类
//
// 【冲突（Conflict）与数据竞争（Data Race）的区分】
//   冲突访问: 两个线程访问同一内存位置，且至少一个是写操作。
//     → 如果这些冲突访问通过同步（fence、lock、acquire/release 等）排序了，
//       那么程序正确（无数据竞争）。
//     → 如果冲突访问之间没有 happens-before 关系，那么就构成数据竞争。
//   因此：数据竞争 = 未排序的冲突访问。
//
//   在实际开发中：
//     - 大多数程序使用同步库（互斥锁、信号量、屏障等），自然避免了数据竞争。
//     - 直接裸访问共享变量而不加同步 → 几乎一定是 bug！
// ============================================================================

void classify_conflicts() {
    std::cout << std::endl;
    std::cout << "=== 冲突访问的分类 ===" << std::endl;
    std::cout << std::endl;
    std::cout << "两个线程的内存访问构成"冲突"（conflict）的条件:" << std::endl;
    std::cout << "  1. 它们访问同一内存位置" << std::endl;
    std::cout << "  2. 其中至少一个是写操作" << std::endl;
    std::cout << std::endl;
    std::cout << "有同步保护的冲突 → 安全:" << std::endl;
    std::cout << "  冲突访问通过同步操作（fence、release/acquire、lock、barrier）" << std::endl;
    std::cout << "  建立了 happens-before 排序 → 无数据竞争，程序正确。" << std::endl;
    std::cout << std::endl;
    std::cout << "无同步保护的冲突 → 数据竞争:" << std::endl;
    std::cout << "  冲突访问之间没有 happens-before 关系 → 构成数据竞争。" << std::endl;
    std::cout << "  输出取决于各线程的相对执行速度（非确定性/非确定性的）。" << std::endl;
    std::cout << std::endl;
    std::cout << "编程实践建议:" << std::endl;
    std::cout << "  绝大多数程序使用同步库（互斥锁、信号量等），无需手动处理排序。" << std::endl;
    std::cout << "  直接不加保护地访问共享变量 → 几乎一定是 bug！" << std::endl;
}

// ============================================================================
// 第六部分：锁（Mutex）作为一种隐式屏障
//
// 互斥锁（std::mutex）的 lock() 和 unlock() 操作隐含 acquire 和 release 语义：
//   lock()   → acquire 语义: 所有在 lock 之后的操作不会被重排序到 lock 之前
//   unlock() → release 语义: 所有在 unlock 之前的操作不会被重排序到 unlock 之后
//
// 这意味着正确使用锁的代码天然满足 SC 的语义要求。
// 两个线程通过锁的"交接"构成了 happens-before 关系：
//   线程 A: unlock() 之前的写入
//     happens-before 线程 B: lock() 之后的读取
//
// 这就是为什么"DRF 程序 = SC"定理成立的原因：
// 所有标准的同步原语内部都包含了必要的内存屏障。
// ============================================================================

void demo_lock_as_fence() {
    std::cout << std::endl;
    std::cout << "=== 锁（Mutex）作为隐式内存屏障 ===" << std::endl;
    std::cout << std::endl;

    int shared_data = 0;
    std::mutex mtx;
    const int N = 100'000;   // 每个线程 10 万次自增

    // 工作线程：通过 lock_guard 自动获取和释放锁
    auto worker = [&](int id) {
        for (int i = 0; i < N; ++i) {
            std::lock_guard<std::mutex> lk(mtx);
            // lock() 有 acquire 语义 —— 此处加载的数据是最新的
            shared_data++;
            // unlock() 有 release 语义 —— 所有写入对下一个锁持有者可见
        }
    };

    std::thread t0(worker, 0);
    std::thread t1(worker, 1);
    std::thread t2(worker, 2);
    std::thread t3(worker, 3);

    t0.join(); t1.join(); t2.join(); t3.join();

    std::cout << "4 个线程，每个 " << N << " 次自增。" << std::endl;
    std::cout << "期望值: " << (4 * N) << std::endl;
    std::cout << "实际值: " << shared_data << std::endl;
    std::cout << "结果正确！互斥锁隐式提供了 acquire/release 屏障。" << std::endl;
    std::cout << "这就是为什么 DRF（Data-Race-Free）程序天然满足 SC（顺序一致性）。" << std::endl;
}

// ============================================================================
// 第七部分：Litmus Test（一致性检验测试）总结
//
// Litmus tests 是微小的并发程序，用于检验硬件或编译器是否遵循
// 声明的内存一致性模型。以下是常见的几个：
// ============================================================================

void litmus_test_summary() {
    std::cout << std::endl;
    std::cout << "=== 常见的一致性检验测试（Litmus Tests） ===" << std::endl;
    std::cout << std::endl;

    // 测试 1: Store Buffering (SB / Dekker)
    std::cout << "1. Store Buffering（存储缓冲 / Dekker / SB）:" << std::endl;
    std::cout << "   P0: X=1; r1=Y     P1: Y=1; r2=X" << std::endl;
    std::cout << "   SC:  (r1,r2) ≠ (0,0)        TSO/x86: (0,0) 允许出现" << std::endl;
    std::cout << "   违反的排序: W→R（写后读，每个线程的读可超越自己的写）" << std::endl;
    std::cout << std::endl;

    // 测试 2: Message Passing (MP)
    std::cout << "2. Message Passing（消息传递 / MP）:" << std::endl;
    std::cout << "   P0: X=1; Y=1      P1: r1=Y; r2=X" << std::endl;
    std::cout << "   SC/TSO: r1=1 ⇒ r2=1         Relaxed: r1=1 但 r2=0 可能出现" << std::endl;
    std::cout << "   修复: Y 用 release store，Y 用 acquire load" << std::endl;
    std::cout << "   违反的排序: W→W 和 W→R（relaxed 下数据可能晚于标志可见）" << std::endl;
    std::cout << std::endl;

    // 测试 3: IRIW (Independent Reads Independent Writes)
    std::cout << "3. IRIW（独立读独立写 / Independent Reads Independent Writes）:" << std::endl;
    std::cout << "   P0: X=1   P1: Y=1   P2: r1=X; r2=Y   P3: r3=Y; r4=X" << std::endl;
    std::cout << "   SC: P2 和 P3 必须对写操作的发生顺序达成一致" << std::endl;
    std::cout << "   TSO: P2 和 P3 可能看到不同的写顺序（但 x86 实际禁止这种结果！）" << std::endl;
    std::cout << std::endl;

    // 测试 4: Coherence（同一地址的一致性）
    std::cout << "4. Coherence（同一地址的写顺序）:" << std::endl;
    std::cout << "   P0: X=1   P1: X=2   P2: r1=X; r2=X" << std::endl;
    std::cout << "   所有模型允许: r1=1,r2=2 或 r1=2,r2=2 或 r1=2,r2=1" << std::endl;
    std::cout << "   但 r1=2,r2=1（读到"先 2 后 1"）→ 违反 coherence！" << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    explain_fences();
    demo_fence_fixing_dekker();
    explain_happens_before();
    demonstrate_data_race();
    classify_conflicts();
    demo_lock_as_fence();
    litmus_test_summary();

    std::cout << std::endl;
    std::cout << "=== 总结 ===" << std::endl;
    std::cout << "1. 内存屏障（fence）在指定位置阻止特定类型的重排序。" << std::endl;
    std::cout << "2. Happens-before 分析: 环 = 不可达的执行结果。" << std::endl;
    std::cout << "3. 数据竞争 → 在 C++11 中是未定义行为。使用原子操作（atomic）或锁。" << std::endl;
    std::cout << "4. DRF 程序天然满足 SC —— 同步库帮你处理所有内存排序问题。" << std::endl;
    std::cout << "5. C++11 内存模型的核心契约: "SC for DRF"（无竞争 = 顺序一致性）。" << std::endl;

    return 0;
}
