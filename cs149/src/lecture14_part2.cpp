// lecture14_part2.cpp — CS149 第14讲：MESI 协议 + 伪共享演示
// ============================================================================
// 【课程核心概念】
// 本文件涵盖三大主题：
//
// 1. MESI 协议（比 MSI 多了 E 状态）：
//    MSI 协议的一个关键低效点：读后写（read-then-write）的常见场景需要 2 次总线事务。
//    MESI 通过引入 E（Exclusive Clean）状态来解决：当某个核心是唯一的读取者时，
//    进入 E 而非 S —— E→M 升级是"静默"的，无需总线事务！
//    这使读后写从 2 次事务降为 1 次，显著减少总线流量。
//
// 2. 伪共享（False Sharing）：
//    当两个线程各自写不同的变量，但这些变量位于同一个缓存行（通常 64 字节）时，
//    缓存一致性协议会强制缓存行在核心之间"乒乓"传递。
//    即使线程之间没有真正共享数据，也会产生大量一致性流量，严重拖慢性能。
//    解决方法：将每个线程的数据填充到独占的缓存行（padding）。
//
// 3. 多处理器 AMAT（平均内存访问时间）：
//    多处理器系统的 AMAT 比单处理器更复杂，因为：
//    - 共享缓存容量被分割（更高的缺失率）
//    - 一致性缺失（伪共享、真共享导致）
//    - NUMA 远程访问延迟
// ============================================================================
// 编译：g++ -std=c++17 -O2 -pthread lecture14_part2.cpp -o lecture14_part2
// 运行：./lecture14_part2

#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>
#include <cstring>
#include <cassert>

// ============================================================================
// 第一部分：MESI 协议详解
//
// MESI = Modified + Exclusive + Shared + Invalid
//
// 【E (Exclusive Clean) 状态的关键意义】
// 在 MSI 中，当一个核心做 PrRd 且该缓存行不在任何其他核心中时，进入 S 状态。
// 此时如果该核心再做 PrWr，就需要 BusRdX（升级）→ 2 次事务（BusRd + BusRdX）。
// 在 MESI 中，如果检测到没有其他核心持有该行，则进入 E 状态（而非 S）。
// E 状态的核心做 PrWr 时直接升级为 M —— 无需任何总线事务！
//
// 【如何检测"独占性"？—— Shared Line】
// 在 snoop-based 协议中，总线有一个"shared"信号线（wired-OR 逻辑）。
// 当某个缓存发出 BusRd 时，所有其他缓存如果持有该地址的有效副本，
// 会在响应周期内拉低 shared 线。发起者通过检测 shared 线来判断：
//   shared 为高（没有被拉低）→ 没有其他副本 → 进入 E 状态
//   shared 为低（被某个缓存拉低了）→ 存在其他副本 → 进入 S 状态
// ============================================================================
void explain_mesi() {
    std::cout << "=== CS149 第14讲：MESI（注意是 MESI，不是足球明星 Messi！） ===" << std::endl;
    std::cout << std::endl;
    std::cout << "MESI 在 MSI 的基础上增加了 E（Exclusive Clean，独占干净）状态。" << std::endl;
    std::cout << std::endl;
    std::cout << "MESI 四种状态:" << std::endl;
    std::cout << "  M (Modified-已修改):    脏数据，独占 —— 唯一副本，可以自由读写" << std::endl;
    std::cout << "  E (Exclusive-独占干净): 干净数据，独占 —— 唯一副本，主存是最新的" << std::endl;
    std::cout << "  S (Shared-共享):        干净数据，共享 —— 主存是最新的，其他核心也可能持有" << std::endl;
    std::cout << "  I (Invalid-无效):       不存在或过期，无法使用" << std::endl;
    std::cout << std::endl;
    std::cout << "MESI 核心创新: E→M 升级无需任何总线事务!" << std::endl;
    std::cout << std::endl;
    std::cout << "MSI 读后写场景（常见模式）:" << std::endl;
    std::cout << "  PrRd: I→S via BusRd   (1 次事务)" << std::endl;
    std::cout << "  PrWr: S→M via BusRdX  (1 次事务) → 共 2 次总线事务" << std::endl;
    std::cout << std::endl;
    std::cout << "MESI 读后写场景（无其他核心持有该行时）:" << std::endl;
    std::cout << "  PrRd: I→E via BusRd   (1 次事务, 检测到没有其他副本)" << std::endl;
    std::cout << "  PrWr: E→M 静默升级    (0 次事务!) → 共仅 1 次事务!" << std::endl;
    std::cout << std::endl;
    std::cout << "缓存如何知道应该进 E 还是 S?" << std::endl;
    std::cout << "  总线上的 shared 信号线（wired-OR）指示是否有其他缓存也持有该行。" << std::endl;
    std::cout << "  如果没有任何缓存拉低 shared 线 → 唯一副本 → 进入 E 状态。" << std::endl;

    std::cout << std::endl;
    std::cout << "MESI 状态转换表:" << std::endl;
    std::cout << "  I + PrRd → E (独占时) 或 S (共享时)" << std::endl;
    std::cout << "  I + PrWr → M (通过 BusRdX 获取独占权)" << std::endl;
    std::cout << "  E + PrRd → E (命中, 无需总线)" << std::endl;
    std::cout << "  E + PrWr → M (静默升级, 无需总线! — MESI 的关键优势)" << std::endl;
    std::cout << "  S + PrRd → S (命中)" << std::endl;
    std::cout << "  S + PrWr → M (通过 BusRdX 升级, 废除其他副本)" << std::endl;
    std::cout << "  M + PrRd → M (命中)" << std::endl;
    std::cout << "  M + PrWr → M (命中)" << std::endl;
    std::cout << "  E/S + BusRdX → I (被其他核心的写请求废除)" << std::endl;
    std::cout << "  M + BusRd → S (降级, 通过 BusWB 提供脏数据)" << std::endl;
    std::cout << "  M + BusRdX → I (废除, 通过 BusWB 提供脏数据)" << std::endl;
}

// ============================================================================
// 第二部分：伪共享（False Sharing）演示
//
// 【什么是伪共享？】
// 现代 CPU 的缓存行通常是 64 字节。当一个核心写入缓存行中的任意位置时，
// 整个 64 字节的行都被标记为 M（Modified），其他核心的同一行被废除。
// 如果两个核心各自写入同一缓存行中的不同变量（如 counter[0] 和 counter[1]），
// 虽然它们没有共享数据，但缓存一致性协议会强制缓存行在核心之间频繁传输。
// 这种"无形的数据共享"称为伪共享（False Sharing）。
//
// 【伪共享的性能影响】
// 每次缓存行从核心 A 传到核心 B 需要约 75 个周期（Core i7 Xeon L3 命中已修改）。
// 当 2 个核心对同一缓存行做 5000 万次写入时，伪共享会造成数百万次缓存行 ping-pong。
// 实测中，伪共享可能导致 4 线程版本比单线程还慢！
//
// 【解决方案：填充（Padding）】
// 将每个线程私有的数据填充到 64 字节，确保不同线程的数据位于不同的缓存行。
// C++17 可以使用 alignas(64)、C++11 可以用 char padding[64 - sizeof(T)] 方式。
// ============================================================================

constexpr int CACHE_LINE_SIZE = 64;          // 典型 x86 缓存行大小
constexpr long long N_ITERATIONS = 50'000'000LL;  // 大量迭代以观察伪共享效果

// ----- 版本 1：未填充 —— 易受伪共享影响 -----
// 相邻的数组元素 counter[0] 和 counter[1] 共享同一个 64 字节缓存行。
// 当线程 0 写 counter[0]、线程 1 写 counter[1] 时，
// 两者修改的是同一个缓存行 → 触发 ping-pong 废除流量。
// 注意：用 volatile 防止编译器将计数器优化到寄存器中
void worker_false_share(volatile long long* counter, long long n) {
    for (long long i = 0; i < n; ++i)
        (*counter)++;
}

// ----- 版本 2：已填充 —— 每个计数器独占自己的缓存行 -----
// sizeof(PaddedCounter) == CACHE_LINE_SIZE (64 字节)。
// 每个线程的计数器位于不同的缓存行 → 没有一致性流量（无伪共享）。
// 使用静态断言确保结构体大小正好是 64 字节。
struct PaddedCounter {
    long long counter;                                    // 8 字节的计数器
    char padding[CACHE_LINE_SIZE - sizeof(long long)];    // 填充到 64 字节
};
static_assert(sizeof(PaddedCounter) == CACHE_LINE_SIZE,
              "PaddedCounter 必须恰好占据一个缓存行");

void worker_no_false_share(volatile long long* counter, long long n) {
    for (long long i = 0; i < n; ++i)
        (*counter)++;
}

// 运行伪共享测试：返回执行时间（秒）
// padded=true 表示使用对齐填充的计数器（无伪共享）
// padded=false 表示使用紧密排列的计数器（有伪共享）
double time_execution(bool padded, int num_threads) {
    std::vector<std::thread> threads;

    // 分配计数器数组
    std::vector<long long> counters_unpadded(num_threads, 0);         // 未填充版本
    std::vector<PaddedCounter> counters_padded(num_threads);          // 已填充版本

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < num_threads; ++t) {
        volatile long long* ptr;
        if (padded)
            ptr = &counters_padded[t].counter;    // 每个计数器独占一个缓存行
        else
            ptr = &counters_unpadded[t];           // 计数器可能在同一缓存行

        threads.emplace_back(worker_no_false_share, ptr, N_ITERATIONS);
    }

    for (auto& th : threads) th.join();

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

void demo_false_sharing() {
    std::cout << "================================================================" << std::endl;
    std::cout << "=== 伪共享（False Sharing）演示 ===" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::endl;
    std::cout << "场景描述: 每个线程执行 " << N_ITERATIONS << " 次自增，共多个线程并行执行。" << std::endl;
    std::cout << "          每个线程写入各自独立的计数器（并无真正的数据共享）。" << std::endl;
    std::cout << "          但相邻计数器可能位于同一缓存行 → 伪共享效应。" << std::endl;
    std::cout << std::endl;

    std::cout << "未填充版本: sizeof(long long) = " << sizeof(long long) << " 字节" << std::endl;
    std::cout << "  Counter[0] 和 Counter[1] 仅相距 " << sizeof(long long)
              << " 字节 → 在同一个 64 字节缓存行内！" << std::endl;
    std::cout << std::endl;

    std::cout << "已填充版本: sizeof(PaddedCounter) = " << sizeof(PaddedCounter) << " 字节" << std::endl;
    std::cout << "  每个计数器独占一个 64 字节缓存行 → 无伪共享。" << std::endl;
    std::cout << std::endl;

    // 获取硬件并发度（逻辑核心数）
    unsigned int hw_threads = std::thread::hardware_concurrency();
    std::cout << "硬件并发数（逻辑核心数）: " << hw_threads << std::endl;
    std::cout << std::endl;

    // 结果对比表格
    std::cout << std::left
              << std::setw(14) << "线程数"
              << std::setw(18) << "未填充 (秒)"
              << std::setw(16) << "已填充 (秒)"
              << std::setw(12) << "加速比" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    // 测试不同线程数：1, 2, 4, min(8, 硬件线程数)
    for (int nt : {1, 2, 4, std::min(8, (int)hw_threads)}) {
        double t_unpadded = time_execution(false, nt);   // 未填充：有伪共享
        double t_padded   = time_execution(true,  nt);   // 已填充：无伪共享
        double speedup    = t_unpadded / t_padded;        // 加速比：填充带来的提升

        std::cout << std::left << std::fixed << std::setprecision(2)
                  << std::setw(14) << nt
                  << std::setw(18) << t_unpadded
                  << std::setw(16) << t_padded
                  << std::setw(10) << speedup << "x" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "=== 伪共享为什么如此致命 ===" << std::endl;
    std::cout << "缓存行在核心之间频繁"乒乓"传递的过程:" << std::endl;
    std::cout << "  核心 P0 写 Counter[0] → 缓存行移到 P0 的 L1 缓存（M 状态）" << std::endl;
    std::cout << "  核心 P1 写 Counter[1] → 缓存行移到 P1 的 L1 缓存（M 状态）" << std::endl;
    std::cout << "    → P0 的缓存行被 BusRdX 废除" << std::endl;
    std::cout << "  核心 P0 再写 Counter[0] → 缓存行又移回 P0" << std::endl;
    std::cout << "    → P1 的缓存行被废除" << std::endl;
    std::cout << "  ... 如此往复，浪费了数千个周期处理一致性流量而非真正的计算！" << std::endl;
    std::cout << std::endl;
    std::cout << "解决方法: 将每个线程的私有数据填充到 CACHE_LINE_SIZE（64 字节）边界。" << std::endl;
    std::cout << "在 C++ 中可以使用 alignas(64) 或手动 padding 实现。" << std::endl;
}

// ============================================================================
// 第三部分：多处理器系统中的 AMAT（平均内存访问时间）
//
// AMAT = Σ (访问频率 × 访问延迟)，涵盖所有访问类型。
//
// 【单处理器 AMAT】只考虑本地缓存层次：寄存器、L1、L2、主存。
// 【多处理器 AMAT】还必须考虑：
//   - L3（共享缓存）命中：其他核心可能持有干净或脏的副本
//   - 一致性缺失：伪共享和真正的数据共享导致额外的缓存行传输
//   - NUMA 延迟：远程 DRAM 访问比本地 DRAM 慢数倍
// ============================================================================
void explain_amat() {
    std::cout << std::endl;
    std::cout << "=== 多处理器系统中的 AMAT（平均内存访问时间） ===" << std::endl;
    std::cout << std::endl;
    std::cout << "AMAT = Σ (各访问类型的频率 × 对应延迟)" << std::endl;
    std::cout << std::endl;
    std::cout << "单处理器访问来源: 寄存器, L1 缓存, L2 缓存, 主存" << std::endl;
    std::cout << "多处理器额外增加: L3 共享缓存命中(未修改), L3 共享缓存命中(已修改/远程)" << std::endl;
    std::cout << std::endl;
    std::cout << "Intel Core i7 Xeon 5500 系列近似延迟数据:" << std::endl;
    std::cout << "  L1 命中:                            约 4 个周期" << std::endl;
    std::cout << "  L2 命中:                            约 10 个周期" << std::endl;
    std::cout << "  L3 命中（未共享, 本地独占）:         约 40 个周期" << std::endl;
    std::cout << "  L3 命中（共享, 其他核心持有干净的 S）: 约 65 个周期" << std::endl;
    std::cout << "  L3 命中（已修改, 其他核心持有脏的 M）: 约 75 个周期" << std::endl;
    std::cout << "  本地 DRAM:                           约 120 个周期" << std::endl;
    std::cout << "  远程 DRAM（NUMA 跨 socket）:          约 400 个周期" << std::endl;
    std::cout << std::endl;
    std::cout << "核心洞察: AMAT_多处理器 > AMAT_单处理器，原因有三:" << std::endl;
    std::cout << "  1. 共享缓存容量被多个核心分割 → 更高的缺失率" << std::endl;
    std::cout << "  2. 一致性缺失（伪共享 + 真共享）→ 额外的缓存行传输开销" << std::endl;
    std::cout << "  3. NUMA 架构下的远程访问延迟" << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    explain_mesi();

    std::cout << std::endl;
    demo_false_sharing();

    explain_amat();

    return 0;
}
