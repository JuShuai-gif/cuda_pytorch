/**
 * lecture6_part2.cpp - 缓存一致性与伪共享
 *
 * 演示 CS149 第6讲的概念：
 * - 缓存一致性协议概念（类 MESI 状态机）
 * - 伪共享（false sharing）：当不同线程修改同一缓存行上的不同变量时
 * - 缓存行填充（padding）以防止伪共享
 * - 共享地址空间硬件（环形互联，ring interconnect）
 * - 由缓存行为导致的人为通信（artifactual communication）
 *
 * 关键概念详解：
 * ─────────────────────────────────────────────────────────────
 * 【缓存一致性协议（MESI）】
 *   在多核 CPU 中，每个核心有自己的本地缓存。当核心 A 修改一个
 *   内存位置，而核心 B 的缓存中也有该位置的副本时，需要保证核心 B
 *   不会读到过期的数据。MESI 是一种广泛使用的窥探（snooping）协议：
 *
 *   - M（Modified，已修改）：该缓存行是脏的，只有本核心拥有有效副本，
 *     必须在被驱逐时写回内存。
 *   - E（Exclusive，独占）：该缓存行是干净的，只有本核心拥有副本，
 *     与内存一致。
 *   - S（Shared，共享）：该缓存行是干净的，其他核心可能也有副本。
 *   - I（Invalid，无效）：该缓存行数据无效，不可使用。
 *
 * 【伪共享（False Sharing）】
 *   当两个线程修改的是逻辑上独立的变量，但这些变量恰好位于同一个
 *   64 字节缓存行上时，每次写操作都会使对方的缓存行失效，即使它们
 *   修改的是不同的字节！这导致大量的缓存一致性流量，严重拖慢性能。
 *
 *   解决方案：将每个线程独占的变量对齐到缓存行边界，
 *   并用填充字节填满整个缓存行（如 alignas(64) + char padding[60]）。
 *
 * 【人为通信（Artifactual Communication）】
 *   由实现细节（而非算法需求）导致的额外数据移动：
 *   - 最小传输粒度：加载 4 字节需要传输 64 字节（16 倍的浪费）
 *   - 不必要的"先读后写"：写入整个缓存行时，先加载再覆盖
 *   - 容量未命中：缓存太小，无法在访问之间保留数据
 *   - 冲突未命中：两个频繁访问的地址映射到同一缓存组
 *
 * 编译：g++ -std=c++17 -pthread lecture6_part2.cpp -o lecture6_part2 && ./lecture6_part2
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>

// ============================================================================
// 第一部分：缓存一致性协议模拟
// ============================================================================

/**
 * 类 MESI 缓存一致性协议的简化模型。
 * 每个缓存行可以处于以下状态之一：
 * - MODIFIED（已修改）：脏的，独占副本，驱逐时必须写回内存
 * - EXCLUSIVE（独占）：干净的，独占副本，与内存一致
 * - SHARED（共享）：干净的，其他缓存也可能有副本
 * - INVALID（无效）：该行数据无效
 *
 * 【状态转换示例】
 * - 核心 A 读 X → X 在 A 中为 EXCLUSIVE（或 SHARED）
 * - 核心 B 也读 X → X 在 A 和 B 中均为 SHARED
 * - 核心 A 写 X → X 在 A 中为 MODIFIED，X 在 B 中为 INVALID
 * - 核心 B 再读 X → 发生缓存未命中，需要从 A 的 MODIFIED 副本获取数据
 */
enum class LineState { MODIFIED, EXCLUSIVE, SHARED, INVALID };

const char* state_name(LineState s) {
    switch (s) {
        case LineState::MODIFIED: return "已修改";
        case LineState::EXCLUSIVE: return "独占";
        case LineState::SHARED:   return "共享";
        case LineState::INVALID:  return "无效";
        default: return "?";
    }
}

class CacheLine {
public:
    int tag;
    LineState state;
    int data;

    CacheLine() : tag(-1), state(LineState::INVALID), data(0) {}
};

class Core {
public:
    int id;
    std::vector<CacheLine> cache;  // 每个核心有 4 行缓存（极简模拟）
    int total_misses;
    int total_invalidations;

    Core(int core_id, int cache_size = 4)
        : id(core_id), cache(cache_size), total_misses(0), total_invalidations(0) {}

    /**
     * 读取一个内存地址。模拟缓存一致性协议：
     * 1. 检查本地缓存
     * 2. 如果未命中，向其他核心广播读取请求（窥探/snoop）
     * 3. 根据 MESI 协议更新状态
     *
     * 核心逻辑：
     * - 如果地址在本地缓存中有有效副本 → 命中，直接返回
     * - 如果其他核心有 MODIFIED 副本 → 从该核心获取数据（cache-to-cache transfer）
     *   并将该核心状态降级为 SHARED
     * - 如果其他核心有 SHARED/EXCLUSIVE 副本 → 获取数据，大家都变为 SHARED
     * - 如果任何缓存中都没有 → 从"内存"加载，状态为 EXCLUSIVE
     */
    bool read(int address, int& value, std::vector<Core>& all_cores) {
        int line = address % cache.size();

        // 检查本地缓存
        if (cache[line].tag == address && cache[line].state != LineState::INVALID) {
            // 缓存命中：数据在本地缓存中且有效
            value = cache[line].data;
            return true;
        }

        // 缓存未命中：需要从内存或其他缓存获取
        total_misses++;

        // 窥探（Snoop）：检查其他核心是否拥有该地址的数据
        for (auto& other : all_cores) {
            if (other.id == id) continue;
            int other_line = address % other.cache.size();
            if (other.cache[other_line].tag == address) {
                if (other.cache[other_line].state == LineState::MODIFIED) {
                    // 其他核心持有脏数据 → 必须先写回再共享
                    // 这是 cache-to-cache transfer，避免访问主存
                    other.cache[other_line].state = LineState::SHARED;
                    cache[line].data = other.cache[other_line].data;
                    cache[line].tag = address;
                    cache[line].state = LineState::SHARED;

                    value = cache[line].data;
                    return true;
                } else if (other.cache[other_line].state == LineState::SHARED ||
                           other.cache[other_line].state == LineState::EXCLUSIVE) {
                    // 在其他核心的缓存中找到了干净的副本
                    other.cache[other_line].state = LineState::SHARED;
                    cache[line].data = other.cache[other_line].data;
                    cache[line].tag = address;
                    cache[line].state = LineState::SHARED;
                    value = cache[line].data;
                    return true;
                }
            }
        }

        // 任何缓存中都没有 → 从"内存"加载
        cache[line].tag = address;
        cache[line].state = LineState::EXCLUSIVE;
        cache[line].data = address * 10;  // 模拟从内存读取的值
        value = cache[line].data;
        return true;
    }

    /**
     * 写入一个内存地址。
     * 1. 必须先将缓存行置于 MODIFIED 状态
     * 2. 使其他核心的副本失效（RFO = Read For Ownership，获取所有权）
     *
     * MESI 协议的关键：在任何核心写入之前，必须使所有其他副本失效，
     * 确保只有写者拥有有效数据。
     */
    void write(int address, int value, std::vector<Core>& all_cores) {
        int line = address % cache.size();

        // 如果已经是 MODIFIED 或 EXCLUSIVE → 可以直接写入
        if (cache[line].tag == address &&
            (cache[line].state == LineState::MODIFIED ||
             cache[line].state == LineState::EXCLUSIVE)) {
            cache[line].data = value;
            cache[line].state = LineState::MODIFIED;
            return;
        }

        // 如果是 SHARED → 需要先使其他核心的副本失效
        if (cache[line].tag == address && cache[line].state == LineState::SHARED) {
            invalidate_others(address, all_cores);
            cache[line].data = value;
            cache[line].state = LineState::MODIFIED;
            return;
        }

        // 未命中或 INVALID → 获取独占所有权
        total_misses++;
        invalidate_others(address, all_cores);

        cache[line].tag = address;
        cache[line].data = value;
        cache[line].state = LineState::MODIFIED;
    }

private:
    /**
     * 使其他核心中该地址的副本失效。
     * 这是写入操作前必须执行的步骤（RFO 信号的一部分）。
     */
    void invalidate_others(int address, std::vector<Core>& all_cores) {
        for (auto& other : all_cores) {
            if (other.id == id) continue;
            int other_line = address % other.cache.size();
            if (other.cache[other_line].tag == address) {
                if (other.cache[other_line].state != LineState::INVALID) {
                    other.total_invalidations++;
                    other.cache[other_line].state = LineState::INVALID;
                }
            }
        }
    }
};

/**
 * 模拟核心内和核心间的缓存行为。
 * 演示为什么共享的可写数据会导致缓存失效流量。
 *
 * 情景演示：
 * 1. 核心0 读取地址5 → EXCLUSIVE（独占，只有它拥有）
 * 2. 核心1 读取地址5 → 两个核心都变为 SHARED（共享读）
 * 3. 核心0 写入地址5 → 自身变为 MODIFIED，核心1 变为 INVALID
 * 4. 核心1 再次读取地址5 → 未命中！必须从核心0 的 MODIFIED 副本获取
 */
void simulate_cache_coherency() {
    std::cout << "\n=== 缓存一致性模拟（类 MESI 协议） ===\n\n";

    const int NUM_CORES = 4;
    std::vector<Core> cores;
    for (int i = 0; i < NUM_CORES; i++) cores.emplace_back(i);

    auto print_state = [&](int addr) {
        std::cout << "  地址 " << addr << "： ";
        for (auto& c : cores) {
            int line = addr % c.cache.size();
            std::cout << "核心" << c.id << "=" << state_name(c.cache[line].state) << "  ";
        }
        std::cout << "\n";
    };

    // 情景1：核心0 读取地址5（首次加载，从内存获取，状态 EXCLUSIVE）
    std::cout << "情景1：核心0 读取地址5（首次加载 → EXCLUSIVE 独占状态）\n";
    int val;
    cores[0].read(5, val, cores);
    print_state(5);

    // 情景2：核心1 读取地址5（应该转为 SHARED 共享状态）
    std::cout << "情景2：核心1 读取地址5 → 两个核心都变为 SHARED 共享状态\n";
    cores[1].read(5, val, cores);
    print_state(5);

    // 情景3：核心0 写入地址5（使核心1 的副本失效）
    std::cout << "情景3：核心0 写入地址5 → 使核心1 的副本失效（INVALID）\n";
    cores[0].write(5, 999, cores);
    print_state(5);

    // 情景4：核心1 再次读取地址5（由于之前被失效，现在发生未命中）
    std::cout << "情景4：核心1 再次读取地址5 → 未命中（之前已被失效）\n";
    cores[1].read(5, val, cores);
    print_state(5);

    std::cout << "\n  核心0 总未命中次数: " << cores[0].total_misses
              << "  失效次数: " << cores[0].total_invalidations << "\n";
    std::cout << "  核心1 总未命中次数: " << cores[1].total_misses
              << "  失效次数: " << cores[1].total_invalidations << "\n";
    std::cout << "\n  关键洞察：核心0 对共享状态的缓存行执行写入操作，\n";
    std::cout << "  导致核心1 中该行的状态变为无效，迫使核心1 在下一次\n";
    std::cout << "  访问时重新获取数据。这就是 MESI 协议的核心机制。\n";
}

// ============================================================================
// 第二部分：伪共享演示
// ============================================================================

/**
 * 伪共享（FALSE SHARING）：
 * 当两个线程修改恰好位于同一缓存行上的不同变量时，
 * 导致不必要的缓存一致性流量。
 *
 * 缓存行大小：通常为 64 字节 = 16 个 int（每个 4 字节）或 8 个 double。
 *
 * 【伪共享的伤害有多大】
 * 如果线程 A 和线程 B 各自递增自己的计数器，但在内存中这两个计数器
 * 相邻（它们在同一缓存行上），每次写操作都会：
 * 1. 获取该缓存行的独占所有权（RFO）
 * 2. 使另一个核心的缓存行失效
 * 3. 另一个核心下次写时又要重新获取所有权
 * 结果：缓存行像乒乓球一样在两个核心之间来回弹跳，
 * 产生大量的缓存一致性流量。
 *
 * 解决方案：将每个线程的数据对齐到缓存行边界，并填充到 64 字节。
 */

// 未填充的结构体（受到伪共享影响）
struct UnpaddedCounter {
    alignas(64) int counter;  // alignas(64) 确保起始于缓存行边界
    // 但是：多个 UnpaddedCounter 在数组中连续存放时，
    // 仍可能共享同一缓存行 → 伪共享！
};
static_assert(sizeof(UnpaddedCounter) == 64, "填充后的计数器必须为 64 字节");

// 带填充的结构体，确保每个计数器独占一个缓存行
struct alignas(64) PaddedCounter {
    int counter;
    char padding[60];  // 填充剩余的缓存行空间（64 - 4 = 60 字节）
};
static_assert(sizeof(PaddedCounter) == 64, "带填充的计数器必须为 64 字节");

/**
 * 基准测试：伪共享 vs 填充后无伪共享。
 * 每个线程反复递增自己的计数器。
 * 不带填充：计数器共享缓存行 → 大量失效流量。
 * 带填充：每个计数器独占自己的缓存行 → 无伪共享。
 *
 * 预期结果：
 * - 未填充版本：速度极慢（缓存行在两个核心之间不断弹跳）
 * - 填充版本：接近线性加速（每个核心独立操作自己的缓存行）
 */
void benchmark_false_sharing() {
    std::cout << "\n=== 伪共享基准测试 ===\n\n";

    const int NUM_THREADS = 4;
    const int ITERATIONS = 10000000;

    // === 未填充：所有计数器共享缓存行 ===
    {
        // 分配未填充计数器数组（在内存中连续）
        // 多个计数器会落入同一 64 字节缓存行
        alignas(64) int counters[NUM_THREADS] = {0};

        auto worker = [&](int tid) {
            for (int i = 0; i < ITERATIONS; i++) {
                counters[tid]++;  // 伪共享：每次写入使整个缓存行失效
            }
        };

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::thread> threads;
        for (int t = 0; t < NUM_THREADS; t++) {
            threads.emplace_back(worker, t);
        }
        for (auto& th : threads) th.join();
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "  未填充（存在伪共享）： "
                  << std::fixed << std::setprecision(4) << elapsed << "秒\n";

        // 验证结果的正确性
        long long total = 0;
        for (int t = 0; t < NUM_THREADS; t++) total += counters[t];
        std::cout << "    总和: " << total << "（期望值: "
                  << (1LL * NUM_THREADS * ITERATIONS) << "）\n";
    }

    // === 填充版本：每个计数器独占一条缓存行 ===
    {
        PaddedCounter counters[NUM_THREADS];
        for (int t = 0; t < NUM_THREADS; t++) counters[t].counter = 0;

        auto worker = [&](int tid) {
            for (int i = 0; i < ITERATIONS; i++) {
                counters[tid].counter++;  // 无伪共享：每个计数器独占一条缓存行
            }
        };

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::thread> threads;
        for (int t = 0; t < NUM_THREADS; t++) {
            threads.emplace_back(worker, t);
        }
        for (auto& th : threads) th.join();
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "  填充版（无伪共享）： "
                  << std::fixed << std::setprecision(4) << elapsed << "秒\n";

        long long total = 0;
        for (int t = 0; t < NUM_THREADS; t++) total += counters[t].counter;
        std::cout << "    总和: " << total << "（期望值: "
                  << (1LL * NUM_THREADS * ITERATIONS) << "）\n";
    }

    std::cout << "\n  伪共享是一个「看不见」的性能杀手。\n";
    std::cout << "  两个线程修改的是逻辑上完全独立的变量，但只要这些变量\n";
    std::cout << "  位于同一缓存行上，就会在缓存之间产生失效流量。\n";
    std::cout << "  解决方案：对齐到缓存行边界，并使用填充字节填充至 64 字节。\n";
}

// ============================================================================
// 第三部分：人为通信示例
// ============================================================================

void explain_artifactual_communication() {
    std::cout << "\n=== 人为通信（Artifactual Communication） ===\n\n";

    std::cout << "人为通信 = 由实现细节（而非算法需求）导致的不必要数据移动。\n";
    std::cout << "它是「人」为的-不是算法本身需要这么多通信，而是硬件/缓存\n";
    std::cout << "的实现机制导致的。\n\n";

    std::cout << "示例1：最小传输粒度带来的浪费\n";
    std::cout << "  - 加载 1 个 4 字节 float → 整个 64 字节缓存行被传输\n";
    std::cout << "  - 通信量是实际需要的 16 倍！\n";
    std::cout << "  - 这是架构限制，无法避免，但可以优化访问模式使其物尽其用\n\n";

    std::cout << "示例2：不必要的「先读后写」（load-before-store）\n";
    std::cout << "  - 写入 16 个连续的 float → 缓存行被加载，然后全部被覆盖\n";
    std::cout << "  - 产生了 2 倍的额外开销：加载实际上是不需要的\n";
    std::cout << "    （因为整行都会被覆写）\n";
    std::cout << "  - 解决方案：使用非时间存储（streaming stores / non-temporal stores）\n";
    std::cout << "    绕过缓存直接写入内存，避免先读后写的浪费\n\n";

    std::cout << "示例3：容量未命中（有限缓存大小）\n";
    std::cout << "  - 缓存太小，无法在两次访问之间保留数据\n";
    std::cout << "  - 相同的数据被多次从内存传输到缓存\n";
    std::cout << "  - 解决方案：分块/平铺（blocking/tiling）让工作集适配缓存\n\n";

    std::cout << "示例4：冲突未命中（cache conflict misses）\n";
    std::cout << "  - 两个频繁访问的地址映射到同一缓存组（set）\n";
    std::cout << "  - 即便是 N 路组相联缓存，也可能在某组内频繁冲突\n";
    std::cout << "  - 解决方案：填充（padding）、数据布局重组\n";
}

// ============================================================================
// 第四部分：环形互联模型
// ============================================================================

void explain_ring_interconnect() {
    std::cout << "\n=== Intel 环形互联（Ring Interconnect） ===\n\n";

    std::cout << "在 Sandy Bridge 微架构中引入：\n\n";
    std::cout << "  四条环形总线，分别用于不同类型的消息：\n";
    std::cout << "    - 请求环（Request ring）：发出内存请求\n";
    std::cout << "    - 窥探环（Snoop ring）：广播缓存一致性查询\n";
    std::cout << "    - 确认环（Acknowledgement ring）：确认操作完成\n";
    std::cout << "    - 数据环（Data ring，32 字节宽）：传输实际数据\n\n";

    std::cout << "  六个互联节点：\n";
    std::cout << "    - 四个 L3 缓存的「切片」（slice），每个 2 MB\n";
    std::cout << "    - 系统代理（System Agent）：连接内存控制器、PCIe\n";
    std::cout << "    - 图形处理器（Graphics）\n\n";

    std::cout << "  每个 L3 bank 与环形总线双向连接（发送和接收各一条路径）\n";
    std::cout << "  在 3.4 GHz 下，核心到 L3 的峰值带宽 ≈ 435 GB/秒\n";
    std::cout << "  （当每个核心都访问自己本地 L3 切片时）\n\n";

    std::cout << "  即使在单插槽上也有 NUMA 效应：不同 L3 切片在\n";
    std::cout << "  环形拓扑上与各核心的距离不同，访问延迟也因此不同。\n";
    std::cout << "  核心访问最近的一个 L3 切片比访问环形对侧的切片更快。\n\n";

    std::cout << "  【环形 vs 网格互联】\n";
    std::cout << "  - 环形拓扑适用于核心数较少的情况（4-12 核）\n";
    std::cout << "  - 网格（mesh）拓扑适用于大规模多核心（如 Xeon Phi, >20 核）\n";
    std::cout << "  - 环的每跳延迟固定，总延迟 = 跳数 × 每跳延迟\n";
    std::cout << "  - 最坏情况（对侧节点）需要跳 n/2 步\n";
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "============================================================\n";
    std::cout << "第6讲 第二部分：缓存一致性与伪共享\n";
    std::cout << "============================================================\n";

    // 第一部分：缓存一致性模拟
    simulate_cache_coherency();

    // 第二部分：伪共享基准测试
    benchmark_false_sharing();

    // 第三部分：人为通信
    explain_artifactual_communication();

    // 第四部分：环形互联
    explain_ring_interconnect();

    // 总结
    std::cout << "\n=== 缓存一致性与伪共享：核心概念 ===\n";
    std::cout << "┌─────────────────────┬─────────────────────────────────────┐\n";
    std::cout << "│ 概念                │ 影响                                │\n";
    std::cout << "├─────────────────────┼─────────────────────────────────────┤\n";
    std::cout << "│ MESI 协议           │ 自动保持所有缓存之间的一致性       │\n";
    std::cout << "│ 写入失效            │ 写入者必须使所有副本失效           │\n";
    std::cout << "│ 伪共享              │ 同一缓存行上的独立变量            │\n";
    std::cout << "│                     │ 造成不必要的缓存失效流量           │\n";
    std::cout << "│ 缓存行填充          │ alignas(64) + char pad[60]          │\n";
    std::cout << "│ 人为通信            │ 最小粒度、容量未命中等引起的浪费   │\n";
    std::cout << "│ 环形互联            │ 4 条环，环上多跳延迟               │\n";
    std::cout << "│ NUMA                │ 访问延迟因数据所在位置而异         │\n";
    std::cout << "└─────────────────────┴─────────────────────────────────────┘\n";

    std::cout << "\n所有测试成功完成。\n";
    return 0;
}
