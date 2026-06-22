// lecture1_part3.cpp - 缓存模拟：LRU、时间局部性与空间局部性
// =============================================================================
// CS149 第1讲核心概念：
//   - 内存层次结构：L1 → L2 → L3 → DRAM（延迟递增，容量递增）
//     L1 最快但最小（~32KB），DRAM 最慢但最大（GB 级别）。
//     各层延迟差异巨大：L1 约 4 周期，DRAM 约 248 周期（约 60 倍差距）。
//
//   - 缓存（Cache）是片上的存储区域，维护内存子集的副本
//     当 CPU 需要访问某个地址时，首先检查该地址是否在缓存中。
//     命中（Hit）：数据在缓存中 → 快速访问。
//     未命中（Miss）：数据不在缓存中 → 必须从更慢的存储层获取。
//
//   - 缓存以"缓存行"（Cache Line）为粒度操作
//     缓存行是缓存与主存之间数据传输的最小单位（典型大小为 64 字节）。
//     即使程序只需要 1 个字节，也会加载整行 64 字节。
//     这是空间局部性优化的基础。
//
//   - LRU（最近最少使用，Least Recently Used）替换策略
//     当缓存已满且需要加载新数据时，必须驱逐某一行。
//     LRU 策略选择最久未被访问的行进行替换。
//     这是硬件中常用的策略（其他：FIFO、随机、LFU）。
//
//   - 时间局部性（Temporal Locality）：对同一地址的重复访问 → 缓存命中
//     如果程序反复使用同一数据（如循环中的累加器），
//     该数据很可能一直留在缓存中，产生高命中率。
//
//   - 空间局部性（Spatial Locality）：加载一个缓存行同时预载了邻近地址
//     顺序访问数组是空间局部性最好的模式——
//     一次未命中后，后续 15 个 int（64B 缓存行 / 4B per int）都命中。
//
//   - 三种缓存未命中：
//     1. 冷未命中（Cold Miss）：数据第一次被访问，缓存中不存在
//     2. 容量未命中（Capacity Miss）：工作集超过缓存容量，被迫驱逐后重新访问
//     3. 冲突未命中（Conflict Miss）：由于组关联度限制导致的未命中
//        （本例使用全关联 LRU，不含冲突未命中）
//
//   - 数据传输的能耗成本：
//     整数操作 ~1pJ，浮点操作 ~20pJ，
//     片上 SRAM 读取 64 位 ~26pJ，移动 DRAM 读取 64 位 ~1200pJ。
//     数据移动的能耗远高于计算——高效的程序应尽量减少数据移动。
//
// 编译: g++ -std=c++17 -O2 lecture1_part3.cpp -o lecture1_part3
// =============================================================================

#include <iostream>
#include <vector>
#include <list>
#include <unordered_map>
#include <iomanip>
#include <algorithm>
#include <cassert>
#include <random>
#include <chrono>
#include <cmath>

// ---------------------------------------------------------------------------
// 可配置的 LRU 缓存模拟器
//
// 实现细节：
//   - 使用链表维护 LRU 顺序（前端 = 最近使用，后端 = 最久未使用）
//   - 使用哈希表（unordered_map）实现 O(1) 的行查找
//   - 支持统计命中/未命中计数和访问延迟计算
//
// LRU 维护逻辑：
//   - 命中时：将该行移到链表前端（表示最近使用）
//   - 未命中时：如果缓存已满，驱逐链表后端的行；然后插入新行到前端
// ---------------------------------------------------------------------------
class LRUCache {
public:
    struct Config {
        int cache_size;       // 缓存总容量（字节）
        int line_size;        // 每行字节数（缓存行大小）
        int word_size;        // 每字字节数（int/float 通常为 4）
        int access_latency;   // 命中时的延迟（周期数）
        int miss_penalty;     // 未命中时的额外延迟（周期数）
    };

    struct Stats {
        int hits = 0;
        int misses = 0;
        int cold_misses = 0;
        int capacity_misses = 0;
        int total_accesses = 0;
        long long total_latency = 0; // 累计访问成本（周期数），用于计算平均访问时间

        double hit_rate() const {
            return total_accesses > 0 
                ? static_cast<double>(hits) / total_accesses * 100.0 : 0.0;
        }
        double avg_access_time() const {
            return total_accesses > 0 
                ? static_cast<double>(total_latency) / total_accesses : 0.0;
        }
    };

    LRUCache(const Config& cfg) : config_(cfg) {
        num_lines_ = cfg.cache_size / cfg.line_size;
        std::cout << "    缓存已初始化：" << num_lines_ << " 行，"
                  << cfg.cache_size << " 字节总容量，"
                  << cfg.line_size << " 字节/行\n";
    }

    // 访问给定地址处的单个字节
    // 参数 address 是以字节为单位的地址。
    // 通过 地址/行大小 计算出所属的缓存行编号，
    // 偏移量（行内位置）被抽象掉（因为我们以行为粒度操作）。
    void access(unsigned int address) {
        unsigned int line_addr = address / config_.line_size;
        int offset = address % config_.line_size;
        (void)offset; // 行内的字级访问被抽象化了

        stats_.total_accesses++;

        auto it = line_map_.find(line_addr);
        if (it != line_map_.end()) {
            // 缓存命中！将该行移到 LRU 链表的前端
            stats_.hits++;
            lru_list_.erase(it->second);
            lru_list_.push_front(line_addr);
            line_map_[line_addr] = lru_list_.begin();
            stats_.total_latency += config_.access_latency;
        } else {
            // 缓存未命中
            stats_.misses++;
            stats_.total_latency += config_.access_latency + config_.miss_penalty;

            if (static_cast<int>(lru_list_.size()) < num_lines_) {
                // 冷未命中：缓存尚未填满
                stats_.cold_misses++;
            } else {
                // 容量未命中：需要驱逐 LRU 行（最久未使用的行）
                stats_.capacity_misses++;
                unsigned int evicted = lru_list_.back();
                lru_list_.pop_back();
                line_map_.erase(evicted);
            }

            // 将新行插入链表前端
            lru_list_.push_front(line_addr);
            line_map_[line_addr] = lru_list_.begin();
        }
    }

    // 顺序访问一段地址范围（演示空间局部性）
    // 连续地址会落入同一缓存行 → 第一次未命中后，后续访问均命中
    void access_range(unsigned int start, unsigned int count) {
        for (unsigned int i = 0; i < count; i++) {
            access(start + i);
        }
    }

    const Stats& stats() const { return stats_; }
    const Config& config() const { return config_; }

    void reset_stats() {
        stats_ = Stats();
        lru_list_.clear();
        line_map_.clear();
    }

private:
    Config config_;
    int num_lines_;
    Stats stats_;
    std::list<unsigned int> lru_list_;               // 前端 = 最近使用
    std::unordered_map<unsigned int, 
        std::list<unsigned int>::iterator> line_map_;
};

// ---------------------------------------------------------------------------
// 缓存访问延迟参考数据（Kaby Lake CPU，4 GHz 下的周期数）：
// L1 缓存：~4 周期
// L2 缓存：~12 周期
// L3 缓存：~38 周期
// DRAM：~248 周期（最优情况）
//
// 这些数字展示了内存层次结构中各层之间巨大的延迟差异。
// L1 访问比 DRAM 快约 60 倍，这就是为什么缓存如此重要。
// ---------------------------------------------------------------------------
void print_latency_reference() {
    std::cout << "    实际延迟参考数据（Kaby Lake @ 4 GHz）：\n";
    std::cout << "    -----------------------------------------------\n";
    std::cout << "    L1 缓存命中：   ~4 周期\n";
    std::cout << "    L2 缓存命中：   ~12 周期\n";
    std::cout << "    L3 缓存命中：   ~38 周期\n";
    std::cout << "    DRAM 访问：     ~248 周期（最优情况）\n" << std::endl;
}

// ---------------------------------------------------------------------------
// 使用缓存演示时间和空间局部性
// 复现课程幻灯片中的示例
//
// 访问模式分析：
//   - 好的局部性：同一缓存行内连续访问 + 重复使用相同地址
//   - 差的局部性：跨多个缓存行的随机/跳跃访问
// ---------------------------------------------------------------------------
void demo_cache_example(LRUCache& cache, 
                         const std::vector<unsigned int>& access_pattern,
                         const std::string& description) 
{
    std::cout << "    模式：" << description << "\n";
    std::cout << "    访问序列：";
    for (size_t i = 0; i < access_pattern.size(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << "0x" << std::hex << access_pattern[i] << std::dec;
    }
    std::cout << "\n\n";

    // 打印表头
    std::cout << "    " << std::setw(6) << "地址" 
              << std::setw(6) << "行号"
              << std::setw(12) << "结果" 
              << std::setw(12) << "缓存状态" << std::endl;
    std::cout << "    " << std::string(45, '-') << std::endl;

    for (unsigned int addr : access_pattern) {
        int hits_before = cache.stats().hits;
        cache.access(addr);
        bool is_hit = (cache.stats().hits > hits_before);
        
        std::cout << "    " << std::setw(6) << "0x" << std::hex << addr << std::dec
                  << std::setw(6) << (addr / cache.config().line_size)
                  << std::setw(12) << (is_hit ? "命中" : "未命中") 
                  << "     ..." << std::endl;
    }

    std::cout << "\n    结果：" 
              << cache.stats().hits << " 次命中，"
              << cache.stats().misses << " 次未命中 "
              << "（" << std::fixed << std::setprecision(1) << cache.stats().hit_rate() 
              << "% 命中率）\n" << std::endl;
}

// ---------------------------------------------------------------------------
// 演示数据移动的能耗成本
//
// 核心观点：在现代计算中，数据移动（从内存读取数据）的能耗
// 远超计算本身的能耗。整数操作只需 ~1pJ，而从 DRAM 读取 64 位数据
// 需要 ~1200pJ——大约 1200 倍。
//
// 高效的程序应当：
//   1. 尽量让数据留在缓存中（利用时间局部性）
//   2. 尽量顺序访问数据（利用空间局部性，减少缓存行加载次数）
// ---------------------------------------------------------------------------
void demo_energy_costs() {
    std::cout << "[3] 数据移动的能耗成本\n" << std::endl;

    std::cout << "    近似能耗成本：\n";
    std::cout << "    -------------------------------------------------\n";
    std::cout << "    整数操作：                 ~1 pJ\n";
    std::cout << "    浮点操作：                 ~20 pJ\n";
    std::cout << "    从片上 SRAM 读取 64 位：   ~26 pJ\n";
    std::cout << "    从移动 DRAM 读取 64 位：   ~1200 pJ\n\n";

    double int_op = 1.0;
    double fp_op = 20.0;
    double sram_read = 26.0;
    double dram_read = 1200.0;

    std::cout << "    相对成本（以整数操作为基准）：\n";
    std::cout << "    整数操作：         1x\n";
    std::cout << "    FP 操作：          " << fp_op / int_op << "x\n";
    std::cout << "    SRAM 读取 64b：    " << sram_read / int_op << "x\n";
    std::cout << "    DRAM 读取 64b：    " << dram_read / int_op << "x\n" << std::endl;

    // 计算内存带宽的能耗
    std::cout << "    以 10 GB/秒 的速度从内存读取数据：\n";
    std::cout << "    " << 10.0e9 / 8.0 * dram_read * 1e-12 
              << " 瓦（移动 LPDDR）\n" << std::endl;

    std::cout << "    iPhone 电池容量：约 7 瓦时\n";
    std::cout << "    → 以 10 GB/s 的内存带宽运行，电池约能支撑 4 小时\n";
    std::cout << "    → 利用局部性对功耗至关重要！\n";
}

// =============================================================================
int main() {
    std::cout << "=== CS149 第1讲：缓存模拟与内存层次结构 ===\n" << std::endl;

    // ---- 第零部分：延迟参考数据 ----
    print_latency_reference();

    // ---- 第一部分：时间与空间局部性（缓存示例 1） ----
    std::cout << "[1] 时间与空间局部性演示\n" << std::endl;

    // 配置：8 字节缓存，4 字节行 → 2 行（与课程幻灯片一致）
    LRUCache::Config cfg;
    cfg.cache_size = 8;
    cfg.line_size = 4;
    cfg.word_size = 1;
    cfg.access_latency = 4;
    cfg.miss_penalty = 50;

    {
        LRUCache cache(cfg);
        
        // 课程示例 1：良好的空间 + 时间局部性
        // 地址 0x0-0x3（第 0 行），然后 0x2, 0x1（第 0 行内的时间局部性）
        // 然后 0x4-0x7（第 1 行，良好的空间局部性）
        // 然后再次 0x1（仍在缓存中）
        std::vector<unsigned int> pattern1 = {
            0x0, 0x1, 0x2, 0x3,  // 第 0 行：冷未命中 + 3 次命中（空间局部性）
            0x2, 0x1,            // 时间局部性：仍命中
            0x4,                 // 第 1 行：冷未命中（容量尚可，第 0 行保留）
            0x1                  // 时间局部性：命中
        };
        
        demo_cache_example(cache, pattern1, 
            "良好局部性：顺序读取 + 重复访问");
    }

    // ---- 第二部分：容量未命中（缓存示例 2） ----
    std::cout << "[2] 容量未命中：大数组的顺序扫描\n" << std::endl;

    {
        LRUCache cache(cfg);
        
        // 课程示例 2：用 8 字节缓存扫描整个 16 字节数组
        // 第一次扫描将数据加载到缓存中，
        // 但数组的后半部分会驱逐前半部分（容量不够）
        // 最后访问 0x0 时是容量未命中（已被 0x8 驱逐）
        std::vector<unsigned int> pattern2 = {
            0x0, 0x1, 0x2, 0x3,  // 加载第 0 行（冷未命中）
            0x4, 0x5, 0x6, 0x7,  // 加载第 1 行（冷未命中）
            0x8, 0x9, 0xA, 0xB,  // 加载第 2 行，驱逐第 0 行（容量未命中）
            0xC, 0xD, 0xE, 0xF,  // 加载第 3 行，驱逐第 1 行（容量未命中）
            0x0                   // 重新加载第 0 行，驱逐第 2 行（容量未命中！）
        };
        
        demo_cache_example(cache, pattern2, 
            "顺序扫描 → 重新访问时产生容量未命中");
    }

    // ---- 第二部分（续）：如果缓存有 4 行而非 2 行 ----
    std::cout << "    [对比] 相同访问模式用 4 行缓存（16 字节）：\n" << std::endl;
    {
        LRUCache::Config cfg4 = cfg;
        cfg4.cache_size = 16; // 4 行，每行 4 字节
        LRUCache cache4(cfg4);
        
        std::vector<unsigned int> pattern2 = {
            0x0, 0x1, 0x2, 0x3,
            0x4, 0x5, 0x6, 0x7,
            0x8, 0x9, 0xA, 0xB,
            0xC, 0xD, 0xE, 0xF,
            0x0
        };
        
        demo_cache_example(cache4, pattern2, 
            "现在：无容量未命中（4 行可容纳所有数据）");
        std::cout << "    → 更大的缓存可以消除容量未命中\n" << std::endl;
    }

    // ---- 第三部分：能耗成本 ----
    demo_energy_costs();

    // ---- 第四部分：模拟更大规模的工作负载 ----
    std::cout << "[4] 大规模工作负载模拟：数组求和\n" << std::endl;

    LRUCache::Config l1_cfg;
    l1_cfg.cache_size = 32 * 1024;       // 32 KB L1 缓存
    l1_cfg.line_size = 64;               // 64 字节缓存行（典型的 L1 行大小）
    l1_cfg.word_size = 4;                // 4 字节 int
    l1_cfg.access_latency = 4;           // L1 命中：4 周期
    l1_cfg.miss_penalty = 12;            // L2 命中：额外 12 周期

    const int ARRAY_SIZE = 1'000'000;    // 100 万个 int = 4 MB

    // 顺序访问（极佳的空间局部性）
    // 分析：64B 缓存行 = 16 个 int，所以每 16 次访问中只有第 1 次可能未命中
    // 理论命中率 = 15/16 = 93.75%
    {
        LRUCache cache(l1_cfg);
        std::cout << "    顺序数组求和（4 MB，int）：\n";
        for (int i = 0; i < ARRAY_SIZE; i++) {
            cache.access(i * 4); // 每个 int 占 4 字节
        }
        auto& s = cache.stats();
        std::cout << "    访问次数：" << s.total_accesses 
                  << " | 命中：" << s.hits << "（" << std::fixed << std::setprecision(1) 
                  << s.hit_rate() << "%）| 平均延迟：" << std::setprecision(1) 
                  << s.avg_access_time() << " 周期\n";
        std::cout << "    缓存行大小（64B）= 每行 16 个 int → 高空间局部性\n\n";
    }

    // 跳跃访问（差的空间局部性）
    // 分析：步长为 256 个 int（1024 字节），远超 64B 缓存行
    // 每次访问都大概率命中不同的缓存行 → 大量未命中
    {
        LRUCache cache(l1_cfg);
        const int STRIDE = 256; // 每次访问跳过 256 个 int = 1024 字节
        std::cout << "    跳跃数组访问（步长=" << STRIDE << " 个 int）：\n";
        for (int i = 0; i < ARRAY_SIZE / STRIDE; i++) {
            cache.access(i * STRIDE * 4);
        }
        auto& s = cache.stats();
        std::cout << "    访问次数：" << s.total_accesses 
                  << " | 命中：" << s.hits << "（" << std::fixed << std::setprecision(1) 
                  << s.hit_rate() << "%）| 平均延迟：" << std::setprecision(1) 
                  << s.avg_access_time() << " 周期\n";
        std::cout << "    大步长 → 每次访问大概率未命中 → 局部性差\n\n";
    }

    // 随机访问（最差的局部性）
    // 分析：随机地址分布导致缓存几乎无法预测下次访问
    // 命中率取决于工作集大小与缓存容量的比例
    {
        LRUCache cache(l1_cfg);
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, ARRAY_SIZE - 1);
        const int NUM_ACCESSES = 100'000;
        
        std::cout << "    随机数组访问（" << NUM_ACCESSES << " 次访问）：\n";
        for (int i = 0; i < NUM_ACCESSES; i++) {
            cache.access(dist(rng) * 4);
        }
        auto& s = cache.stats();
        std::cout << "    访问次数：" << s.total_accesses 
                  << " | 命中：" << s.hits << "（" << std::fixed << std::setprecision(1) 
                  << s.hit_rate() << "%）| 平均延迟：" << std::setprecision(1) 
                  << s.avg_access_time() << " 周期\n";
        std::cout << "    随机访问 → 大部分未命中 → 性能最差\n\n";
    }

    // ---- 第五部分：核心要点 ----
    std::cout << "[5] 核心要点\n" << std::endl;
    std::cout << "    - 缓存层次结构：L1（快、小）→ L2 → L3 → DRAM（慢、大）\n";
    std::cout << "    - 时间局部性：重复使用最近访问过的数据\n";
    std::cout << "    - 空间局部性：访问连续数据（缓存行会预载相邻数据）\n";
    std::cout << "    - 冷未命中：数据第一次被访问\n";
    std::cout << "    - 容量未命中：工作集 > 缓存大小\n";
    std::cout << "    - LRU：最近最少使用驱逐策略（常见的硬件策略）\n";
    std::cout << "    - 数据移动主导能耗成本（约是整数操作的 1200 倍）\n";
    std::cout << "    - 高效程序应尽量减少数据移动 → 善用局部性\n";

    return 0;
}
