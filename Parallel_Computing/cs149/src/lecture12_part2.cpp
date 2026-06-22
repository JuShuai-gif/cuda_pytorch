// lecture12_part2.cpp
// DRAM 模拟器：存储体 (Bank)、行缓冲区 (Row Buffer)、突发传输模式 (Burst Mode)、内存控制器
// 模拟 Lecture 12 中描述的 DRAM 内部结构和工作原理
//
// 核心概念详解：
// 1. DRAM Bank（存储体）：DRAM 芯片内部由多个独立的 bank 组成（典型为 8 个）。
//    每个 bank 可以独立处理请求，从而支持请求流水线化（request pipelining），
//    提高引脚（pin）利用率。这是 HBM 实现高带宽的关键——更多 bank = 更多并行请求。
// 2. Row Buffer（行缓冲区/感测放大器）：每个 bank 内部有一个行缓冲区（也叫感测放大器，
//    sense amplifier）。当一个行被激活（RAS，Row Access Strobe）时，该行的所有数据
//    被读到行缓冲区。后续对该行任意列的访问（CAS，Column Access Strobe）都直接从
//    行缓冲区读取（行命中/row hit），无需重新激活行。
//    行命中延迟 = tCAS + tBURST（~17 ns）
//    行缺失延迟 = tRP + tRAS + tCAS + tBURST（~62 ns）—— 约 3.5x 的差距！
// 3. 预充电（Precharge, PRE）：关闭当前激活的行。需要将行缓冲区中的数据写回存储单元。
//    完整访问流程：PRE（关闭旧行）→ RAS（激活新行）→ CAS（列访问）→ BURST（突发传输）。
// 4. 突发传输（Burst Transfer）：一次 CAS 命令可连续传输多个数据节拍（burstLength）。
//    DDR4 典型 burst 长度为 8 拍，每次传输 dataBusWidth 位（通常 8 位/芯片）。
//    8 个芯片并行工作 → 64 位内存总线 → 每个 burst 传输 64 字节（一个 cache line）。
//    计算：8 burst × 64 bit / 8 = 64 bytes。
// 5. FR-FCFS（First-Ready, First-Come-First-Serve）调度策略：
//    - 第一步：优先服务行缓冲区命中的请求（最大化行局部性/row locality）
//    - 第二步：其他请求按 FIFO 顺序处理
//    这是现代内存控制器的核心调度策略，对性能有显著影响。
// 6. DRAM 时序参数（DDR4-like）：
//    - tRC（Row Cycle）= tRP + tRAS + tCAS：完整的行周期时间
//    - tRP（Row Precharge）：预充电时间
//    - tRAS（Row Access Strobe）：行激活时间
//    - tCAS（Column Access Strobe）：列访问时间
//    - tRRD（Row-to-Row Delay）：连续激活两行的最小间隔
//    - tFAW（Four Activate Window）：四个激活窗口约束
//    - tBURST：突发传输时间
// 7. DRAM 与 SRAM 的层次差异：
//    - SRAM（片上 cache/shared memory）：访问延迟 ~1-5 个周期，能量 ~5 pJ/操作
//    - DRAM（片外 HBM/DDR4）：访问延迟 ~50-200 ns，能量 ~640-1200 pJ/操作
//    - 能量比例：DRAM/SRAM ≈ 1200/26 ≈ 46x，DRAM/FP32 ≈ 640/0.9 ≈ 711x
//    - 关键推论：重新计算值通常比从 DRAM 存储和重新加载更省能耗！
// 8. HBM（High Bandwidth Memory）优势：
//    - 3D 堆叠：多层 DRAM die 通过 TSV（硅通孔）垂直堆叠
//    - 硅中介层（Silicon Interposer）：高密度互连
//    - H100：6 个 HBM3 堆栈 × 1024 位 = 6144 位接口 → 峰值 3.2 TB/s
//    - 对比：双通道 DDR4-2400 仅 ~38.4 GB/s（HBM 宽约 83 倍！）
//    - 每比特能耗比 GDDR5 降低 94%（AMD 估算）
// 9. DIMM 组织：8 个 DRAM 芯片 → 64 位总线（一个 rank）。物理地址以字节粒度
//    交错分布在芯片间。内存控制器将物理地址映射为 (bank, row, column) 三元组。
//
// Stanford CS149, Fall 2025 - Lecture 12: Mapping AI to the AI Datacenter

#include <iostream>
#include <vector>
#include <queue>
#include <iomanip>
#include <string>
#include <cassert>
#include <algorithm>
#include <random>

// 时间单位：纳秒 (ns)
using Time = double;

// DRAM 时序参数（DDR4 风格）
// 这些参数决定了内存访问延迟的各个组成部分
struct DRAMTiming {
    double tRC_ns = 45.0;     // 行周期时间（PRE + RAS + CAS 的完整周期）
    double tRAS_ns = 32.0;    // 行地址选通（Row Access Strobe）：激活行所需时间
    double tRP_ns = 13.0;     // 行预充电时间（Row Precharge）：关闭当前行的时间
    double tCAS_ns = 13.0;    // 列地址选通（Column Access Strobe）：列访问时间
    double tBURST_ns = 4.0;   // 突发传输时间（8 拍连续数据）
    double tRRD_ns = 6.0;     // 行激活到行激活间隔（Row-to-Row Delay）
    double tFAW_ns = 30.0;    // 四激活窗口（Four Activate Window）
    double tCCD_ns = 5.0;     // 列到列延迟（Column-to-Column Delay）
    int burstLength = 8;      // 每次列访问的数据节拍数
    int dataBusWidth = 8;     // 每个 DRAM 芯片的数据总线宽度（位）
    int numBanks = 8;         // 每个 DRAM 芯片的 bank 数量
    int rowsPerBank = 16384;  // 每个 bank 的行数
    int colsPerRow = 1024;    // 每行的列数（每列 = burstLength * dataBusWidth 位）
};

// 单个 DRAM Bank（存储体）
// 每个 bank 独立运作，拥有自己的行缓冲区和状态
class DRAMBank {
public:
    DRAMBank(int bankId, const DRAMTiming& timing)
        : bankId_(bankId), timing_(timing),
          rowBuffer_(timing.colsPerRow * timing.burstLength * timing.dataBusWidth / 8, 0),
          openRow_(-1),          // -1 表示没有打开的行
          rowBufferValid_(false), // 行缓冲区数据是否有效
          busyUntil_(0.0), stats_{0, 0} {}

    // 处理对此 bank 的读请求
    // 返回延迟（纳秒）
    Time read(int row, int col, Time currentTime) {
        Time startTime = std::max(currentTime, busyUntil_);
        Time latency = 0.0;

        if (openRow_ == row && rowBufferValid_) {
            // 行缓冲区命中（row buffer hit）：目标行已经激活，只需列访问
            latency = timing_.tCAS_ns + timing_.tBURST_ns;
            stats_.rowHits++;
        } else {
            // 行缓冲区缺失（row buffer miss）：需要先关闭当前行，再激活目标行
            if (rowBufferValid_) {
                // 将当前行缓冲区的数据写回存储单元（预充电）
                latency += timing_.tRP_ns;  // 预充电时间
            }
            latency += timing_.tRAS_ns;      // 激活新行
            latency += timing_.tCAS_ns;      // 列访问
            latency += timing_.tBURST_ns;    // 突发传输

            openRow_ = row;
            rowBufferValid_ = true;
            stats_.rowMisses++;
        }

        busyUntil_ = startTime + latency;
        stats_.totalRequests++;
        return latency;
    }

    // 预充电（Precharge）：关闭当前打开的行
    // 强制将行缓冲区数据写回并清空缓冲区
    void precharge(Time currentTime) {
        if (rowBufferValid_) {
            busyUntil_ = std::max(currentTime, busyUntil_) + timing_.tRP_ns;
            rowBufferValid_ = false;
            openRow_ = -1;
        }
    }

    // 检查 bank 是否就绪（可以接受新请求）
    bool isReady(Time currentTime) const {
        return busyUntil_ <= currentTime;
    }

    int openRow() const { return openRow_; }
    bool rowBufferValid() const { return rowBufferValid_; }
    int bankId() const { return bankId_; }

    // Bank 统计信息（行命中/缺失计数）
    struct Stats {
        long long totalRequests = 0;
        long long rowHits = 0;
        long long rowMisses = 0;
    };
    const Stats& stats() const { return stats_; }

private:
    int bankId_;
    DRAMTiming timing_;
    std::vector<uint8_t> rowBuffer_;  // 行缓冲区存储
    int openRow_;                     // 当前打开的行号（-1 = 无）
    bool rowBufferValid_;             // 行缓冲区数据是否有效
    Time busyUntil_;                  // bank 忙碌到何时
    Stats stats_;                     // 统计计数器
};

// 来自 LLC（Last Level Cache，末级缓存）的内存请求
struct MemRequest {
    int id;               // 请求 ID
    int bankId;           // 目标 bank
    int row;              // 目标行
    int col;              // 目标列
    Time arrivalTime;     // 到达时间
    Time completionTime;  // 完成时间
    bool isRead;          // 是否为读请求
};

// 内存控制器：使用 FR-FCFS 调度策略
// FR-FCFS = First-Ready（优先就绪）, First-Come-First-Serve（先来先服务）
// 分两级调度：
//   1. 优先服务行缓冲区命中的请求（最大化行局部性/吞吐量）
//   2. 其余请求按 FIFO 顺序处理（保证公平性）
class MemoryController {
public:
    MemoryController(int numBanks, const DRAMTiming& timing)
        : timing_(timing), currentTime_(0.0) {
        for (int b = 0; b < numBanks; ++b) {
            banks_.emplace_back(b, timing);
        }
    }

    // 向控制器提交一个内存请求
    void submitRequest(int bankId, int row, int col, bool isRead = true) {
        requestQueue_.push_back({nextReqId_++, bankId, row, col, currentTime_, 0.0, isRead});
    }

    // 处理所有待处理请求
    void processAll() {
        while (!requestQueue_.empty()) {
            // === FR-FCFS 调度逻辑 ===
            // 第一步：在请求队列中查找行缓冲区命中的请求
            auto hitIt = requestQueue_.end();
            auto firstIt = requestQueue_.begin();

            // 搜索第一个行缓冲区命中的请求（优先处理）
            for (auto it = requestQueue_.begin(); it != requestQueue_.end(); ++it) {
                int b = it->bankId;
                if (banks_[b].isReady(currentTime_) &&      // bank 空闲
                    banks_[b].rowBufferValid() &&            // 行缓冲区有效
                    banks_[b].openRow() == it->row) {       // 命中了当前激活的行
                    hitIt = it;
                    break;
                }
            }

            // 第二步：如果没有命中请求，查找第一个可响应的请求（FIFO 顺序）
            auto readyIt = requestQueue_.end();
            if (hitIt == requestQueue_.end()) {
                for (auto it = requestQueue_.begin(); it != requestQueue_.end(); ++it) {
                    if (banks_[it->bankId].isReady(currentTime_)) {
                        readyIt = it;
                        break;
                    }
                }
            }

            // 选择最终要服务的请求
            auto chosenIt = (hitIt != requestQueue_.end()) ? hitIt : readyIt;

            if (chosenIt == requestQueue_.end()) {
                // 当前没有请求可以被服务：推进时间到下一个 bank 就绪的时刻
                Time nextReady = 1e9;
                for (auto& b : banks_) {
                    if (!b.isReady(currentTime_)) {
                        nextReady = std::min(nextReady, b.isReady(currentTime_) ?
                                             currentTime_ : currentTime_ + 1.0);
                    }
                }
                currentTime_ = std::max(currentTime_ + 1.0, nextReady);
                continue;
            }

            // 服务选中的请求
            int b = chosenIt->bankId;
            Time latency = banks_[b].read(chosenIt->row, chosenIt->col, currentTime_);
            currentTime_ += latency;
            chosenIt->completionTime = currentTime_;
            completedRequests_.push_back(*chosenIt);
            requestQueue_.erase(chosenIt);
        }
        totalTime_ = currentTime_;
    }

    // 打印内存控制器统计信息
    void printStats() const {
        std::cout << "══════════════════════════════════════════════════════════════\n";
        std::cout << "内存控制器统计 (FR-FCFS 调度策略)\n";
        std::cout << "══════════════════════════════════════════════════════════════\n\n";

        std::cout << "总时间: " << std::fixed << std::setprecision(1)
                  << totalTime_ << " ns\n";
        std::cout << "已完成的请求数: " << completedRequests_.size() << "\n\n";

        std::cout << std::left
                  << std::setw(10) << "Bank"
                  << std::setw(14) << "请求数"
                  << std::setw(10) << "行命中"
                  << std::setw(12) << "行缺失"
                  << std::setw(10) << "命中率\n";
        std::cout << std::string(56, '-') << "\n";

        long long totalReqs = 0, totalHits = 0, totalMisses = 0;
        for (const auto& bank : banks_) {
            auto& s = bank.stats();
            double hitRate = s.totalRequests > 0 ?
                100.0 * s.rowHits / s.totalRequests : 0.0;
            std::cout << std::left
                      << std::setw(10) << bank.bankId()
                      << std::setw(14) << s.totalRequests
                      << std::setw(10) << s.rowHits
                      << std::setw(12) << s.rowMisses
                      << std::fixed << std::setprecision(1) << hitRate << "%\n";
            totalReqs += s.totalRequests;
            totalHits += s.rowHits;
            totalMisses += s.rowMisses;
        }

        std::cout << std::string(56, '-') << "\n";
        double overallHitRate = totalReqs > 0 ? 100.0 * totalHits / totalReqs : 0.0;
        std::cout << std::left
                  << std::setw(10) << "总计"
                  << std::setw(14) << totalReqs
                  << std::setw(10) << totalHits
                  << std::setw(12) << totalMisses
                  << std::fixed << std::setprecision(1) << overallHitRate << "%\n\n";
    }

    // 计算有效带宽（字节/秒）
    // 有效带宽 = 总传输字节数 / 总时间
    double effectiveBandwidth() const {
        if (totalTime_ == 0) return 0.0;
        double totalBytes = completedRequests_.size() *
                            timing_.burstLength * timing_.dataBusWidth / 8.0;
        return totalBytes / (totalTime_ * 1e-9);  // 转换为字节/秒
    }

    // 打印延迟分解（最佳/最差情况）
    void printTimingBreakdown() const {
        std::cout << "时序参数：\n";
        std::cout << "  行周期 (tRC):      " << timing_.tRC_ns << " ns\n";
        std::cout << "  行激活 (tRAS):     " << timing_.tRAS_ns << " ns\n";
        std::cout << "  预充电 (tRP):      " << timing_.tRP_ns << " ns\n";
        std::cout << "  列访问 (tCAS):     " << timing_.tCAS_ns << " ns\n";
        std::cout << "  突发传输 (Burst):  " << timing_.tBURST_ns << " ns ("
                  << timing_.burstLength << " 拍)\n\n";

        std::cout << "最佳延迟（行命中）:  CAS + Burst = "
                  << (timing_.tCAS_ns + timing_.tBURST_ns) << " ns\n";
        std::cout << "最差延迟（行缺失）:  PRE + RAS + CAS + Burst = "
                  << (timing_.tRP_ns + timing_.tRAS_ns + timing_.tCAS_ns + timing_.tBURST_ns)
                  << " ns\n\n";
    }

    double totalTime() const { return totalTime_; }

private:
    DRAMTiming timing_;
    std::vector<DRAMBank> banks_;            // 所有 DRAM bank
    std::vector<MemRequest> requestQueue_;   // 请求队列
    std::vector<MemRequest> completedRequests_;  // 已完成的请求列表
    Time currentTime_;
    Time totalTime_;
    int nextReqId_ = 0;
};

// === 三种不同的内存访问模式模拟 ===

// 顺序访问（Sequential Access）
// 同一行内连续列 → 行缓冲区命中率极高
// 这是 GPU 上矩阵乘法等计算的典型访存模式
void simulateSequentialAccess(MemoryController& mc, int numRequests) {
    for (int i = 0; i < numRequests; ++i) {
        int bank = i % 8;          // 轮转 bank（bank 交错）
        int row = i / 1024;        // 行号按列数推进
        int col = i % 1024;        // 同行的连续列
        mc.submitRequest(bank, row, col);
    }
}

// 随机访问（Random Access）
// 随机 bank、随机行、随机列 → 行缓冲区命中率极低
// 代表最差情况的访存模式
void simulateRandomAccess(MemoryController& mc, int numRequests,
                          std::mt19937& rng) {
    std::uniform_int_distribution<int> bankDist(0, 7);
    std::uniform_int_distribution<int> rowDist(0, 16383);
    std::uniform_int_distribution<int> colDist(0, 1023);

    for (int i = 0; i < numRequests; ++i) {
        mc.submitRequest(bankDist(rng), rowDist(rng), colDist(rng));
    }
}

// 跨步访问（Strided Access）
// 每次跳过 `stride` 行 → 命中率介于顺序和随机之间
// 模拟矩阵转置、卷积等访存模式
void simulateStridedAccess(MemoryController& mc, int numRequests, int stride) {
    for (int i = 0; i < numRequests; ++i) {
        int bank = (i * stride) % 8;           // 跨步后的 bank
        int row = (i * stride / 8) % 16384;     // 跨步后的行
        int col = i % 1024;                     // 连续列
        mc.submitRequest(bank, row, col);
    }
}

// 数据移动能耗分析
// 基于 Lecture 12 中引用的经典数据（Han, ICLR 2016; Bill Dally）
void analyzeEnergyCost() {
    std::cout << "══════════════════════════════════════════════════════════════\n";
    std::cout << "数据移动能耗分析\n";
    std::cout << "══════════════════════════════════════════════════════════════\n\n";

    struct EnergyEntry {
        std::string operation;   // 操作类型
        double energy_pJ;        // 能耗（皮焦耳, pJ）
        std::string note;        // 数据来源/说明
    };

    std::vector<EnergyEntry> costs = {
        {"FP32 数学运算",             0.9,  "45nm CMOS 工艺 (Han, ICLR 2016)"},
        {"本地 SRAM 访问",            5.0,  "片上，约 1mm 距离"},
        {"从 LPDDR 加载 32 位",       640.0, "片外 DRAM 访问"},
        {"从 SRAM 读取 64 位",        26.0,  "片上 (Bill Dally 数据)"},
        {"从 LPDDR 读取 64 位",       1200.0,"片外，移动端 DRAM"},
        {"从 DRAM 读取 10 GB/s",      1.6,   "约 1.6 瓦特总功耗（每秒）"},
    };

    std::cout << std::left
              << std::setw(30) << "操作"
              << std::setw(15) << "能耗"
              << "说明\n";
    std::cout << std::string(70, '-') << "\n";

    for (const auto& c : costs) {
        std::cout << std::left
                  << std::setw(30) << c.operation
                  << std::setw(15) << std::fixed << std::setprecision(1)
                  << c.energy_pJ << " pJ"
                  << c.note << "\n";
    }

    std::cout << "\n关键比值：\n";
    std::cout << "  DRAM/SRAM 能耗比例: " << 1200.0/26.0 << "x\n";
    std::cout << "  DRAM/FP32 计算能耗比例: " << 640.0/0.9 << "x\n";
    std::cout << "\n重要推论：重新计算值往往比从 DRAM 存储并重新加载更省能耗！\n";
    std::cout << "这就是 kernel fusion（内核融合）和 checkpointing（检查点）策略的能耗动机。\n\n";
}

int main() {
    std::cout << "=== Lecture 12：DRAM 模拟器 ===\n";
    std::cout << "Stanford CS149 - 将 AI 映射到 AI 数据中心\n\n";

    DRAMTiming timing;
    std::random_device rd;
    std::mt19937 rng(rd());

    // 场景 1：顺序访问（良好的局部性）
    {
        std::cout << "══════════════════════════════════════════════════════════════\n";
        std::cout << "场景 1：顺序访问（高局部性）\n";
        std::cout << "══════════════════════════════════════════════════════════════\n\n";

        MemoryController mc(timing.numBanks, timing);
        simulateSequentialAccess(mc, 64);
        mc.processAll();
        mc.printStats();
        std::cout << "有效带宽: " << std::fixed << std::setprecision(1)
                  << mc.effectiveBandwidth() / 1e9 << " GB/s\n\n";
        mc.printTimingBreakdown();
    }

    // 场景 2：随机访问（极差的局部性）
    {
        std::cout << "══════════════════════════════════════════════════════════════\n";
        std::cout << "场景 2：随机访问（低局部性）\n";
        std::cout << "══════════════════════════════════════════════════════════════\n\n";

        MemoryController mc(timing.numBanks, timing);
        simulateRandomAccess(mc, 64, rng);
        mc.processAll();
        mc.printStats();
        std::cout << "有效带宽: " << std::fixed << std::setprecision(1)
                  << mc.effectiveBandwidth() / 1e9 << " GB/s\n\n";

        std::cout << "对比分析：顺序 vs 随机\n";
        std::cout << "  顺序访问：行缓冲区命中率高 → 低延迟\n";
        std::cout << "  随机访问：行缓冲区频繁抖动（thrashing） → 高延迟、低带宽\n";
        std::cout << "  FR-FCFS 通过优先处理命中行的请求来缓解随机访问的性能损失\n\n";
    }

    // 场景 3：模拟真实 GPU 内存访问模式
    {
        std::cout << "══════════════════════════════════════════════════════════════\n";
        std::cout << "场景 3：GPU 风格的内存访问\n";
        std::cout << "══════════════════════════════════════════════════════════════\n\n";

        std::cout << "CPU 内存子系统 (DRAM/DDR)：\n";
        std::cout << "  - 每个通道 64 位内存总线\n";
        std::cout << "  - DDR4 2400：每通道 19.2 GB/s，双通道约 38.4 GB/s\n";
        std::cout << "  - 约 13 ns 的 CAS 延迟\n\n";

        std::cout << "GPU 内存子系统 (HBM)：\n";
        std::cout << "  - H100：6 个 HBM3 堆栈 × 1024 位 = 6144 位接口\n";
        std::cout << "  - 峰值带宽：3.2 TB/s（比双通道 DDR4 宽 83 倍！）\n";
        std::cout << "  - 3D 堆叠 DRAM：通过 TSV（硅通孔）垂直连接各层芯片\n";
        std::cout << "  - 硅中介层（silicon interposer）提供高带宽互连\n\n";

        std::cout << "DIMM 组织结构：\n";
        std::cout << "  - 8 个 DRAM 芯片 → 64 位总线（一个 rank）\n";
        std::cout << "  - 物理地址以字节粒度交错分布在各芯片间\n";
        std::cout << "  - 64 字节 cache line：8 个 burst × 64 位（所有芯片并行）\n";
        std::cout << "  - 内存控制器将物理地址映射为：bank → row → column\n\n";

        std::cout << "HBM 优势：\n";
        std::cout << "  - 更高带宽：每堆栈 1024 位（vs DDR4 的 64 位）\n";
        std::cout << "  - 更高能效：导线更短，电容更小\n";
        std::cout << "  - 更小体积：3D 堆叠减少 PCB 面积需求\n";
        std::cout << "  - 每比特能耗比 GDDR5 降低 94%（AMD 估算数据）\n\n";
    }

    // 能耗分析部分
    analyzeEnergyCost();

    // 总结
    std::cout << "══════════════════════════════════════════════════════════════\n";
    std::cout << "关键要点：\n";
    std::cout << "1. DRAM 延迟取决于行缓冲区状态（命中 vs 缺失，差距约 3.5x）\n";
    std::cout << "2. FR-FCFS 调度优先处理行命中请求以最大化吞吐量\n";
    std::cout << "3. 多个 bank 支持请求流水线化 → 提高引脚利用率\n";
    std::cout << "4. HBM：3D 堆叠 + 宽接口 → H100 达 3.2 TB/s\n";
    std::cout << "5. 数据移动主导能耗：DRAM 访问 ≈ FP32 运算的约 700 倍能耗\n";
    std::cout << "6. 核心原则：将数据尽量靠近处理器，减少数据移动量\n";
    std::cout << "══════════════════════════════════════════════════════════════\n";

    return 0;
}
