// lecture12_part1.cpp
// 分布式矩阵乘法与集合通信
// 模拟：张量并行 (Tensor Parallel) GEMM，使用 Reduce-Scatter / AllReduce 集合通信原语
//
// 核心概念详解 —— 将 AI 模型映射到 AI 数据中心：
// 1. 张量并行 (Tensor Parallel, TP)：沿隐藏维度（hidden dimension）切分权重矩阵，
//    每个 rank 持有权重的一部分。GEMM 计算后通过 Reduce-Scatter 汇总部分结果，
//    再用 All-Gather 分发完整结果。适合单节点内的高带宽通信（如 NVLink 900 GB/s）。
// 2. 流水线并行 (Pipeline Parallel, PP)：沿层维度切分模型，不同 rank 负责不同层，
//    通过点对点（P2P）Send/Recv 传递激活值。通信量小但存在流水线气泡（bubble）。
// 3. 数据并行 (Data Parallel, DP)：每个 rank 持有完整模型副本，独立处理不同 mini-batch，
//    通过 AllReduce 同步梯度。通信量与模型参数量成线性关系。
// 4. Reduce-Scatter（规约-分散）：每个 rank 发送完整的部分结果给所有 rank，
//    经 reduction（求和/平均）后，每个 rank 仅保留结果的 1/numRanks 片段。
//    典型的环算法（Ring Algorithm）复杂度：O((N-1) * (latency + data/N/BW))。
// 5. AllReduce（全规约）= Reduce-Scatter + All-Gather：所有 rank 最终获得完整的规约结果。
//    在训练中用于梯度同步。
// 6. All-to-All：用于专家并行（Expert Parallel, EP），每个 rank 向其他所有 rank 发送
//    不同的数据量（MoE 路由分发）。
// 7. 计算-通信重叠（Compute-Communication Overlap）：对扩展效率至关重要。
//    无重叠时，利用率随 rank 数增加急剧下降（如 8→32 socket 从 88%→52%）。
//    有重叠时（RDU 架构优势），利用率基本持平（70-79%）。
//    经典数据：BS=16, M=24576, K=131074, N=8192 的矩阵规模。
// 8. RDU 架构的通信优势：AllReduce 可与权重加载和计算完全重叠，
//    不消耗 HBM 带宽。520 MB 片上 SRAM 支持激进的内核融合，
//    减少 100x 的 kernel 调用（3 vs 800 per token）。
// 9. 扩展策略组合：total GPUs = TP × PP × DP。大模型需要更大的 TP（隐藏维度增长），
//    超大规模（530B+）还需要 PP（35-64 阶段）来减少单 rank 内存压力。
//
// Stanford CS149, Fall 2025 - Lecture 12: Mapping AI to the AI Datacenter

#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>

// 通信原语的模拟时间参数
// 这些参数决定了分布式计算中通信开销的占比
struct CommConfig {
    double bandwidth_GBs;         // 互联带宽（GB/s），如 NVLink 双向 900 GB/s
    double latency_us;            // 每条消息的固定延迟（微秒）
    int numRanks;                 // 进程（rank）数量 = GPU/RDU 数量
};

// 分布式矩阵：每个 rank 持有矩阵的一个切片（按列或按行切分）
struct DistributedMatrix {
    int globalRows, globalCols;   // 全局矩阵维度
    int localRows, localCols;     // 本地矩阵片段维度
    int rank;                     // 当前 rank 编号
    int numRanks;                 // 总 rank 数
    std::vector<std::vector<double>> localData;  // 本地矩阵数据

    DistributedMatrix(int gRows, int gCols, int rank, int numRanks, bool splitCols = true)
        : globalRows(gRows), globalCols(gCols), rank(rank), numRanks(numRanks) {
        if (splitCols) {
            // 按列切分：每个 rank 持有相同数量的行、不同范围的列
            localRows = gRows;
            localCols = (gCols + numRanks - 1) / numRanks;
        } else {
            // 按行切分：每个 rank 持有不同范围的行、相同数量的列
            localRows = (gRows + numRanks - 1) / numRanks;
            localCols = gCols;
        }

        // 以与 rank 相关的简单数据初始化（便于验证结果正确性）
        localData.resize(localRows, std::vector<double>(localCols, 0.0));
        for (int i = 0; i < localRows; ++i)
            for (int j = 0; j < localCols; ++j)
                localData[i][j] = (rank + 1) * 0.1;  // 简单初始化
    }
};

// 集合通信操作模拟
// 支持 Reduce-Scatter、All-Gather、AllReduce、All-to-All、P2P Send/Recv
class CollectiveComm {
public:
    CollectiveComm(const CommConfig& cfg) : cfg_(cfg) {}

    // Reduce-Scatter 时间估算：每个 rank 发送 data/numRanks 字节并接收相同数量
    // 使用环算法（Ring Algorithm）：共 (numRanks-1) 步，每步传输 bytesPerRank 数据
    double reduceScatterTime(size_t totalBytes) const {
        size_t bytesPerStep = totalBytes / cfg_.numRanks;
        double transferTime = bytesPerStep / (cfg_.bandwidth_GBs * 1e9) * 1e6;  // 转换为微秒
        return (cfg_.numRanks - 1) * (cfg_.latency_us + transferTime);
    }

    // All-Gather 时间估算：收集所有 rank 的切片
    // 复杂度与 Reduce-Scatter 相同（对称操作）
    double allGatherTime(size_t totalBytes) const {
        // 与 Reduce-Scatter 复杂度相同（对称操作）
        return reduceScatterTime(totalBytes);
    }

    // AllReduce 总时间 = Reduce-Scatter + All-Gather
    // Reduce-Scatter 完成规约并分散结果 → All-Gather 使所有 rank 获得完整结果
    double allReduceTime(size_t totalBytes) const {
        return reduceScatterTime(totalBytes) + allGatherTime(totalBytes);
    }

    // All-to-All 时间估算：每个 rank 向其他所有 rank 发送数据
    // 用于专家并行（MoE）中的 token 路由
    double allToAllTime(size_t bytesPerRank) const {
        double transferTime = bytesPerRank / (cfg_.bandwidth_GBs * 1e9) * 1e6;
        return (cfg_.numRanks - 1) * (cfg_.latency_us + transferTime);
    }

    // 点对点 Send/Recv 时间估算（用于流水线并行）
    double sendRecvTime(size_t bytes) const {
        return cfg_.latency_us + bytes / (cfg_.bandwidth_GBs * 1e9) * 1e6;
    }

private:
    CommConfig cfg_;
};

// 分布式 GEMM（矩阵乘法）模拟
// 沿 K 维度切分矩阵 B，各 rank 独立计算部分结果后通过 Reduce-Scatter 汇总
class DistributedGEMM {
public:
    DistributedGEMM(int M, int N, int K, int numRanks, const CommConfig& comm)
        : M_(M), N_(N), K_(K), numRanks_(numRanks), comm_(comm) {}

    void simulate() {
        std::cout << "══════════════════════════════════════════════════════════════\n";
        std::cout << "分布式 GEMM: A[" << M_ << "x" << K_ << "] * B[" << K_ << "x" << N_
                  << "] → C[" << M_ << "x" << N_ << "]\n";
        std::cout << "Rank 数量: " << numRanks_ << " | 切分维度: K 维度\n";
        std::cout << "══════════════════════════════════════════════════════════════\n\n";

        // 步骤 1：本地 GEMM 计算
        // 每个 rank 计算 A[m*k/numRanks] * B[k/numRanks*n] 的部分结果
        double flopsPerRank = 2.0 * M_ * (K_ / numRanks_) * N_;
        double totalFlops = flopsPerRank * numRanks_;
        double computeTime_us = flopsPerRank / (989e12) * 1e6;  // 假设 989 TFLOPS（如 H100 tensor core）

        std::cout << "步骤 1：本地 GEMM（每个 rank）\n";
        std::cout << "  每 rank FLOPs: " << std::scientific << std::setprecision(2)
                  << flopsPerRank << "\n";
        std::cout << "  每 rank 计算时间: " << std::fixed << std::setprecision(1)
                  << computeTime_us << " us\n\n";

        // 步骤 2：Reduce-Scatter 汇总部分结果
        // 每个 rank 持有 [M x N] 的部分结果 → reduce-scatter → [M x N/numRanks] 最终片段
        size_t resultBytes = M_ * N_ * 4;  // fp32 = 4 字节
        double rsTime = comm_.reduceScatterTime(resultBytes);

        std::cout << "步骤 2：Reduce-Scatter（规约-分散）\n";
        std::cout << "  数据量: " << resultBytes / 1e6 << " MB\n";
        std::cout << "  RS 时间: " << rsTime << " us\n\n";

        // 步骤 3：All-Gather 分发最终结果（可选，仅在需要完整结果时执行）
        double agTime = comm_.allGatherTime(resultBytes);

        std::cout << "步骤 3：All-Gather（全收集，可选，仅当需要完整结果时）\n";
        std::cout << "  AG 时间: " << agTime << " us\n\n";

        // AllReduce 路径的总时间
        double arTime = rsTime + agTime;
        double totalTime = computeTime_us + arTime;

        std::cout << "汇总：\n";
        std::cout << "  计算时间:     " << std::setw(10) << computeTime_us << " us\n";
        std::cout << "  AllReduce:    " << std::setw(10) << arTime << " us\n";
        std::cout << "  总时间:       " << std::setw(10) << totalTime << " us\n";
        std::cout << "  计算利用率:   " << std::setw(10) << std::setprecision(1)
                  << (100.0 * computeTime_us / totalTime) << "%\n\n";
    }

private:
    int M_, N_, K_, numRanks_;
    CollectiveComm comm_;
};

// 分析计算-通信重叠（RDU 的核心架构优势）
// 基于 Lecture 12 中的实验数据
void analyzeOverlap() {
    std::cout << "══════════════════════════════════════════════════════════════\n";
    std::cout << "计算-通信重叠分析\n";
    std::cout << "（基于 Lecture 数据：BS=16, M=24576, K=131074, N=8192）\n";
    std::cout << "══════════════════════════════════════════════════════════════\n\n";

    struct OverlapCase {
        int numSockets;              // Socket 数量（一个 socket = 一个 RDU）
        double totalTFLOPS;          // 总算力（TFLOPS）
        double computeRoofline_ms;   // @100% 利用率下的计算时间（ms）
        double reduceScatter_ms;     // @100% 链路利用率下的 RS 时间（ms）
        double theoreticalNoOverlap; // 无重叠时的理论利用率（%）
        double measuredWithOverlap;  // 有重叠时的实测利用率（%，来自 Lecture 数据）
    };

    std::vector<OverlapCase> cases = {
        {8,  12744, 66.3, 8.6,  88.5, 72.0},
        {16, 25488, 33.1, 9.7,  77.0, 75.0},
        {32, 50976, 16.5, 15.0, 52.0, 79.0},
    };

    std::cout << std::left
              << std::setw(14) << "Socket 数"
              << std::setw(16) << "总 TFLOPS"
              << std::setw(20) << "Roofline (ms)"
              << std::setw(18) << "RS 时间 (ms)"
              << std::setw(22) << "无重叠利用率%"
              << std::setw(18) << "有重叠利用率%"
              << "收益\n";
    std::cout << std::string(108, '-') << "\n";

    for (const auto& c : cases) {
        double gain = c.measuredWithOverlap - c.theoreticalNoOverlap;
        std::cout << std::left
                  << std::setw(14) << c.numSockets
                  << std::setw(16) << c.totalTFLOPS
                  << std::setw(20) << std::fixed << std::setprecision(1) << c.computeRoofline_ms
                  << std::setw(18) << std::setprecision(1) << c.reduceScatter_ms
                  << std::setw(22) << std::setprecision(1) << c.theoreticalNoOverlap << "%"
                  << std::setw(18) << std::setprecision(1) << c.measuredWithOverlap << "%"
                  << "+" << std::setprecision(1) << gain << "%\n";
    }

    std::cout << "\n关键洞察：无重叠时，利用率从 88% 降至 52%（8→32 socket 扩展）。\n";
    std::cout << "有重叠时（RDU），利用率在所有规模下保持 70-79%。\n";
    std::cout << "重叠策略：AllReduce 与权重加载 + 计算完全重叠。\n\n";
}

// 并行策略综合分析器
// 覆盖 TP、PP、DP、EP、SP、CP 六种策略
void analyzeParallelismStrategies() {
    std::cout << "══════════════════════════════════════════════════════════════\n";
    std::cout << "AI 训练的并行策略\n";
    std::cout << "══════════════════════════════════════════════════════════════\n\n";

    struct Strategy {
        std::string name;             // 策略名称
        std::string splitDim;         // 切分维度
        std::string commPrimitive;    // 所需通信原语
        double commVolume_GB;         // 大步长下每次迭代的通信量（GB，大模型数据）
        std::string scalingNote;      // 扩展性说明
    };

    std::vector<Strategy> strategies = {
        {"数据并行 (DP)",     "Batch 维度",        "Reduce-Scatter + All-Gather", 2.0,  "随模型大小线性增长"},
        {"张量并行 (TP)",     "隐藏维度 (Hidden)", "Reduce-Scatter + All-Gather", 8.0,  "通信量大，限节点内"},
        {"流水线并行 (PP)",   "层维度 (Layers)",   "Send-Recv (P2P 点对点)",     1.0,  "通信量小，有气泡"},
        {"专家并行 (EP)",     "MoE 专家",          "All-to-All",                 4.0,  "稀疏、选择性路由"},
        {"序列并行 (SP)",     "序列长度",          "Reduce-Scatter",             0.5,  "与 TP 合并使用"},
        {"上下文并行 (CP)",   "上下文 Token",      "AllReduce",                  1.0,  "用于长上下文场景"},
    };

    std::cout << std::left
              << std::setw(22) << "策略"
              << std::setw(16) << "切分维度"
              << std::setw(30) << "通信方式"
              << std::setw(15) << "通信量 (GB)"
              << "说明\n";
    std::cout << std::string(100, '-') << "\n";

    for (const auto& s : strategies) {
        std::cout << std::left
                  << std::setw(22) << s.name
                  << std::setw(16) << s.splitDim
                  << std::setw(30) << s.commPrimitive
                  << std::setw(15) << std::fixed << std::setprecision(1) << s.commVolume_GB
                  << s.scalingNote << "\n";
    }
    std::cout << std::endl;
}

// 分析 Lecture 中的扩展数据表（TP, PP, DP 组合）
void analyzeScalingTable() {
    std::cout << "══════════════════════════════════════════════════════════════\n";
    std::cout << "扩展配置表（来自 Lecture：序列长度 2048）\n";
    std::cout << "══════════════════════════════════════════════════════════════\n\n";

    struct ScalingEntry {
        double params_B;       // 模型参数规模（B = 十亿）
        int attentionHeads;    // 注意力头数
        int hiddenSize;        // 隐藏维度大小
        int numLayers;         // Transformer 层数
        int tpSize, ppSize, mpSize, dpSize, numGPUs;
        int batchSize;
        double peakFlopsPct;   // 峰值 FLOPS 利用率（%）
    };

    std::vector<ScalingEntry> entries = {
        {1.7,  24,  2304,  24,  1,  1,  1,  32,  32,  512,   44.0},
        {3.6,  32,  3072,  30,  2,  1,  2,  32,  64,  512,   42.0},
        {7.5,  32,  4096,  36,  4,  1,  4,  32,  128, 512,   41.0},
        {18.0, 48,  6144,  40,  8,  1,  8,  32,  256, 1024,  41.0},
        {39.0, 64,  8192,  48,  8,  2,  16, 32,  512, 1536,  41.0},
        {76.0, 80, 10240,  60,  8,  4,  32, 32, 1024, 1792,  43.0},
        {145.0,96, 12288,  80,  8,  8,  64, 24, 1536, 2304,  44.0},
        {291.0,128,16384,  90,  8, 18, 144, 15, 2160, 2430,  45.0},
        {530.0,128,20480, 105,  8, 35, 280,  9, 2520, 2520,  49.0},
        {1000.0,160,25600,128, 8, 64, 512,  6, 3072, 3072,  49.0},
    };

    std::cout << std::left
              << std::setw(10) << "参数量"
              << std::setw(8)  << "Head"
              << std::setw(10) << "隐藏维"
              << std::setw(8)  << "层数"
              << std::setw(8)  << "TP"
              << std::setw(8)  << "PP"
              << std::setw(8)  << "MP"
              << std::setw(8)  << "DP"
              << std::setw(10) << "GPU 数"
              << std::setw(12) << "% 峰值\n";
    std::cout << std::string(100, '-') << "\n";

    for (const auto& e : entries) {
        std::cout << std::left
                  << std::setw(10) << (std::to_string((int)e.params_B) + "B")
                  << std::setw(8)  << e.attentionHeads
                  << std::setw(10) << e.hiddenSize
                  << std::setw(8)  << e.numLayers
                  << std::setw(8)  << e.tpSize
                  << std::setw(8)  << e.ppSize
                  << std::setw(8)  << e.mpSize
                  << std::setw(8)  << e.dpSize
                  << std::setw(10) << e.numGPUs
                  << std::fixed << std::setprecision(1) << e.peakFlopsPct << "%\n";
    }

    std::cout << "\n观察结论：\n";
    std::cout << "  - 利用率在大模型上稳定在 41-49% 左右\n";
    std::cout << "  - TP 随模型规模增长（隐藏维度在增大）\n";
    std::cout << "  - 超大模型（530B+）需要 PP（35-64 阶段）\n";
    std::cout << "  - DP ≈ batch_size / micro_batch；总 GPU 数 = TP × PP × DP\n\n";
}

int main() {
    std::cout << "=== Lecture 12：分布式 AI 计算 ===\n";
    std::cout << "Stanford CS149 - 将 AI 映射到 AI 数据中心\n\n";

    // 第1部分：带通信的分布式 GEMM
    // NVLink 双向 900 GB/s 是 DGX 节点内 GPU 间互联的峰值带宽
    CommConfig nvlink = {
        900.0,   // 900 GB/s NVLink 双向带宽
        1.0,     // 1 us 固定延迟
        8        // DGX 节点内 8 张 GPU
    };

    DistributedGEMM dgemm(24576, 8192, 131072, 8, nvlink);
    dgemm.simulate();

    // 第2部分：计算-通信重叠分析
    analyzeOverlap();

    // 第3部分：并行策略总览
    analyzeParallelismStrategies();

    // 第4部分：扩展配置表
    analyzeScalingTable();

    // 总结
    std::cout << "══════════════════════════════════════════════════════════════\n";
    std::cout << "关键要点：\n";
    std::cout << "1. 没有计算-通信重叠时，通信可能成为主导瓶颈\n";
    std::cout << "2. RDU 优势：AllReduce 完全重叠，不消耗 HBM 带宽\n";
    std::cout << "3. TP + PP + DP 组合实现模型扩展；TP 随隐藏维度增长\n";
    std::cout << "4. RDU 上 kernel 调用减少 100 倍（3 vs 800 per token）\n";
    std::cout << "5. 数据流融合消除了 GB 级别的片外中间数据流量\n";
    std::cout << "══════════════════════════════════════════════════════════════\n";

    return 0;
}
