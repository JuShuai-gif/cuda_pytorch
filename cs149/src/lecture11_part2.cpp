// lecture11_part2.cpp
// Metapipelining（元流水线）模拟
// 层次化粗粒度流水线（"流水线中的流水线"）
// 模拟 SambaNova metapipelining 和流式数据流执行模型
//
// 核心概念详解：
// 1. Metapipelining（元流水线）：一种层次化的粗粒度流水线技术，在 SambaNova SN40L
//    RDU（可重构数据流单元）上实现。其核心思想是用编译器将嵌套循环自动转换为
//    多级流水线，使得外层和内层循环的各阶段可以并行执行。
//    语法：METAPIPE(M/MM) { METAPIPE(N/NN) { LOAD_TILE; MAT_MUL; STORE_TILE } }
// 2. 数据流执行（Dataflow Execution）：与传统指令驱动（instruction-driven）的执行不同，
//    数据流架构通过 token 传递触发计算——当所有操作数就绪时，操作自动执行。
//    无需取指/译码循环，消除了指令流水线开销。
// 3. 双缓冲（Double Buffering）：使用两个缓冲区交替读写，使得生产者可以在消费者
//    使用当前数据的同时向另一个缓冲区写入新数据，实现无锁的流水线重叠。
// 4. Token 控制的数据流（Token-Controlled Dataflow）：区别于 GPU 上基于锁（mbarrier）
//    的同步机制，SambaNova 通过 token 传递实现同步——一个阶段完成后产生 token，
//    下游阶段收到 token 后自动开始执行。零同步开销。
// 5. 内核融合（Kernel Fusion）：将多个独立的 CUDA kernel 合并为一个数据流流水线，
//    避免中间结果写入 HBM 后再读取的往返开销。RDU 上 520MB 片上 SRAM 使得整个
//    decoder 层可以在一个 kernel 中完成（vs GPU 需数百个 kernel）。
// 6. 数据并行模式（Data-Parallel Patterns）：Map（逐元素操作）、Zip（逐元素二元操作）、
//    Reduce（归约）、GEMM（矩阵乘法）——编译器将这些模式映射到 PCU（Pattern Compute Unit）
//    和 PMU（Pattern Memory Unit）上。
// 7. 矩阵平铺策略（Tiling）的关键作用：将大矩阵分解为 tile，使得每个 tile 的计算
//    所需数据可以被片上 SRAM 容纳，减少 HBM 往返次数。
//    理想加速比 ≈ TotalTiles * max(stageLatency) 而非 TotalTiles * sum(stageLatencies)。
// 8. AGCU（Address Generation and Communication Unit）→ PCU → PMU 的数据流路径：
//    AGCU 负责地址生成和片外数据加载，PMU 提供片上模式存储，PCU 执行脉动计算。
//
// Stanford CS149, Fall 2025 - Lecture 11: Programming Specialized Hardware for AI

#include <iostream>
#include <vector>
#include <queue>
#include <iomanip>
#include <string>
#include <cassert>

// 时间单位（周期数或纳秒）
using Time = long long;

// 在流水线中流动的数据 tile
struct DataTile {
    int id;
    int m_idx;   // M 维度上的 tile 索引
    int n_idx;   // N 维度上的 tile 索引
    Time readyTime = 0;  // 数据就绪的时间点
};

// 流水线阶段（Stage）：处理 tile 数据并传递给下一阶段
// 每个阶段有独立的处理延迟，多个阶段可以并行执行（流水线重叠）
class PipelineStage {
public:
    PipelineStage(const std::string& name, Time latency, int capacity = 1)
        : name_(name), latency_(latency), capacity_(capacity), busyUntil_(0) {}

    // 处理一个 tile：返回处理完成时的时间点
    // inputReady: 输入数据就绪的时间
    // currentTime: 当前系统时间
    Time process(Time inputReady, Time currentTime) {
        // 开始时间 = max(输入就绪时间, 本阶段空闲时间, 当前时间)
        Time start = std::max(inputReady, std::max(busyUntil_, currentTime));
        busyUntil_ = start + latency_;
        return busyUntil_;
    }

    // 检查当前阶段是否可以接收新数据
    bool canAccept(Time currentTime) const {
        return busyUntil_ <= currentTime;
    }

    const std::string& name() const { return name_; }
    Time latency() const { return latency_; }

private:
    std::string name_;
    Time latency_;
    int capacity_;
    Time busyUntil_;  // 本阶段忙到何时
};

// 双缓冲（Double Buffer）：允许同时进行读和写操作
// 生产者写入一个缓冲，消费者同时从另一个缓冲读取
// 通过交换读写指针实现无锁同步
class DoubleBuffer {
public:
    DoubleBuffer(const std::string& name, int size)
        : name_(name), size_(size),
          writeBuffer_(0), readBuffer_(1),
          writeReady_(0), readReady_(0),
          writeBusy_(0), readBusy_(0) {}

    // 生产者写入缓冲区（写完即交换指针让消费者可见）
    Time write(Time dataReady) {
        Time start = std::max(dataReady, writeBusy_);
        writeBusy_ = start + 1;  // 1 个周期用于交换缓冲区指针
        writeReady_ = writeBusy_;
        std::swap(writeBuffer_, readBuffer_);  // 原子性地交换读写指针
        return writeReady_;
    }

    // 消费者从缓冲区读取
    Time read(Time requestTime) {
        Time start = std::max(requestTime, std::max(readReady_, readBusy_));
        readBusy_ = start + 1;  // 1 个周期用于交换缓冲区指针
        return readReady_;
    }

    const std::string& name() const { return name_; }

private:
    std::string name_;
    int size_;
    int writeBuffer_, readBuffer_;  // 写入缓冲区和读取缓冲区的索引
    Time writeReady_, readReady_;   // 写/读就绪的时间点
    Time writeBusy_, readBusy_;     // 写/读忙到何时
};

// Metapipeline（元流水线）：带有嵌套阶段的层次化流水线
// 模型：METAPIPE(M/MM) { METAPIPE(N/NN) { ... } }
// M 维度的外层循环展开为最外层流水线，N 维度的内层循环展开为内层流水线
// 内层流水线对外层流水线完全重叠执行
class MetaPipeline {
public:
    MetaPipeline(int M, int N, int K,
                 int tileM, int tileN,
                 Time loadLatency, Time computeLatency, Time storeLatency)
        : M_(M), N_(N), K_(K),
          tileM_(tileM), tileN_(tileN),
          loadLatency_(loadLatency),
          computeLatency_(computeLatency),
          storeLatency_(storeLatency),
          numM_Tiles_(M / tileM),
          numN_Tiles_(N / tileN),
          totalTime_(0) {}

    // 模拟 metapipeline 执行并返回总执行时间
    Time execute() {
        // METAPIPE over M dimension（外层循环，沿M维度流水化）
        std::vector<Time> a_load_complete(numM_Tiles_, 0);

        for (int m = 0; m < numM_Tiles_; ++m) {
            // LOAD_TILE A：为此 M 迭代加载 a_tile
            Time aLoadStart = (m > 0) ? a_load_complete[m-1] : 0;
            a_load_complete[m] = aLoadStart + loadLatency_;

            // METAPIPE over N dimension（内层循环，沿N维度流水化）
            // 内层流水线可以对外层流水线完全重叠，形成"流水线中的流水线"
            std::vector<Time> b_load_complete(numN_Tiles_, 0);
            std::vector<Time> compute_complete(numN_Tiles_, 0);
            std::vector<Time> store_complete(numN_Tiles_, 0);

            for (int n = 0; n < numN_Tiles_; ++n) {
                // LOAD_TILE B：异步加载 b_tile
                Time bLoadStart = std::max(
                    a_load_complete[m],     // 等待 A tile 就绪
                    (n > 0) ? b_load_complete[n-1] : 0  // 等待前一个 B tile 加载完毕
                );
                b_load_complete[n] = bLoadStart + loadLatency_;

                // MAT_MUL：计算 C = A * B（当两个 tile 都就绪时开始）
                Time compStart = std::max(
                    b_load_complete[n],     // 等待 B tile 就绪
                    (n > 0) ? compute_complete[n-1] : 0  // 等待前一个 tile 计算完成
                );
                compute_complete[n] = compStart + computeLatency_;

                // BUFFER + STORE_TILE：将结果写入双缓冲并存储
                Time storeStart = std::max(
                    compute_complete[n],    // 等待计算完成
                    (n > 0) ? store_complete[n-1] : 0  // 等待前一个 tile 存储完成
                );
                store_complete[n] = storeStart + storeLatency_;
            }

            totalTime_ = std::max(totalTime_, store_complete[numN_Tiles_ - 1]);
        }

        return totalTime_;
    }

    // 朴素顺序执行（无流水线，逐tile串行处理）
    Time executeSequential() {
        Time totalPerTile = loadLatency_ + computeLatency_ + storeLatency_;
        Time seqTime = 0;
        // 每个 M tile 加载一次 A，随后对每个 N tile 执行：加载 B + 计算 + 存储
        for (int m = 0; m < numM_Tiles_; ++m) {
            seqTime += loadLatency_;  // 加载 A tile
            for (int n = 0; n < numN_Tiles_; ++n) {
                seqTime += loadLatency_;      // 加载 B tile
                seqTime += computeLatency_;   // 计算
                seqTime += storeLatency_;     // 存储
            }
        }
        return seqTime;
    }

    // 理想时间（完全流水化，无停顿 stall）
    // 第一个 tile 需要完整延迟；后续每个 tile 只需要瓶颈阶段的延迟
    Time executeIdeal() {
        // 第一个 tile：完整延迟（冷启动惩罚）
        Time firstTile = loadLatency_ + computeLatency_ + storeLatency_;
        // 后续 tile：每个 tile 仅需瓶颈阶段的延迟
        Time stageLatency = loadLatency_;
        if (computeLatency_ > stageLatency) stageLatency = computeLatency_;
        if (storeLatency_ > stageLatency) stageLatency = storeLatency_;
        Time nTotal = numM_Tiles_ * numN_Tiles_;
        return firstTile + (nTotal - 1) * stageLatency;
    }

    void printConfig() const {
        std::cout << "矩阵维度: " << M_ << " x " << K_ << " * " << K_ << " x " << N_ << "\n";
        std::cout << "平铺策略: " << tileM_ << " x " << tileN_ << "\n";
        std::cout << "Tile 数量: " << numM_Tiles_ << " (M) x " << numN_Tiles_ << " (N) = "
                  << (numM_Tiles_ * numN_Tiles_) << " 总计\n";
        std::cout << "各阶段延迟: Load=" << loadLatency_ << "  Compute=" << computeLatency_
                  << "  Store=" << storeLatency_ << "\n";
    }

private:
    int M_, N_, K_;
    int tileM_, tileN_;
    Time loadLatency_, computeLatency_, storeLatency_;
    int numM_Tiles_, numN_Tiles_;
    Time totalTime_;
};

// FlashAttention 风格的 metapipeline，用于注意力计算
// 模型：QK^T → Scale（缩放）→ Mask（掩码）→ Softmax → ×V，按 head 维度平铺
// 这是 Lecture 11 中重点讨论的内核融合与 metapipelining 的结合
class AttentionMetaPipeline {
public:
    AttentionMetaPipeline(int seqLen, int headDim, int numHeads,
                          Time matmulLatency, Time softmaxLatency, Time loadLatency)
        : seqLen_(seqLen), headDim_(headDim), numHeads_(numHeads),
          matmulLatency_(matmulLatency), softmaxLatency_(softmaxLatency),
          loadLatency_(loadLatency) {}

    void simulate() {
        int tileSize = 16;  // 16x16 tile 粒度
        int numSeqTiles = seqLen_ / tileSize;

        std::cout << "FlashAttention Metapipeline 模拟\n";
        std::cout << "  SeqLen=" << seqLen_ << ", HeadDim=" << headDim_
                  << ", Heads=" << numHeads_ << "\n\n";

        // 对每个 head 进行流水线处理：QK^T → Scale → Mask → Softmax → ×V
        for (int h = 0; h < std::min(numHeads_, 4); ++h) {
            std::cout << "  Head " << h << " 流水线：\n";

            Time qkTime = 0, scaleTime = 0, softmaxTime = 0, pvTime = 0;
            Time totalHead = 0;

            for (int t = 0; t < numSeqTiles; ++t) {
                // QK^T：查询-键矩阵乘法
                Time qkStart = (t > 0) ? qkTime : 0;
                qkTime = qkStart + matmulLatency_;

                // Scale：逐元素缩放（很快，接近零延迟）
                scaleTime = std::max(qkTime, scaleTime) + 5;

                // Softmax：沿 K 维度计算 softmax 归一化
                softmaxTime = std::max(scaleTime, softmaxTime) + softmaxLatency_;

                // PV：注意力权重与值矩阵乘法
                Time pvStart = std::max(softmaxTime, pvTime);
                pvTime = pvStart + matmulLatency_;
            }

            totalHead = qkTime;
            if (softmaxTime > totalHead) totalHead = softmaxTime;
            if (pvTime > totalHead) totalHead = pvTime;

            std::cout << "    Tile 数量: " << numSeqTiles << "\n";
            std::cout << "    QK^T 矩阵乘法时间: " << qkTime << "\n";
            std::cout << "    Softmax 时间:      " << softmaxTime << "\n";
            std::cout << "    PV 矩阵乘法时间:   " << pvTime << "\n";
            std::cout << "    Head 总时间:       " << totalHead << "\n\n";
        }

        // 内核融合（Kernel Fusion）的收益分析
        // 融合后将所有操作放在一个数据流流水线中，无需中间结果写入 HBM
        Time tilesPerHead = numSeqTiles;
        Time fusedTime = tilesPerHead * (matmulLatency_ + softmaxLatency_ +
                         matmulLatency_);
        Time unfusedTime = tilesPerHead * (matmulLatency_ + softmaxLatency_ +
                           matmulLatency_ + 3 * loadLatency_);  // 无融合时需要 3 次额外的 HBM 加载

        std::cout << "内核融合收益分析：\n";
        std::cout << "  无融合（分离 kernel）：" << unfusedTime << " 周期\n";
        std::cout << "  有融合（数据流，片内计算）：" << fusedTime << " 周期\n";
        std::cout << "  节省： " << (unfusedTime - fusedTime) << " 周期 ("
                  << std::fixed << std::setprecision(1)
                  << (100.0 * (unfusedTime - fusedTime) / unfusedTime) << "%)\n\n";

        std::cout << "RDU 优势：520 MB 片上 SRAM 支持激进的内核融合。\n";
        std::cout << "GPU 局限：仅 100 MB → 中间结果必须溢出到 HBM。\n";
    }

private:
    int seqLen_, headDim_, numHeads_;
    Time matmulLatency_, softmaxLatency_, loadLatency_;
};

int main() {
    std::cout << "=== Lecture 11：Metapipelining 模拟 ===\n";
    std::cout << "Stanford CS149 - 面向 AI 的专用硬件编程\n\n";

    // 第1部分：基本矩阵乘法 metapipeline
    {
        std::cout << "══════════════════════════════════════════════════════════════\n";
        std::cout << "第1部分：矩阵乘法 Metapipeline\n";
        std::cout << "METAPIPE(M/MM) { METAPIPE(N/NN) { MAT_MUL } }\n";
        std::cout << "══════════════════════════════════════════════════════════════\n\n";

        // 矩阵：1024x1024 * 1024x1024，按 256x64 平铺
        MetaPipeline mp(1024, 1024, 1024, 256, 64,
                        100,   // 加载延迟（AGCU → PMU）
                        500,   // 计算延迟（PCU 脉动阵列计算）
                        50);   // 存储延迟（PMU → AGCU）

        mp.printConfig();

        Time seqTime = mp.executeSequential();  // 顺序（无流水线）
        Time mpTime = mp.execute();             // Metapipelining
        Time idealTime = mp.executeIdeal();     // 理想（完全重叠）

        std::cout << "\n结果对比：\n";
        std::cout << "  顺序执行（无流水线）：     " << seqTime << " 周期\n";
        std::cout << "  Metapipelining：           " << mpTime << " 周期\n";
        std::cout << "  理想情况（完全重叠）：     " << idealTime << " 周期\n";
        std::cout << "  加速比（vs 顺序）：        " << std::fixed << std::setprecision(2)
                  << (double)seqTime / mpTime << "x\n";
        std::cout << "  效率（vs 理想）：          " << std::setprecision(1)
                  << (100.0 * idealTime / mpTime) << "%\n\n";

        std::cout << "关键洞察：metapipelining 将嵌套循环转换为\n";
        std::cout << "流式流水线。各阶段可并行执行。\n";
        std::cout << "中间数据存储在双缓冲中。\n";
        std::cout << "与平铺（tiling）和内核融合（kernel fusion）协同工作。\n\n";
    }

    // 第2部分：比较不同流水线深度
    {
        std::cout << "══════════════════════════════════════════════════════════════\n";
        std::cout << "第2部分：流水线深度对比\n";
        std::cout << "══════════════════════════════════════════════════════════════\n\n";

        struct Config {
            int M, N, K, tileM, tileN;
            Time loadL, compL, storeL;
            std::string label;
        };

        std::vector<Config> configs = {
            {256, 256, 256, 64, 64, 100, 500, 50, "小规模 (256x256)，计算受限"},
            {4096, 4096, 4096, 256, 64, 100, 500, 50, "大规模 (4096x4096)"},
            {8192, 8192, 8192, 256, 64, 200, 500, 100, "超大 (8192x8192)，访存受限"},
        };

        std::cout << std::left
                  << std::setw(40) << "配置"
                  << std::setw(15) << "顺序执行"
                  << std::setw(15) << "Metapipe"
                  << "加速比\n";
        std::cout << std::string(85, '-') << "\n";

        for (const auto& c : configs) {
            MetaPipeline mp(c.M, c.N, c.K, c.tileM, c.tileN,
                           c.loadL, c.compL, c.storeL);
            Time seq = mp.executeSequential();
            Time met = mp.execute();

            std::cout << std::left
                      << std::setw(40) << c.label
                      << std::setw(15) << seq
                      << std::setw(15) << met
                      << std::fixed << std::setprecision(2) << (double)seq / met << "x\n";
        }
        std::cout << "\n";
    }

    // 第3部分：FlashAttention 风格的 metapipeline
    {
        std::cout << "══════════════════════════════════════════════════════════════\n";
        std::cout << "第3部分：FlashAttention Metapipeline\n";
        std::cout << "══════════════════════════════════════════════════════════════\n\n";

        AttentionMetaPipeline attn(2048, 64, 32,
                                   200,   // 矩阵乘法延迟
                                   100,   // softmax 延迟
                                   150);  // 加载延迟
        attn.simulate();
    }

    // 第4部分：总结 —— ThunderKittens vs Metapipelining 编程模型对比
    {
        std::cout << "══════════════════════════════════════════════════════════════\n";
        std::cout << "第4部分：编程模型对比\n";
        std::cout << "══════════════════════════════════════════════════════════════\n\n";

        std::cout << "ThunderKittens（GPU H100/B100）：\n";
        std::cout << "   - 嵌入式 CUDA DSL，以 16x16 tile 为原语\n";
        std::cout << "   - 生产者-消费者流水线：TMA 加载 + MMA 计算\n";
        std::cout << "   - Warp 分组：8 个消费者 warp，4 个生产者 warp\n";
        std::cout << "   - mbarrier 同步机制用于异步协调\n";
        std::cout << "   - B100：单线程 MMA，无需 warp 分组，tcgen05 指令\n\n";

        std::cout << "Metapipelining（SambaNova SN40L）：\n";
        std::cout << "   - 层次化粗粒度流水线\n";
        std::cout << "   - 数据并行模式：Map, Zip, Reduce, GEMM\n";
        std::cout << "   - 双缓冲用于中间数据传递\n";
        std::cout << "   - Token 控制的数据流（无需锁！无同步开销）\n";
        std::cout << "   - 激进的内核融合：kernel 调用减少 100 倍\n";
        std::cout << "   - 520 MB 片上 SRAM 使整个 decoder 可在一个 kernel 内完成\n\n";

        std::cout << "两者均实现了异步性，但采用了不同的方法：\n";
        std::cout << "   GPU：硬件管理的异步（TMA）+ 软件 DSL（ThunderKittens）\n";
        std::cout << "   RDU：编译器管理的空间调度 + metapipelining\n";
    }

    return 0;
}
