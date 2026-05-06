// lecture11_part1.cpp
// 异步流水线执行模拟
// 本程序模拟生产者-消费者流水线模型，展示计算与内存访问的并行重叠
// 对应 Lecture 11 的核心概念：如何在 GPU（NVIDIA H100/B100）和专用硬件上通过异步机制隐藏访存延迟
//
// 核心概念详解：
// 1. TMA (Tensor Memory Accelerator)：NVIDIA H100/B100 的硬件异步数据加载引擎，
//    可独立于计算单元运行，自动生成地址并加载 tile 数据到共享内存（shared memory）。
//    编程模型中通过 pipeline 原语管理 TMA 深度（pipelineDepth = 流水线中同时存在的请求数）。
// 2. Tensor Core MMA：矩阵乘加运算单元，以 warp group 形式组织（H100为4个warp一组），
//    以 16x16 tile 为基本粒度进行矩阵乘法。B100 进一步简化为单线程 MMA（tcgen05）。
// 3. 流水线重叠（Pipeline Overlap）：将 Load（TMA加载）、Compute（Tensor Core计算）、
//    Store（写回HBM）三阶段以流水线方式重叠执行，隐藏访存延迟。
//    关键公式：理想加速比 = (LD+COMP+ST)*N / max(LD,COMP,ST)*N ≈ 当阶段平衡时逼近 3x。
// 4. SynchronousExecutor：顺序执行每一 tile 的全流程（LD→COMP→ST），无重叠。
// 5. AsynchronousExecutor：Load 与 Compute 重叠的简化模型（两阶段流水线）。
// 6. FullPipeline：完整三阶段流水线（Load→Compute→Store），模拟 H100/B100 的
//    ThunderKittens 风格生产者-消费者流水线。
// 7. 同步原语 mbarrier：用于生产者（TMA load warp）与消费者（compute warp）之间的
//    异步协调——当 tile 数据加载完毕时通知消费者开始计算。
// 8. 吞吐量分析：当 Load 时间远小于 Compute 时（Compute-bound），异步帮助有限；
//    当 Load 时间接近或大于 Compute 时（Memory-bound），异步对性能至关重要。
//
// Stanford CS149, Fall 2025 - Lecture 11: Programming Specialized Hardware for AI

#include <iostream>
#include <vector>
#include <queue>
#include <iomanip>
#include <string>
#include <thread>
#include <chrono>
#include <climits>
#include <algorithm>
#include <cassert>

// 模拟时间单位（周期数，cycles）
using Cycles = long long;

// Tile 描述符（类似 TMA 拷贝描述符 / ThunderKittens tile 原语）
// 一个 tile 是 GPU 上数据移动和计算的基本粒度
struct Tile {
    int id;
    int rows, cols;
    Cycles loadTime;       // 从 HBM 加载到共享内存的时间
    Cycles computeTime;    // 计算时间（如 tensor core 上的 MMA 运算）
    Cycles storeTime;      // 将结果写回 HBM 的时间
};

// 流水线各阶段的时间参数配置
struct PipelineConfig {
    Cycles tmaLoadCycles;      // TMA 异步加载的延迟
    Cycles tensorCoreCycles;   // Tensor core MMA 矩阵乘法计算时间
    Cycles storeCycles;        // 结果写回 HBM 的时间
    int numTiles;              // 需要处理的总 tile 数量
    int pipelineDepth;         // 流水线深度（TMA 深度 / warp 组数量）
};

// 同步（阻塞）执行器：严格按照 LD → COMP → ST → LD → COMP → ST ... 的顺序执行
// 即每个 tile 完全结束后才开始下一个 tile，不存在任何阶段间的并行重叠
class SynchronousExecutor {
public:
    Cycles execute(const PipelineConfig& cfg) {
        Cycles total = 0;
        for (int t = 0; t < cfg.numTiles; ++t) {
            total += cfg.tmaLoadCycles;       // 加载 tile
            total += cfg.tensorCoreCycles;    // 计算
            total += cfg.storeCycles;         // 存储结果
        }
        return total;
    }
};

// 异步执行器：采用生产者-消费者流水线模式
// 模拟 ThunderKittens 风格的流水线：生产者通过 TMA 加载 tile，
// 消费者使用 tensor core 进行计算，两者在时间上重叠
class AsynchronousExecutor {
public:
    AsynchronousExecutor(int pipelineDepth)
        : pipelineDepth_(pipelineDepth) {}

    int pipelineDepth() const { return pipelineDepth_; }

    Cycles execute(const PipelineConfig& cfg) {
        Cycles total = 0;
        Cycles computeDone = 0;
        Cycles loadDone = 0;

        // 第一个 tile：必须先完成加载才能开始计算（冷启动，cold start）
        loadDone = cfg.tmaLoadCycles;

        // 流水线稳态：计算和加载在时间上重叠
        for (int t = 0; t < cfg.numTiles; ++t) {
            // 计算 tile t（只要 tile 数据已加载完毕即可开始）
            Cycles computeStart = std::max(loadDone, computeDone);
            computeDone = computeStart + cfg.tensorCoreCycles;

            // 开始加载下一个(批) tile —— 异步、非阻塞
            if (t + pipelineDepth_ < cfg.numTiles) {
                loadDone = std::max(loadDone, computeStart) + cfg.tmaLoadCycles;
            }
        }

        // 最后的存储操作（所有 tile 计算完毕后执行）
        computeDone += cfg.storeCycles;

        // 等待所有加载完成
        total = std::max(computeDone, loadDone);
        return total;
    }

private:
    int pipelineDepth_;
};

// 完整三阶段流水线：Load（加载） → Compute（计算） → Store（存储）
// 模拟 H100/B100 上 TMA + Tensor Core 的完整流水线
// 每个阶段有独立的"忙到何时"（busyUntil）时间戳，允许三阶段并行执行
class FullPipeline {
public:
    FullPipeline(int pipelineDepth) : pipelineDepth_(pipelineDepth) {}

    Cycles execute(const PipelineConfig& cfg) {
        // 为每个阶段模拟独立的时间戳
        // 生产者（producer）：TMA 加载 tile 到共享内存缓冲区
        // 消费者（consumer）：tensor core 从共享内存读取数据、计算、写入寄存器

        struct Buffer {
            Cycles loadedAt = -1;   // 数据可用的时间点
            Cycles consumedAt = -1; // 数据被消耗的时间点
        };

        std::vector<Buffer> buffers(cfg.numTiles);
        std::vector<Cycles> computeComplete(cfg.numTiles, 0);
        std::vector<Cycles> storeComplete(cfg.numTiles, 0);

        Cycles tmaBusyUntil = 0;   // TMA 单元空闲的时间
        Cycles tcBusyUntil = 0;    // Tensor Core 单元空闲的时间
        Cycles storeBusyUntil = 0; // 存储单元空闲的时间

        int nextLoad = 0;    // 下一个待加载的 tile 索引
        int nextCompute = 0; // 下一个待计算的 tile 索引
        int nextStore = 0;   // 下一个待存储的 tile 索引

        while (nextStore < cfg.numTiles) {
            // === 生产者阶段 ===
            // TMA 加载：如果缓冲区有空位且未超出流水线深度，则发起加载
            if (nextLoad < cfg.numTiles &&
                nextLoad - nextCompute < pipelineDepth_) {
                Cycles loadStart = tmaBusyUntil;
                buffers[nextLoad].loadedAt = loadStart + cfg.tmaLoadCycles;
                tmaBusyUntil = buffers[nextLoad].loadedAt;
                ++nextLoad;
            }

            // === 消费者阶段 ===
            // 如果 tile 数据已就绪（loadedAt <= tcBusyUntil），则启动计算
            if (nextCompute < cfg.numTiles &&
                nextCompute < nextLoad &&
                buffers[nextCompute].loadedAt <= tcBusyUntil) {
                Cycles compStart = std::max(tcBusyUntil, buffers[nextCompute].loadedAt);
                computeComplete[nextCompute] = compStart + cfg.tensorCoreCycles;
                tcBusyUntil = computeComplete[nextCompute];
                ++nextCompute;
            }

            // === 存储阶段 ===
            // 将计算结果写回 HBM
            if (nextStore < nextCompute &&
                computeComplete[nextStore] <= storeBusyUntil) {
                Cycles stStart = std::max(storeBusyUntil, computeComplete[nextStore]);
                storeComplete[nextStore] = stStart + cfg.storeCycles;
                storeBusyUntil = storeComplete[nextStore];
                ++nextStore;
            }

            // 时间推进：如果当前没有可执行的操作，快进到下一个可用事件时间点
            Cycles nextEvent = std::min({
                (Cycles)(nextLoad < cfg.numTiles ? tmaBusyUntil : LLONG_MAX),
                (Cycles)(nextCompute < cfg.numTiles && nextCompute < nextLoad
                    ? std::max(tcBusyUntil, buffers[nextCompute].loadedAt) : LLONG_MAX),
                (Cycles)(nextStore < nextCompute
                    ? std::max(storeBusyUntil, computeComplete[nextStore]) : LLONG_MAX)
            });

            if (nextEvent < LLONG_MAX) {
                tmaBusyUntil = std::max(tmaBusyUntil, nextEvent);
                tcBusyUntil = std::max(tcBusyUntil, nextEvent);
                storeBusyUntil = std::max(storeBusyUntil, nextEvent);
            }
        }

        return storeComplete.back();
    }

private:
    int pipelineDepth_;
};

// 打印三种执行模式的对比结果
void printComparison(const PipelineConfig& cfg) {
    SynchronousExecutor sync;
    AsynchronousExecutor async(cfg.pipelineDepth);
    FullPipeline fullPipe(cfg.pipelineDepth);

    Cycles syncTime = sync.execute(cfg);
    Cycles asyncTime = async.execute(cfg);
    Cycles pipeTime = fullPipe.execute(cfg);

    double asyncSpeedup = (double)syncTime / asyncTime;
    double pipeSpeedup = (double)syncTime / pipeTime;

    std::cout << std::left
              << std::setw(18) << "流水线深度" << ": " << cfg.pipelineDepth << "\n"
              << std::setw(18) << "Tile 数量" << ": " << cfg.numTiles << "\n"
              << std::setw(18) << "TMA 延迟" << ": " << cfg.tmaLoadCycles << " 周期\n"
              << std::setw(18) << "TC 计算" << ": " << cfg.tensorCoreCycles << " 周期\n"
              << std::setw(18) << "存储延迟" << ": " << cfg.storeCycles << " 周期\n\n";

    std::cout << std::left
              << std::setw(22) << "执行模式"
              << std::setw(18) << "总周期数"
              << "加速比\n";
    std::cout << std::string(55, '-') << "\n";

    std::cout << std::left
              << std::setw(22) << "同步（Synchronous）"
              << std::setw(18) << syncTime
              << "1.00x（基线）\n";

    std::cout << std::left
              << std::setw(22) << "异步（简化版）"
              << std::setw(18) << asyncTime
              << std::fixed << std::setprecision(2) << asyncSpeedup << "x\n";

    std::cout << std::left
              << std::setw(22) << "完整流水线"
              << std::setw(18) << pipeTime
              << std::fixed << std::setprecision(2) << pipeSpeedup << "x\n\n";
}

int main() {
    std::cout << "=== Lecture 11：异步流水线执行模拟 ===\n";
    std::cout << "Stanford CS149 - 面向 AI 的专用硬件编程\n";
    std::cout << "模型对比：同步执行 vs TMA+TensorCore 流水线\n\n";

    // 场景1：计算密集型 tile（类似 GEMM 矩阵乘法）
    {
        std::cout << "--- 场景1：计算密集型 (类似 GEMM) ---\n";
        PipelineConfig cfg;
        cfg.tmaLoadCycles = 100;      // TMA 加载相对于计算来说很快
        cfg.tensorCoreCycles = 1000;  // Tensor core 计算占据主导地位
        cfg.storeCycles = 50;
        cfg.numTiles = 32;
        cfg.pipelineDepth = 4;        // 4 阶段流水线（类似 ThunderKittens）
        printComparison(cfg);

        std::cout << "分析：GPU 计算受限（compute-bound）。异步有一定帮助，但重叠有限。\n";
        std::cout << "  TMA 加载仅占计算的 10% → 大部分时间用于计算。\n\n";
    }

    // 场景2：内存密集（访存受限）型 tile（类似注意力机制，带宽受限）
    {
        std::cout << "--- 场景2：内存密集型 (类似 Attention) ---\n";
        PipelineConfig cfg;
        cfg.tmaLoadCycles = 800;      // 从 HBM 加载数据占据主导地位
        cfg.tensorCoreCycles = 200;   // 计算很快
        cfg.storeCycles = 100;
        cfg.numTiles = 64;
        cfg.pipelineDepth = 8;        // 更深的流水线以隐藏加载延迟
        printComparison(cfg);

        std::cout << "分析：HBM 带宽受限。异步对隐藏加载延迟至关重要。\n";
        std::cout << "  TMA 加载时间是计算的 4 倍 → 流水线重叠不可或缺。\n";
        std::cout << "  ThunderKittens：默认 4 阶段输入流水线，8 个消费者 warp。\n\n";
    }

    // 场景3：平衡型（典型的 Transformer 层）
    {
        std::cout << "--- 场景3：平衡型流水线 ---\n";
        PipelineConfig cfg;
        cfg.tmaLoadCycles = 500;
        cfg.tensorCoreCycles = 500;
        cfg.storeCycles = 200;
        cfg.numTiles = 16;
        cfg.pipelineDepth = 4;
        printComparison(cfg);

        std::cout << "分析：当加载时间 ≈ 计算时间时，可实现近乎完美的重叠。\n\n";
    }

    // TPU 与 GPU 对比
    {
        std::cout << "══════════════════════════════════════════════════════════════\n";
        std::cout << "TPU 与 GPU 对比：\n";
        std::cout << "══════════════════════════════════════════════════════════════\n\n";

        std::cout << "Google TPU v1（脉动阵列/Systolic Array）：\n";
        std::cout << "  - 不需要异步指令（数据流执行，dataflow execution）\n";
        std::cout << "  - 权重预先加载：weight-stationary 数据流模式\n";
        std::cout << "  - 输入数据流经阵列：兼具空间局部性和时间局部性\n";
        std::cout << "  - 关键指令：read_weights, matrix_multiply, activate\n";
        std::cout << "  - 约 30% 的芯片面积用于算术运算（而 CPU 仅约 5%）\n\n";

        std::cout << "NVIDIA H100/B100：\n";
        std::cout << "  - TMA：异步 tensor 加载，硬件自动生成地址\n";
        std::cout << "  - Tensor core：warp-group MMA，16x16 tile 粒度\n";
        std::cout << "  - 需要谨慎的流水线管理（ThunderKittens DSL）\n";
        std::cout << "  - B100：单线程 MMA，无需 warp 分组，使用 tcgen05 指令\n\n";

        std::cout << "SambaNova SN40L（数据流架构/Dataflow）：\n";
        std::cout << "  - 无指令 → 无需指令取指/译码开销\n";
        std::cout << "  - Metapipelining：层次化粗粒度流水线\n";
        std::cout << "  - Token 控制的数据流：无需基于锁的同步\n";
        std::cout << "  - 520 MB 片上 SRAM，而 H100 仅 100 MB\n\n";
    }

    return 0;
}
