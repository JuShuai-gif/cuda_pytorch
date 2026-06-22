// lecture10_part2.cpp
// Roofline 模型（屋顶线模型）与算术强度（Arithmetic Intensity）分析
// 建模计算能力 vs 内存带宽的性能天花板（performance ceiling）
// Stanford CS149, 2025年秋季 - 第10讲：硬件专用化（Hardware Specialization）
//
// 核心概念说明：
//
// 1. Roofline 模型（屋顶线模型）：
//    - 用于分析一个计算核（kernel）在特定硬件平台上的性能上限
//    - 横轴：算术强度 AI（FLOPs / Byte）
//    - 纵轴：可达到的算力（TFLOPS）
//    - 由两条上限线组成：
//      • 内存带宽天花板（倾斜线）：可达到算力 = AI × 内存带宽
//      • 计算天花板（水平线）：可达到算力 = 峰值算力
//    - 两条线的交点称为"脊点"（Ridge Point）
//
// 2. 算术强度（Arithmetic Intensity, AI）：
//    - 定义：AI = 总浮点运算次数 / 总访存字节数（单位：FLOPs/Byte）
//    - AI 越高，单位访存数据能被复用的次数越多
//    - 例：GEMM (4096×4096) 的 AI ≈ 1000+ FLOPs/Byte → 计算受限
//    - 例：向量加法 的 AI ≈ 0.08 FLOPs/Byte → 带宽受限
//
// 3. 脊点（Ridge Point）：
//    - 计算公式：脊点 = 峰值 TFLOPS × 10^12 / (内存带宽 GB/s × 10^9)
//    - AI < 脊点 → 内存带宽受限（Memory-bound）
//    - AI >= 脊点 → 计算受限（Compute-bound）
//    - 更高算力的硬件反而需要更高的 AI 才能达到计算受限
//
// 4. 数据移动的能量成本：
//    - 课程核心观点：数据移动比计算本身消耗更多能量
//    - DRAM 访问：~1200 pJ / 64位，而 FP32 运算仅 ~20 pJ
//    - "重新计算比存储后重新加载更省电"（recomputing > storing + reloading）
//
// 5. 指令开销摊销（Instruction Overhead Amortization）：
//    - 通用 CPU：每条指令通常只完成 1-2 次运算 → 控制开销比例高
//    - 脉动阵列/矩阵乘法单元：一条复杂指令可触发 256 次运算
//      → 指令开销从 2000% 降至 27%
//
// 编译命令：g++ -std=c++17 -O2 lecture10_part2.cpp -o lecture10_part2
// 运行命令：./lecture10_part2

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <algorithm>

// ============================================================================
// 硬件规格（HardwareSpec）：表示一个硬件平台的性能极限参数
// ============================================================================

struct HardwareSpec {
    std::string name;              // 硬件平台名称
    double peakTFLOPS;             // 峰值计算能力（TFLOPS，万亿次浮点运算/秒）
    double memoryBandwidthGBs;     // 峰值内存带宽（GB/s，千兆字节/秒）
    double onChipSRAM_MB;          // 片上 SRAM 容量（MB，兆字节）
    double perfPerWatt;            // 相对能效比（越高越好，以CPU为基准=1.0）
};

// ============================================================================
// 计算核（Kernel）：表示一个计算任务的特性
// ============================================================================

struct Kernel {
    std::string name;              // 内核名称
    double flops;                  // 总浮点运算次数
    double bytesAccessed;          // 从片外内存访问的总字节数
    // 算术强度 = 总浮点运算次数 / 总访存字节数（单位：FLOPs/Byte）
    double arithmeticIntensity() const {
        return flops / bytesAccessed;
    }
};

// ============================================================================
// Roofline 分析结果：判断一个核是计算受限还是内存受限
//
// 判断逻辑：
//   - 内存带宽受限时的可达到 TFLOPS = AI × 内存带宽 × (单位换算)
//   - 计算受限时的可达到 TFLOPS = 峰值 TFLOPS
//   - 实际可达到 TFLOPS = min(带宽限制值, 计算限制值)
//   - 利用率（Utilization）= 实际可达到 TFLOPS / 峰值 TFLOPS × 100%
// ============================================================================

struct RooflineResult {
    double arithmeticIntensity;    // 算术强度
    double achievableTFLOPS;       // 可达到的 TFLOPS（受限于瓶颈）
    double ridgePoint;             // 脊点 AI 值（AI >= 此值则为计算受限）
    bool isComputeBound;           // 是否为计算受限
    double utilizationPercent;     // 峰值算力利用率（%）
};

class RooflineAnalyzer {
public:
    RooflineAnalyzer(const HardwareSpec& hw) : hw_(hw) {
        // 计算脊点（Ridge Point）：
        // 在脊点上，内存带宽限制 == 计算能力限制
        // ridgePoint = 峰值TFLOPS × 10^12 / (内存带宽GB/s × 10^9) = FLOPs/Byte
        ridgePoint_ = (hw_.peakTFLOPS * 1e12) / (hw_.memoryBandwidthGBs * 1e9);
    }

    RooflineResult analyze(const Kernel& k) const {
        RooflineResult r;
        r.arithmeticIntensity = k.arithmeticIntensity();
        r.ridgePoint = ridgePoint_;

        // 内存带宽受限时的上限：可达到 TFLOPS = AI × 内存带宽
        double memoryBoundTFLOPS = r.arithmeticIntensity * hw_.memoryBandwidthGBs * 1e9 / 1e12;

        // 计算受限时的上限：峰值 TFLOPS
        double computeBoundTFLOPS = hw_.peakTFLOPS;

        // 取瓶颈（最小值）
        r.achievableTFLOPS = std::min(memoryBoundTFLOPS, computeBoundTFLOPS);
        r.isComputeBound = (r.arithmeticIntensity >= ridgePoint_);
        r.utilizationPercent = (r.achievableTFLOPS / hw_.peakTFLOPS) * 100.0;

        return r;
    }

    // 打印完整的 Roofline 分析表格
    void printRoofline(const std::vector<Kernel>& kernels) const {
        std::cout << "\n══════════════════════════════════════════════════════════════\n";
        std::cout << "Roofline 模型：" << hw_.name << "\n";
        std::cout << "  峰值算力：        " << hw_.peakTFLOPS << " TFLOPS\n";
        std::cout << "  内存带宽：        " << hw_.memoryBandwidthGBs << " GB/s\n";
        std::cout << "  脊点（Ridge Pt）：" << ridgePoint_ << " FLOPs/Byte\n";
        std::cout << "  片上 SRAM：       " << hw_.onChipSRAM_MB << " MB\n";
        std::cout << "══════════════════════════════════════════════════════════════\n\n";

        std::cout << std::left
                  << std::setw(25) << "计算核（Kernel）"
                  << std::setw(18) << "算术强度"
                  << std::setw(18) << "可达到TFLOPS"
                  << std::setw(15) << "利用率"
                  << std::setw(18) << "受限类型"
                  << "\n";
        std::cout << std::string(94, '-') << "\n";

        for (const auto& k : kernels) {
            auto r = analyze(k);
            std::cout << std::left
                      << std::setw(25) << k.name
                      << std::setw(18) << std::fixed << std::setprecision(2)
                      << r.arithmeticIntensity
                      << std::setw(18) << std::setprecision(4) << r.achievableTFLOPS
                      << std::setw(15) << std::setprecision(1) << r.utilizationPercent << "%"
                      << std::setw(18) << (r.isComputeBound ? "计算受限" : "内存受限")
                      << "\n";
        }
        std::cout << std::endl;
    }

private:
    HardwareSpec hw_;      // 硬件规格
    double ridgePoint_;    // 脊点（AI阈值）
};

// ============================================================================
// 专用化硬件的能效权衡分析
//
// 关键趋势（从CPU到ASIC）：
//   - 能效比从 1x 提升到 1000x
//   - 设计成本从 $0 上升到数亿美元
//   - 可编程性从"最容易"降到"不可编程"
//
// 这就是专用化的本质权衡：更高的效率 = 更低的灵活性
// ============================================================================

void analyzeEnergyEfficiency() {
    std::cout << "══════════════════════════════════════════════════════════════\n";
    std::cout << "专用化硬件的能效分析\n";
    std::cout << "（相对于CPU上高质量C代码的相对能效比，假设均为计算受限场景）\n";
    std::cout << "══════════════════════════════════════════════════════════════\n\n";

    struct Platform {
        std::string name;                  // 平台名称
        double perfPerWattMultiplier;      // 相对于CPU的每瓦性能倍数
        double designCost_M;               // 设计成本（百万美元）
        std::string programmability;       // 可编程性描述
    };

    std::vector<Platform> platforms = {
        {"能效优化CPU",              1.0,    0.0,    "最容易"},
        {"高吞吐GPU",                10.0,   0.0,    "中等（CUDA）"},
        {"可编程DSP",                20.0,   1.0,    "有限领域"},
        {"领域专用加速器（DSA）",      50.0,   5.0,    "DSL专用语言（如DNN）"},
        {"FPGA/可重构硬件",         100.0,  10.0,   "困难（Verilog）"},
        {"固定功能ASIC",            1000.0, 100.0,  "不可编程"},
    };

    std::cout << std::left
              << std::setw(28) << "平台"
              << std::setw(20) << "每瓦性能(vs CPU)"
              << std::setw(18) << "设计成本"
              << "可编程性\n";
    std::cout << std::string(85, '-') << "\n";

    for (const auto& p : platforms) {
        std::cout << std::left
                  << std::setw(28) << p.name
                  << std::setw(20) << (std::to_string((int)p.perfPerWattMultiplier) + "倍")
                  << std::setw(18) << ("$" + std::to_string((int)p.designCost_M) + "M")
                  << p.programmability << "\n";
    }
    std::cout << std::endl;
}

// ============================================================================
// 数据移动的能量成本分析（来自课程讲义的关键数据）
//
// 核心发现：
//   - 从 DRAM（LPDDR）读取 64 位数据的能耗约 1200 pJ
//   - 执行一次 FP32 运算仅需约 20 pJ
//   - DRAM 访问能耗是计算能耗的约 60 倍
//   - 结论：减少数据移动是提升能效的关键
//   - "重新计算 > 存储再加载" 是 DNN 硬件设计的重要原则
// ============================================================================

void analyzeDataMovementEnergy() {
    std::cout << "══════════════════════════════════════════════════════════════\n";
    std::cout << "数据移动能量成本分析（每次操作的近似能耗）\n";
    std::cout << "══════════════════════════════════════════════════════════════\n\n";

    struct EnergyCost {
        std::string operation;     // 操作类型
        double energy_pJ;          // 能耗（皮焦耳，pJ）
    };

    std::vector<EnergyCost> costs = {
        {"整数运算",                     1.0},
        {"FP32 浮点运算",               20.0},
        {"读取64位本地SRAM",            26.0},
        {"读取64位LPDDR（DRAM）",       1200.0},
    };

    std::cout << std::left
              << std::setw(35) << "操作"
              << "能耗 (pJ)\n";
    std::cout << std::string(50, '-') << "\n";

    for (const auto& c : costs) {
        std::cout << std::left << std::setw(35) << c.operation << c.energy_pJ << " pJ\n";
    }
    std::cout << "\n核心启示：重新计算比存储后再重新加载更省电！\n";
    std::cout << "SRAM 访问：26 pJ，LPDDR 访问：1200 pJ（相差约 46 倍）\n\n";
}

// ============================================================================
// 指令开销摊销分析
//
// 核心概念：
//   - 通用处理器每个指令通常只完成 1-2 次运算，控制开销占比很高
//   - 专用硬件的一条"复杂指令"可以触发大量并行运算，摊销控制开销
//   - 示例：
//     • 半精度 FMA（1次乘加）：控制开销 ~2000%
//     • 半精度 DP4（4次点积）：控制开销 ~500%
//     • 半精度 4×4 MMA（矩阵乘加，256次运算）：控制开销仅 ~27%
//
//   这就是脉动阵列/Tensor Core的根本优势：
//   一条指令触发大量运算，大幅降低控制开销比例
// ============================================================================

void analyzeInstructionOverhead() {
    std::cout << "══════════════════════════════════════════════════════════════\n";
    std::cout << "指令开销摊销分析\n";
    std::cout << "（可编程性相对于实际有用计算的开销比例）\n";
    std::cout << "══════════════════════════════════════════════════════════════\n\n";

    struct InstrType {
        std::string name;              // 指令类型
        double overheadPercent;        // 控制开销百分比
    };

    std::vector<InstrType> instrs = {
        {"半精度FMA（1次乘加）",            2000.0},
        {"半精度DP4（4次乘加）",            500.0},
        {"半精度 4×4 MMA（256次乘加）",     27.0},
    };

    std::cout << std::left
              << std::setw(40) << "指令类型"
              << "控制开销\n";
    std::cout << std::string(60, '-') << "\n";

    for (const auto& i : instrs) {
        std::cout << std::left << std::setw(40) << i.name
                  << i.overheadPercent << "%\n";
    }
    std::cout << "\n核心原则：将指令流的开销摊销到一条复杂指令的大量运算中\n";
    std::cout << "运算次数越多，每条运算分摊的指令开销越小\n\n";
}

// ============================================================================
// 主函数 —— 多个硬件平台的 Roofline 分析 + 能效/能量/指令开销综合分析
// ============================================================================

int main() {
    std::cout << "=== 第10讲：Roofline 模型与算术强度分析 ===\n";
    std::cout << "Stanford CS149 - 硬件专用化\n\n";

    // ---------- 定义多种硬件平台 ----------

    // CPU：通用处理器，算力低但通用性强
    HardwareSpec cpu = {
        "CPU（2 GHz，8核，AVX-512）",
        0.5,    // 0.5 TFLOPS 峰值算力
        50.0,   // 50 GB/s 内存带宽
        30.0,   // 30 MB L3 缓存
        1.0     // 能效基线（以CPU为标准=1.0）
    };

    // GPU：高吞吐SIMD架构，适合数据并行计算
    HardwareSpec gpu = {
        "NVIDIA H100 GPU",
        67.0,   // 67 TFLOPS FP32（SIMD）；Tensor Core 可达 989 TFLOPS
        3350.0, // 3.35 TB/s HBM3 高带宽内存
        50.0,   // 50 MB L2 缓存
        10.0    // 约 10 倍于 CPU 的能效
    };

    // GPU（Tensor Core模式）：矩阵乘法专用硬件单元
    HardwareSpec gpuTensor = {
        "NVIDIA H100（Tensor Cores，fp16）",
        989.0,  // 989 TFLOPS（Tensor Core 半精度模式）
        3350.0, // 3.35 TB/s HBM3
        50.0,   // 50 MB L2 缓存
        10.0
    };

    // TPU v1：Google 的第一代张量处理单元，基于脉动阵列
    HardwareSpec tpu = {
        "Google TPU v1（脉动阵列）",
        92.0,   // 92 TFLOPS（int8 精度）
        30.0,   // 30 GB/s（到主机内存的总线带宽，而非片上带宽）
        28.0,   // 28 MB 片上 SRAM
        80.0    // 约 80 倍的每瓦性能（相对于CPU+GPU组合）
    };

    // ---------- 定义典型计算核 ----------

    // 矩阵乘法（GEMM 4096³）：极高算术强度
    // C = A(M×K) × B(K×N)：FLOPs = 2 × M × N × K（每次乘加 = 2 FLOPs）
    int M = 4096, K = 4096, N = 4096;
    double flops_gemm = 2.0 * M * N * K;
    double bytes_gemm = (double)(M * K + K * N + M * N) * 4.0;  // fp32 = 4 字节/元素

    // 向量加法：极低算术强度
    int vecLen = 1 << 20;  // 1M 个元素
    double flops_vec = (double)vecLen;
    double bytes_vec = (double)vecLen * 3.0 * 4.0;  // 2个输入 + 1个输出，各 fp32

    // 卷积：中等算术强度
    int H = 224, W = 224, C_in = 3, C_out = 64, Kh = 3, Kw = 3;
    double flops_conv = 2.0 * H * W * C_in * C_out * Kh * Kw;
    double bytes_conv = (H * W * C_in + C_out * C_in * Kh * Kw + H * W * C_out) * 4.0;

    std::vector<Kernel> kernels = {
        {"GEMM 4096×4096×4096", flops_gemm, bytes_gemm},      // 极高 AI → 计算受限
        {"向量加法（1M元素）",  flops_vec, bytes_vec},         // 极低 AI → 内存受限
        {"3×3卷积（224×224）",  flops_conv, bytes_conv},       // 中等 AI
        {"DNN层（典型值）",     1e10, 2e8},                     // ~50 FLOPs/Byte
        {"注意力机制（seq=2048）", 8e9, 4e9},                   // ~2 FLOPs/Byte（内存受限）
    };

    // 在每种硬件平台上进行分析
    RooflineAnalyzer cpuAnalyzer(cpu);
    cpuAnalyzer.printRoofline(kernels);

    RooflineAnalyzer gpuAnalyzer(gpu);
    gpuAnalyzer.printRoofline(kernels);

    RooflineAnalyzer tensorAnalyzer(gpuTensor);
    tensorAnalyzer.printRoofline(kernels);

    // 能量效率和开销分析（不依赖于具体的核）
    analyzeEnergyEfficiency();
    analyzeDataMovementEnergy();
    analyzeInstructionOverhead();

    // 总结
    std::cout << "══════════════════════════════════════════════════════════════\n";
    std::cout << "关键要点总结：\n";
    std::cout << "1. GEMM 即使在 H100 上也是计算受限（AI = " << flops_gemm/bytes_gemm << " FLOPs/Byte）\n";
    std::cout << "2. 向量加法永远内存受限（AI = " << flops_vec/bytes_vec << " FLOPs/Byte）\n";
    std::cout << "3. Tensor Core 提升了脊点 AI → 需要更高的 AI 才能达到计算受限\n";
    std::cout << "4. 脉动阵列消除指令开销 → 控制开销从 2000% 降至 27%\n";
    std::cout << "5. 数据移动主导能耗 → DRAM 访存相比 SRAM 多约 46 倍能量\n";
    std::cout << "6. 硬件专用化的核心权衡：效率 vs 灵活性 → Roofline模型指导架构选择\n";
    std::cout << "══════════════════════════════════════════════════════════════\n";

    return 0;
}
