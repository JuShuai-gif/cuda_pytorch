// lecture3_part1.cpp - 延迟 vs 带宽 + 流水线模拟
// =============================================================================
// CS149 第3讲核心概念：
//   - 延迟（Latency）：完成单个操作所需的时间
//     （例如：从旧金山开车到斯坦福需要0.5小时）
//   - 带宽（Bandwidth）：完成操作的速率
//     （例如：4车道高速公路上每小时可通行4辆车）
//   - 类比分析：开得更快（降低延迟）vs. 修建更多车道（增加带宽）
//     这两种优化策略完全不同。降低延迟让每一辆车更快到达，而增加带宽
//     让单位时间内可以通过更多车辆。
//   - 内存带宽：内存提供数据的速率（例如：NVIDIA V100 为 900 GB/s）
//     这是现代计算系统中极为关键的资源瓶颈。
//   - 带宽受限计算（Bandwidth-Limited Computation）：
//     处理器请求数据的速度超过了内存能够供应数据的速度，
//     导致ALU大量空闲等待数据到达。
//   - 指令流水线（Instruction Pipeline）：IF → D → EX → WB
//     （取指→译码→执行→写回），单条指令延迟4个周期，但吞吐量可达1条/周期
//     这是因为流水线允许不同指令的不同阶段同时执行。
//   - 洗衣流水线类比：洗涤、烘干、折叠三个阶段可以重叠执行
//     瓶颈阶段（烘干，60分钟）决定了整体吞吐量。
//   - 管道类比：最大流量由管道中最窄的部分决定（瓶颈瓶颈原理）
//     任何流水线系统的吞吐量都受限于最慢的流水级。
//   - 流水线填充时间（Pipeline Fill Time）：
//     从第一条指令进入流水线到稳态吞吐量之间的初始延迟。
//
// 编译命令：g++ -std=c++17 -O2 lecture3_part1.cpp -o lecture3_part1
// =============================================================================

#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <queue>
#include <thread>
#include <chrono>

// ---------------------------------------------------------------------------
// 高速公路（汽车）类比：延迟 vs 带宽
//
// 这个类比帮助理解延迟和带宽的本质区别：
// - 延迟 = 单辆车从旧金山到斯坦福所需的时间
// - 带宽 = 高速公路上每小时可以通过的车辆总数
// - 增加车道数（并行度）可以提高带宽，但不会减少单辆车的延迟
// - 这个类比直接映射到计算机体系结构中的延迟 vs 带宽概念
// ---------------------------------------------------------------------------
void demo_highway_analogy() {
    std::cout << "[1] 高速公路类比：延迟 vs 带宽\n" << std::endl;
    std::cout << "    路程：旧金山 → 斯坦福（约50公里）\n\n";

    struct Scenario {
        std::string name;
        double speed_kmph;   // 车辆速度（公里/小时）
        int lanes;            // 车道数量
        double spacing_km;    // 车辆之间的间距（公里）
    };

    std::vector<Scenario> scenarios = {
        {"基准方案", 100.0, 1, 50.0},
        {"加快速度", 200.0, 1, 50.0},
        {"增加车道", 100.0, 4, 50.0},
        {"缩小间距", 100.0, 1, 1.0},
        {"缩小间距+增加车道", 100.0, 4, 1.0},
    };

    std::cout << "    " << std::left << std::setw(22) << "方案"
              << std::setw(12) << "延迟(h)"
              << std::setw(16) << "吞吐量(辆/h)"
              << "备注\n";
    std::cout << "    " << std::string(70, '-') << std::endl;

    const double distance = 50.0; // 距离：公里

    for (const auto& s : scenarios) {
        // 延迟 = 距离 / 速度（单辆车到达目的地的时间）
        double latency = distance / s.speed_kmph;
        // 吞吐量 = 车道数 × 速度 / 车间距（每小时的车辆数）
        double throughput = s.lanes * s.speed_kmph / s.spacing_km;

        std::cout << "    " << std::left << std::setw(22) << s.name
                  << std::setw(12) << std::fixed << std::setprecision(2) << latency
                  << std::setw(16) << std::setprecision(1) << throughput;
        
        if (s.name == "基准方案") std::cout << "（高速公路上一次只有1辆车）";
        else if (s.name == "加快速度") std::cout << "（速度翻倍，车道数不变）";
        else if (s.name == "增加车道") std::cout << "（车道数翻4倍，速度不变）";
        else if (s.name == "缩小间距") std::cout << "（间距1公里 → 每36秒一辆车）";
        std::cout << std::endl;
    }

    std::cout << "\n    核心洞察：提高吞吐量 ≠ 降低延迟。\n"
              << "    修建更多车道可以提高吞吐量，\n"
              << "    但不会减少任何一辆车的行驶时间。\n" << std::endl;
}

// ---------------------------------------------------------------------------
// 洗衣流水线类比（Laundry Pipelining）
//
// 这个经典类比展示了流水线如何在不改变单件任务延迟的情况下提高吞吐量。
// 阶段1：洗涤（45分钟），阶段2：烘干（60分钟），阶段3：折叠（15分钟）
//
// 关键概念：
// - 顺序执行：3批衣服需要 (45+60+15)×3 = 360分钟
// - 流水线执行：第一批120分钟 + 后续每60分钟一批 = 共240分钟
// - 瓶颈阶段（烘干60分钟）决定了流水线的最大吞吐量
// - 单批延迟 = 各阶段时间之和（120分钟），与流水线无关
// ---------------------------------------------------------------------------
void demo_laundry_pipeline() {
    std::cout << "[2] 洗衣流水线类比\n" << std::endl;

    struct Stage { std::string name; int minutes; };
    std::vector<Stage> stages = {
        {"洗涤", 45},
        {"烘干",  60},
        {"折叠", 15}
    };

    // 顺序执行：3批衣服，无流水线
    // 每批依次经过洗涤→烘干→折叠，总时间 = 3 × (45+60+15) = 360分钟
    int seq_total = 0;
    for (int load = 0; load < 3; load++) {
        for (const auto& s : stages) seq_total += s.minutes;
    }

    // 流水线执行：不同批次在不同阶段同时处理
    // 瓶颈是最慢的阶段（烘干 = 60分钟）
    int bottleneck = 60; // 瓶颈阶段耗时（分钟）
    int pipeline_latency = 45 + 60 + 15; // 第一批的延迟：120分钟
    // 流水线总时间 = 第一批延迟 + (批次-1) × 瓶颈时间
    int pipeline_total = pipeline_latency + (3 - 1) * bottleneck;

    std::cout << "    任务：洗3批衣服\n";
    std::cout << "    阶段：洗涤(45分钟) → 烘干(60分钟) → 折叠(15分钟)\n\n";

    std::cout << "    顺序执行：" << seq_total << " 分钟\n";
    std::cout << "    流水线执行：" << pipeline_total << " 分钟（第一批："
              << pipeline_latency << " 分钟）\n";
    std::cout << "    加速比：" << std::fixed << std::setprecision(1) 
              << static_cast<double>(seq_total) / pipeline_total << "x\n" << std::endl;

    std::cout << "    流水线时间线：\n";
    std::cout << "    " << std::setw(8) << "时间" 
              << std::setw(14) << "洗衣机" 
              << std::setw(14) << "烘干机" 
              << std::setw(14) << "折叠机" << std::endl;
    std::cout << "    " << std::string(50, '-') << std::endl;

    // 为3批衣服绘制流水线时间线
    // 模拟每个阶段依次可用的流水线调度
    struct Job { int load_id; int stage; int start; int end; };
    std::vector<Job> timeline;
    int washer_available = 0, dryer_available = 0, folder_available = 0;

    for (int load = 0; load < 3; load++) {
        // 洗涤阶段：只要洗衣机空闲就开始
        int wash_start = washer_available;
        int wash_end = wash_start + 45;
        washer_available = wash_end;
        timeline.push_back({load, 0, wash_start, wash_end});

        // 烘干阶段：需要等待洗涤完成且烘干机空闲
        int dry_start = std::max(wash_end, dryer_available);
        int dry_end = dry_start + 60;
        dryer_available = dry_end;
        timeline.push_back({load, 1, dry_start, dry_end});

        // 折叠阶段：需要等待烘干完成且折叠空闲
        int fold_start = std::max(dry_end, folder_available);
        int fold_end = fold_start + 15;
        folder_available = fold_end;
        timeline.push_back({load, 2, fold_start, fold_end});
    }

    int max_time = 0;
    for (const auto& j : timeline) max_time = std::max(max_time, j.end);
    // 向上取整到最近的15分钟间隔
    max_time = ((max_time + 14) / 15) * 15;

    const char* names[] = {"洗涤", "烘干", "折叠"};
    for (int t = 0; t <= max_time; t += 15) {
        std::cout << "    t=" << std::setw(4) << t;
        for (int s = 0; s < 3; s++) {
            bool busy = false;
            int load_id = -1;
            for (const auto& j : timeline) {
                if (j.stage == s && j.start < t + 15 && j.end > t) {
                    busy = true;
                    load_id = j.load_id;
                    break;
                }
            }
            if (busy) {
                std::cout << std::setw(14) << ("  第" + std::to_string(load_id + 1) + "批");
            } else {
                std::cout << std::setw(14) << "  空闲";
            }
        }
        std::cout << std::endl;
    }

    std::cout << "\n    关键：瓶颈阶段（烘干，60分钟）决定了吞吐量 = 1批/小时\n";
    std::cout << "    单批延迟 = 120分钟，但吞吐量 = 每60分钟完成1批。\n" << std::endl;
}

// ---------------------------------------------------------------------------
// 指令流水线模拟（IF → D → EX → WB）
//
// 模拟经典的4级RISC流水线：
// - IF（Instruction Fetch）：从内存中取出指令
// - D（Decode）：译码指令操作码和操作数
// - EX（Execute）：执行算术/逻辑运算
// - WB（Write Back）：将结果写回寄存器
//
// 关键特性：
// - 每条指令需要经过全部4个阶段（延迟 = 4个周期）
// - 但每个周期可以有4条不同指令在不同阶段执行
// - 稳态吞吐量 = 1条指令/周期（经过流水线填充后）
// - 现代CPU可能有20+级流水线，甚至更深
// ---------------------------------------------------------------------------
void demo_instruction_pipeline() {
    std::cout << "[3] 指令流水线（4级）\n" << std::endl;

    const int STAGES = 4;
    const char* stage_names[] = {"IF", "D ", "EX", "WB"};
    const int NUM_INSTRS = 6;

    std::cout << "    流水线阶段：IF(取指) → D(译码) → EX(执行) → WB(写回)\n";
    std::cout << "    每个阶段耗时：1个周期。单条指令总延迟：4个周期。\n";
    std::cout << "    吞吐量：1条指令/周期（流水线填充后）。\n\n";

    // 绘制流水线时间线示意图
    std::cout << "    ";
    for (int i = 0; i < NUM_INSTRS; i++) {
        std::cout << " 指令" << std::setw(2) << i << " ";
    }
    std::cout << "\n    ";
    for (int i = 0; i < NUM_INSTRS; i++) {
        for (int s = 0; s < STAGES; s++) std::cout << "---";
    }
    std::cout << std::endl;

    // 每条指令在每个阶段占用一个时隙
    // 调度表：行=周期，列=流水阶段
    std::vector<std::vector<int>> schedule(NUM_INSTRS + STAGES - 1, 
                                            std::vector<int>(STAGES, -1));

    // 指令 i 在周期 (i+stage) 进入阶段 stage
    for (int instr = 0; instr < NUM_INSTRS; instr++) {
        for (int stage = 0; stage < STAGES; stage++) {
            schedule[instr + stage][stage] = instr;
        }
    }

    // 按周期打印（每一行表示一个周期中各指令所在的阶段）
    for (int cycle = 0; cycle < NUM_INSTRS + STAGES - 1; cycle++) {
        std::cout << "    ";
        for (int instr = 0; instr < NUM_INSTRS; instr++) {
            bool found = false;
            for (int stage = 0; stage < STAGES; stage++) {
                if (cycle < static_cast<int>(schedule.size()) && 
                    schedule[cycle][stage] == instr) {
                    std::cout << " " << stage_names[stage] << " ";
                    found = true;
                    break;
                }
            }
            if (!found) std::cout << " .. ";
        }
        std::cout << std::endl;
    }

    std::cout << "\n    " << NUM_INSTRS << " 条指令的总周期数：" 
              << (NUM_INSTRS + STAGES - 1) << "\n";
    std::cout << "    顺序执行（1条指令/周期完成后再开始下一条）：" << (NUM_INSTRS * STAGES) << " 个周期\n";
    std::cout << "    流水线加速比：" << std::fixed << std::setprecision(2)
              << static_cast<double>(NUM_INSTRS * STAGES) / (NUM_INSTRS + STAGES - 1) << "x\n" 
              << std::endl;

    // 大规模N的情况：吞吐量渐近趋于 1条指令/周期
    std::cout << "    当 N 很大时：吞吐量 → 1条指令/周期\n";
    std::cout << "    （相比非流水线提升4倍）\n";
    std::cout << "    现代CPU：某些指令的流水线可达约20级\n" << std::endl;
}

// ---------------------------------------------------------------------------
// 带宽受限计算模拟
//
// 从课程讲义中的示例：向量逐元素乘法
// 每次MUL操作需要3次内存访问（12字节）→ 带宽受限
//
// 分析逻辑：
// NVIDIA V100 拥有 5120 个 fp32 MUL/时钟周期，
// 以1.6GHz运行，理论上需要约98 TB/s的带宽才能充分利用。
// 但实际 HBM2 带宽只有 900 GB/s，因此利用率不到1%。
//
// 结论：现代计算中，内存带宽往往是最关键的瓶颈资源。
// 必须通过数据复用（时间局部性）、线程间数据共享和
// 更高的算术强度来弥补带宽的不足。
// ---------------------------------------------------------------------------
void demo_bandwidth_bound() {
    std::cout << "[4] 带宽受限计算\n" << std::endl;

    // NVIDIA V100 规格参数
    double v100_sms = 80;              // SM（流多处理器）数量
    double v100_alu_per_sm = 64;       // 每个SM的fp32 ALU数量
    double v100_clock = 1.6e9;         // 频率（Hz）
    double v100_bandwidth = 900e9;     // HBM2带宽（字节/秒）
    double bytes_per_mul = 12;         // 每次MUL的内存操作：3次 × 4字节 = 12字节
    double mul_per_clock = v100_sms * v100_alu_per_sm; // 每时钟周期的MUL操作数

    // 完全利用所有ALU所需的带宽
    double required_bw = mul_per_clock * v100_clock * bytes_per_mul;
    // 实际带宽占所需带宽的百分比（即带宽利用率/效率）
    double efficiency = v100_bandwidth / required_bw * 100.0;

    std::cout << "    任务：逐元素向量乘法（A[i] × B[i]）\n";
    std::cout << "    每次MUL的内存操作：3次（加载A、加载B、存储C）= 12字节\n\n";

    std::cout << "    NVIDIA V100：\n";
    std::cout << "    - " << v100_sms << " 个SM × " << v100_alu_per_sm 
              << " 个fp32 ALU = " << mul_per_clock << " 个ALU\n";
    std::cout << "    - 频率：" << v100_clock / 1e9 << " GHz\n";
    std::cout << "    - 峰值计算能力：" << std::fixed << std::setprecision(0) 
              << (mul_per_clock * v100_clock / 1e12) << " TFLOPs\n";
    std::cout << "    - 内存带宽：" << v100_bandwidth / 1e9 << " GB/s (HBM2)\n\n";

    std::cout << "    需求带宽：" << std::setprecision(1) 
              << required_bw / 1e12 << " TB/s\n";
    std::cout << "    可用带宽：" << v100_bandwidth / 1e12 << " TB/s\n";
    std::cout << "    此计算中的GPU效率：< " << std::setprecision(0) 
              << efficiency << "%\n" << std::endl;

    // 与CPU对比：展示即使是CPU也是带宽受限的
    double cpu_cores = 8;
    double cpu_clock = 3.2e9;          // Xeon E5v4 频率
    double cpu_bw = 76e9;              // 内存总线带宽（字节/秒）
    double cpu_alus = cpu_cores * 8 * 2; // 8核、AVX2（8宽）、2个FMA单元
    double cpu_required_bw = cpu_alus * cpu_clock * bytes_per_mul;
    double cpu_efficiency = cpu_bw / cpu_required_bw * 100.0;

    std::cout << "    8核 Xeon E5v4（3.2 GHz，76 GB/s总线）：\n";
    std::cout << "    - 需求带宽：" << std::setprecision(1) 
              << cpu_required_bw / 1e12 << " TB/s\n";
    std::cout << "    - 效率：约" << std::setprecision(0) << cpu_efficiency << "%\n" 
              << std::endl;

    std::cout << "    核心洞察：这种计算是带宽受限的！\n";
    std::cout << "    → 处理器请求数据的速度超过了内存的供给速度\n";
    std::cout << "    → 必须复用数据（时间局部性）或在多个线程间共享数据\n";
    std::cout << "    → 在现代计算中，带宽是最关键的瓶颈资源\n" << std::endl;
}

// =============================================================================
// 主函数：按顺序展示课程第3讲的所有核心概念
// =============================================================================
int main() {
    std::cout << "=== CS149 第3讲：延迟、带宽与流水线 ===\n" << std::endl;

    demo_highway_analogy();
    demo_laundry_pipeline();
    demo_instruction_pipeline();
    demo_bandwidth_bound();

    // ---- 总结 ----
    std::cout << "[5] 关键要点\n" << std::endl;
    std::cout << "    - 延迟（Latency）：完成单个操作所需的时间（降低延迟很困难）\n";
    std::cout << "    - 带宽（Bandwidth）：完成操作的速率（可通过并行度扩展）\n";
    std::cout << "    - 流水线（Pipelining）：在不降低延迟的情况下提高吞吐量\n";
    std::cout << "    - 瓶颈决定了最大吞吐量（管道中最薄弱的环节）\n";
    std::cout << "    - 内存带宽通常是现代计算中的限制因素\n";
    std::cout << "    - 策略：复用数据、在线程间共享、每次加载做更多数学运算\n";
    std::cout << "    - 流水线填充时间：达到稳态吞吐量之前的初始延迟\n";

    return 0;
}
