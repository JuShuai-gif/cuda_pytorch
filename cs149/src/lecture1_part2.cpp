// lecture1_part2.cpp - 指令级并行与超标量模拟
// =============================================================================
// CS149 第1讲核心概念：
//   - 程序本质上是处理器指令的列表
//     每一行高级语言代码（如 C++）被编译器翻译成一系列机器指令。
//     理解指令之间的依赖关系是理解 ILP 的关键。
//
//   - 超标量执行（Superscalar Execution）：处理器在运行时自动发现
//     相互独立的指令，并在多个执行单元上并行执行它们。
//     这与程序员显式创建的多线程并行不同——
//     超标量是完全由硬件自动完成的透明优化。
//
//   - 指令依赖图决定 ILP（指令级并行度）
//     如果指令 B 需要指令 A 的结果，则 B 依赖于 A。
//     依赖关系形成了有向无环图（DAG），关键路径的长度
//     决定了程序的最小执行时间。
//     ILP = 总指令数 / 关键路径长度
//
//   - 收益递减：大多数可用的 ILP 在 ~4 发射宽度时已被挖掘
//     超过 4-wide 的超标量设计带来的额外加速微乎其微，
//     因为典型程序中的指令依赖限制了可并行度。
//     这也是 Intel 等厂商在 ~4-wide 后转向多核的原因之一。
//
//   - "功耗墙"（Power Wall）：
//     动态功耗 ∝ 电容 × 电压² × 频率
//     提高频率需要提高电压（为了保持信号稳定性），
//     而功耗随电压的平方增长 → 物理冷却极限。
//     约 2005 年后，单纯提高时钟频率不再可行。
//
//   - 频率缩放终结 + ILP 被挖掘殆尽 → 多核时代的来临
//     晶体管预算不再用于制造更复杂的单核，
//     而是用于制造更多更简单的核心。
//
// 编译: g++ -std=c++17 -O2 lecture1_part2.cpp -o lecture1_part2
// =============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <map>
#include <set>
#include <algorithm>
#include <iomanip>
#include <cmath>

// ---------------------------------------------------------------------------
// 模拟一条简单指令及其依赖关系
//
// 每条指令包含：
//   - id：唯一标识符
//   - name：指令的汇编级表示
//   - dependencies：此指令依赖的其他指令的 ID 列表
//   - latency：执行所需的时钟周期数
// ---------------------------------------------------------------------------
struct Instruction {
    int id;
    std::string name;
    std::vector<int> dependencies; // 此指令依赖的指令 ID
    int latency;                   // 执行所需的时钟周期

    Instruction() : id(-1), name("nop"), latency(1) {}
    Instruction(int i, const std::string& n, int lat = 1) 
        : id(i), name(n), latency(lat) {}
};

// ---------------------------------------------------------------------------
// 指令依赖图示例：a = x*x + y*y + z*z
// （来自课程讲义：包含 3 个可并行乘法的 5 指令程序）
//
// 依赖关系说明：
//   - I0、I1、I2（三条乘法）之间无依赖 → 可同时发射
//   - I3 依赖于 I0 和 I1（需要两者的乘积）
//   - I4 依赖于 I2 和 I3（需要中间结果）
//
// 关键路径：I0→I3→I4 或 I2→I4（长度 = 3）
// 最大 ILP = 5/3 ≈ 1.67
// ---------------------------------------------------------------------------
std::vector<Instruction> build_simple_program() {
    // 寄存器映射：R0=x、R1=y、R2=z、R3=a（输出）
    // 0: mul R0, R0, R0   (R0 = x*x)
    // 1: mul R1, R1, R1   (R1 = y*y)
    // 2: mul R2, R2, R2   (R2 = z*z)
    // 3: add R0, R0, R1   (R0 = x*x + y*y)
    // 4: add R3, R0, R2   (R3 = a)
    std::vector<Instruction> program;
    program.emplace_back(0, "mul R0, R0, R0", 1);   // 无依赖
    program.emplace_back(1, "mul R1, R1, R1", 1);   // 无依赖
    program.emplace_back(2, "mul R2, R2, R2", 1);   // 无依赖
    program.emplace_back(3, "add R0, R0, R1", 1);   // 依赖于 0,1
    program.emplace_back(4, "add R3, R0, R2", 1);   // 依赖于 1,3
    
    program[3].dependencies = {0, 1};
    program[4].dependencies = {2, 3};
    
    return program;
}

// ---------------------------------------------------------------------------
// 来自课程讲义的更复杂的 8 指令程序
// PC 00: a = 2
// PC 01: b = 4
// PC 02: tmp2 = a + b
// PC 03: tmp3 = tmp2 + a
// PC 04: tmp4 = b + b
// PC 05: tmp5 = b * b
// PC 06: tmp6 = tmp2 + tmp4
// PC 07: tmp7 = tmp5 + tmp6
//
// 依赖关系分析：
//   - I0(a=2) 和 I1(b=4) 无依赖，可同时发射
//   - I2 依赖 I0、I1；I4 和 I5 各自只依赖 I1 → I2/I4/I5 可部分并行
//   - I3 依赖 I2；I6 依赖 I2 和 I4
//   - I7 依赖 I5 和 I6（关键路径的终点）
//
// 关键路径长度 = 4（0→2→6→7 或 1→5→7）
// 最大 ILP = 8/4 = 2.0
// ---------------------------------------------------------------------------
std::vector<Instruction> build_complex_program() {
    std::vector<Instruction> program;
    program.emplace_back(0, "a = 2", 1);
    program.emplace_back(1, "b = 4", 1);
    program.emplace_back(2, "tmp2 = a + b", 1);
    program.emplace_back(3, "tmp3 = tmp2 + a", 1);
    program.emplace_back(4, "tmp4 = b + b", 1);
    program.emplace_back(5, "tmp5 = b * b", 1);
    program.emplace_back(6, "tmp6 = tmp2 + tmp4", 1);
    program.emplace_back(7, "tmp7 = tmp5 + tmp6", 1);

    program[2].dependencies = {0, 1};
    program[3].dependencies = {2};     // 也需要 'a'，但 a=2 已在依赖链中
    program[4].dependencies = {1};
    program[5].dependencies = {1};
    program[6].dependencies = {2, 4};
    program[7].dependencies = {5, 6};

    return program;
}

// ---------------------------------------------------------------------------
// 超标量调度器：模拟 issue_width 宽度的处理器
// 返回完成所有指令所需的时钟周期数
//
// 算法说明：
//   每个周期内，处理器可以"发射"多达 issue_width 条新指令，
//   前提是它们的依赖已全部满足（即依赖的指令已完成执行）。
//   这是一个简化的"按序发射、乱序完成"模型。
//
// 限制：此模拟假设所有指令延迟均为 1 周期（简化），
//       实际处理器中不同指令延迟差异很大
//       （如 mul 可能是 3 周期，load 可能是 4 周期）。
// ---------------------------------------------------------------------------
struct ScheduleResult {
    int total_cycles;
    std::vector<int> completion_cycle; // completion_cycle[i] = 指令 i 完成的周期
    std::vector<std::vector<int>> schedule_per_cycle; // 每个周期发射了哪些指令
    double avg_ipc; // 每周期平均指令数（Average Instructions Per Clock）
};

ScheduleResult superscalar_schedule(
    const std::vector<Instruction>& program, 
    int issue_width) 
{
    int n = static_cast<int>(program.size());
    std::vector<int> completion(n, -1);   // 每条指令完成的周期
    std::vector<bool> issued(n, false);
    std::vector<bool> completed(n, false);

    std::vector<std::vector<int>> schedule; // schedule[cycle] = 发射的指令列表

    int cycle = 0;
    int completed_count = 0;

    while (completed_count < n) {
        // 首先：检查哪些指令在本周期完成
        for (int i = 0; i < n; i++) {
            if (issued[i] && !completed[i] && completion[i] <= cycle) {
                completed[i] = true;
                completed_count++;
            }
        }

        if (completed_count >= n) break;

        // 发射新指令（每周期最多 issue_width 条）
        std::vector<int> issued_this_cycle;
        for (int i = 0; i < n && static_cast<int>(issued_this_cycle.size()) < issue_width; i++) {
            if (issued[i]) continue;

            // 检查所有依赖是否已完成
            bool deps_ready = true;
            for (int dep : program[i].dependencies) {
                if (!completed[dep]) {
                    deps_ready = false;
                    break;
                }
            }
            if (!deps_ready) continue;

            // 发射此指令
            issued[i] = true;
            issued_this_cycle.push_back(i);
            completion[i] = cycle + program[i].latency;
        }

        schedule.push_back(issued_this_cycle);
        cycle++;
    }

    // 去除尾部的空周期
    while (!schedule.empty() && schedule.back().empty()) {
        schedule.pop_back();
    }

    ScheduleResult result;
    result.total_cycles = static_cast<int>(schedule.size());
    result.completion_cycle = completion;
    result.schedule_per_cycle = schedule;
    result.avg_ipc = static_cast<double>(n) / result.total_cycles;

    return result;
}

// ---------------------------------------------------------------------------
// 计算程序的最大 ILP = 总指令数 / 关键路径长度
//
// 关键路径（Critical Path）是依赖图中最长的路径。
// 它决定了程序的最小可能执行时间——无论有多少执行单元，
// 程序都不可能比关键路径更快完成。
//
// 算法：拓扑排序 + 动态规划求最长路径
//   depth[i] = max(depth[dep]) + 1，遍历所有依赖 dep
// ---------------------------------------------------------------------------
int compute_critical_path(const std::vector<Instruction>& program) {
    int n = static_cast<int>(program.size());
    // 简单的拓扑最长路径算法
    std::vector<int> depth(n, 1);
    int max_depth = 1;

    for (int i = 0; i < n; i++) {
        for (int dep : program[i].dependencies) {
            depth[i] = std::max(depth[i], depth[dep] + 1);
        }
        max_depth = std::max(max_depth, depth[i]);
    }
    return max_depth;
}

// ---------------------------------------------------------------------------
// 以表格形式显示调度结果（类似甘特图）
//
// X 表示该指令在该周期被发射/执行
// . 表示该指令尚未发射或已执行完毕
// ---------------------------------------------------------------------------
void print_schedule(const ScheduleResult& result, const std::vector<Instruction>& program) {
    std::cout << "    ";
    for (size_t t = 0; t < result.schedule_per_cycle.size(); t++) {
        std::cout << "[" << std::setw(2) << t << "] ";
    }
    std::cout << "\n    ";
    for (size_t t = 0; t < result.schedule_per_cycle.size(); t++) {
        std::cout << "-----";
    }
    std::cout << std::endl;

    // 以类似甘特图的形式打印
    for (size_t i = 0; i < program.size(); i++) {
        std::cout << "    I" << std::setw(2) << i << " ";
        for (size_t t = 0; t < result.schedule_per_cycle.size(); t++) {
            bool found = false;
            for (int instr_id : result.schedule_per_cycle[t]) {
                if (instr_id == static_cast<int>(i)) {
                    found = true;
                    break;
                }
            }
            std::cout << (found ? "  X  " : "  .  ");
        }
        std::cout << " | " << program[i].name;
        if (!program[i].dependencies.empty()) {
            std::cout << " (依赖:";
            for (int d : program[i].dependencies) std::cout << " " << d;
            std::cout << ")";
        }
        std::cout << std::endl;
    }
}

// =============================================================================
int main() {
    std::cout << "=== CS149 第1讲：指令级并行与超标量 ===\n" << std::endl;

    // ---- 第一部分：简单程序（5 条指令：a = x*x + y*y + z*z） ----
    std::cout << "[1] 简单程序：a = x*x + y*y + z*z（5 条指令）\n" << std::endl;
    
    auto simple_prog = build_simple_program();
    
    std::cout << "    依赖图：\n";
    std::cout << "    I0(mul x*x)  I1(mul y*y)  I2(mul z*z)\n";
    std::cout << "          \\          /            |\n";
    std::cout << "           I3(add)                |\n";
    std::cout << "               \\                /\n";
    std::cout << "                I4(add -> 结果)\n" << std::endl;

    int critical = compute_critical_path(simple_prog);
    int total_instr = static_cast<int>(simple_prog.size());
    std::cout << "    总指令数：" << total_instr << "\n";
    std::cout << "    关键路径长度：" << critical << "\n";
    std::cout << "    最大 ILP：" << std::fixed << std::setprecision(1) 
              << static_cast<double>(total_instr) / critical << "\n" << std::endl;

    // 模拟不同的发射宽度
    for (int width : {1, 2, 3, 4}) {
        auto result = superscalar_schedule(simple_prog, width);
        std::cout << "    发射宽度 = " << width << "：" 
                  << result.total_cycles << " 周期，IPC = " 
                  << std::fixed << std::setprecision(2) << result.avg_ipc 
                  << std::endl;
    }
    std::cout << std::endl;

    // 显示 2-wide 超标量的详细调度
    std::cout << "    详细调度（2-wide 超标量）：\n";
    auto sched2 = superscalar_schedule(simple_prog, 2);
    print_schedule(sched2, simple_prog);
    std::cout << std::endl;

    // 显示 3-wide 超标量的详细调度
    std::cout << "    详细调度（3-wide 超标量）：\n";
    auto sched3 = superscalar_schedule(simple_prog, 3);
    print_schedule(sched3, simple_prog);

    // ---- 第二部分：更宽超标量的收益递减 ----
    std::cout << "\n[2] 更宽超标量执行的收益递减\n" << std::endl;
    
    auto complex_prog = build_complex_program();
    int total_complex = static_cast<int>(complex_prog.size());
    
    std::cout << "    程序：a=2, b=4, tmp2=a+b, tmp3=tmp2+a, tmp4=b+b,\n"
              << "          tmp5=b*b, tmp6=tmp2+tmp4, tmp7=tmp5+tmp6\n" << std::endl;
    std::cout << "    总指令数：" << total_complex << "\n";
    
    critical = compute_critical_path(complex_prog);
    std::cout << "    关键路径：" << critical << "\n";
    std::cout << "    最大 ILP：" << static_cast<double>(total_complex) / critical << "\n" << std::endl;

    std::cout << "    " << std::setw(14) << "发射宽度" 
              << std::setw(12) << "周期数" 
              << std::setw(10) << "IPC"
              << std::setw(12) << "加速比" << std::endl;
    std::cout << "    " << std::string(48, '-') << std::endl;

    double base_cycles = 0;
    for (int w : {1, 2, 4, 8, 16}) {
        auto res = superscalar_schedule(complex_prog, w);
        double cycles = res.total_cycles;
        if (w == 1) base_cycles = cycles;
        double sp = base_cycles / cycles;
        std::cout << "    " << std::setw(14) << w
                  << std::setw(12) << static_cast<int>(cycles)
                  << std::setw(10) << std::fixed << std::setprecision(2) << res.avg_ipc
                  << std::setw(12) << std::setprecision(2) << sp << "x"
                  << std::endl;
    }

    // ---- 第三部分：功耗墙解释 ----
    std::cout << "\n[3] 功耗墙\n" << std::endl;
    std::cout << "    功耗 ∝ 电容 × 电压² × 频率\n" << std::endl;
    std::cout << "    关键结论：\n";
    std::cout << "    - 动态功耗与电压的平方成正比增长\n";
    std::cout << "    - 提高频率需要提高电压（信号完整性要求）\n";
    std::cout << "    - 高功耗 → 高发热 → 散热物理极限\n";
    std::cout << "    - 这终结了「免费频率缩放」时代（约 2005 年）\n" << std::endl;

    // 计算不同频率下的功耗缩放
    std::cout << "    不同频率下的相对功耗（简化模型）：\n";
    double base_freq = 1.0;
    std::cout << "    " << std::setw(10) << "频率(GHz)" 
              << std::setw(14) << "相对电压"
              << std::setw(14) << "相对功耗" << std::endl;
    std::cout << "    " << std::string(38, '-') << std::endl;
    for (double f : {1.0, 2.0, 3.0, 4.0, 5.0}) {
        // 简化模型：电压大致与频率成线性比例
        double voltage_ratio = f / base_freq;
        double power_ratio = voltage_ratio * voltage_ratio * f / base_freq;
        std::cout << "    " << std::setw(10) << std::fixed << std::setprecision(1) << f
                  << std::setw(14) << std::setprecision(2) << voltage_ratio << "x"
                  << std::setw(14) << std::setprecision(1) << power_ratio << "x"
                  << std::endl;
    }

    // ---- 第四部分：总结 ----
    std::cout << "\n[4] 核心要点\n" << std::endl;
    std::cout << "    - 超标量处理器自动并行执行独立的指令\n";
    std::cout << "    - ILP 受到指令依赖的限制（关键路径决定最小执行时间）\n";
    std::cout << "    - 大多数可用 ILP 在 ~4-wide 超标量时已被挖掘\n";
    std::cout << "    - 功耗墙终结了频率缩放时代（约 2005 年）\n";
    std::cout << "    - 结果：业界转向多核架构\n";
    std::cout << "    - 软件必须被并行化才能获得性能提升\n";

    return 0;
}
