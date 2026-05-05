// lecture1_part2.cpp - Instruction-Level Parallelism & Superscalar Simulation
// =============================================================================
// Key concepts from CS149 Lecture 1:
//  - A program is a list of processor instructions
//  - Superscalar execution: processor finds independent instructions and
//    executes them in parallel on multiple execution units
//  - Instruction dependency graph determines ILP (instruction-level parallelism)
//  - Diminishing returns: most ILP exploited by ~4-wide issue
//  - The "power wall": power ∝ capacitance × voltage² × frequency
//  - End of frequency scaling + ILP tapped out → multi-core era
//
// Compile: g++ -std=c++17 -O2 lecture1_part2.cpp -o lecture1_part2
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
// Simulate a simple instruction and its dependencies
// ---------------------------------------------------------------------------
struct Instruction {
    int id;
    std::string name;
    std::vector<int> dependencies; // IDs of instructions this depends on
    int latency;                   // cycles needed to execute

    Instruction() : id(-1), name("nop"), latency(1) {}
    Instruction(int i, const std::string& n, int lat = 1) 
        : id(i), name(n), latency(lat) {}
};

// ---------------------------------------------------------------------------
// Instruction dependency graph for: a = x*x + y*y + z*z
// (From the lecture: 5-instruction program with 3 parallel multiplies)
// ---------------------------------------------------------------------------
std::vector<Instruction> build_simple_program() {
    // Register mapping: R0=x, R1=y, R2=z, R3=a (output)
    // 0: mul R0, R0, R0   (R0 = x*x)
    // 1: mul R1, R1, R1   (R1 = y*y)
    // 2: mul R2, R2, R2   (R2 = z*z)
    // 3: add R0, R0, R1   (R0 = x*x + y*y)
    // 4: add R3, R0, R2   (R3 = a)
    std::vector<Instruction> program;
    program.emplace_back(0, "mul R0, R0, R0", 1);   // no deps
    program.emplace_back(1, "mul R1, R1, R1", 1);   // no deps
    program.emplace_back(2, "mul R2, R2, R2", 1);   // no deps
    program.emplace_back(3, "add R0, R0, R1", 1);   // depends on 0,1
    program.emplace_back(4, "add R3, R0, R2", 1);   // depends on 1,3
    
    program[3].dependencies = {0, 1};
    program[4].dependencies = {2, 3};
    
    return program;
}

// ---------------------------------------------------------------------------
// A more complex 7-instruction program from the lecture
// PC 00: a = 2
// PC 01: b = 4
// PC 02: tmp2 = a + b
// PC 03: tmp3 = tmp2 + a
// PC 04: tmp4 = b + b
// PC 05: tmp5 = b * b
// PC 06: tmp6 = tmp2 + tmp4
// PC 07: tmp7 = tmp5 + tmp6
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
    program[3].dependencies = {2};     // also needs 'a' but a=2 has no dep chain
    program[4].dependencies = {1};
    program[5].dependencies = {1};
    program[6].dependencies = {2, 4};
    program[7].dependencies = {5, 6};

    return program;
}

// ---------------------------------------------------------------------------
// Superscalar scheduler: simulate an issue_width-wide processor
// Returns the number of clock cycles needed to complete all instructions
// ---------------------------------------------------------------------------
struct ScheduleResult {
    int total_cycles;
    std::vector<int> completion_cycle; // completion_cycle[i] = when instr i finishes
    std::vector<std::vector<int>> schedule_per_cycle; // which instrs issued each cycle
    double avg_ipc; // average instructions per clock
};

ScheduleResult superscalar_schedule(
    const std::vector<Instruction>& program, 
    int issue_width) 
{
    int n = static_cast<int>(program.size());
    std::vector<int> completion(n, -1);   // cycle when each instruction completes
    std::vector<bool> issued(n, false);
    std::vector<bool> completed(n, false);

    std::vector<std::vector<int>> schedule; // schedule[cycle] = list of issued instrs

    int cycle = 0;
    int completed_count = 0;

    while (completed_count < n) {
        // First: check which instructions complete this cycle
        for (int i = 0; i < n; i++) {
            if (issued[i] && !completed[i] && completion[i] <= cycle) {
                completed[i] = true;
                completed_count++;
            }
        }

        if (completed_count >= n) break;

        // Issue new instructions (up to issue_width per cycle)
        std::vector<int> issued_this_cycle;
        for (int i = 0; i < n && static_cast<int>(issued_this_cycle.size()) < issue_width; i++) {
            if (issued[i]) continue;

            // Check if all dependencies are completed
            bool deps_ready = true;
            for (int dep : program[i].dependencies) {
                if (!completed[dep]) {
                    deps_ready = false;
                    break;
                }
            }
            if (!deps_ready) continue;

            // Issue this instruction
            issued[i] = true;
            issued_this_cycle.push_back(i);
            completion[i] = cycle + program[i].latency;
        }

        schedule.push_back(issued_this_cycle);
        cycle++;
    }

    // Trim trailing empty cycles
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
// Compute maximum ILP of a program = total_instrs / critical_path_length
// ---------------------------------------------------------------------------
int compute_critical_path(const std::vector<Instruction>& program) {
    int n = static_cast<int>(program.size());
    // Simple topological longest path
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
// Display the schedule in a table format
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

    // Print as a Gantt-like chart
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
            std::cout << " (deps:";
            for (int d : program[i].dependencies) std::cout << " " << d;
            std::cout << ")";
        }
        std::cout << std::endl;
    }
}

// =============================================================================
int main() {
    std::cout << "=== CS149 Lecture 1: Instruction-Level Parallelism & Superscalar ===\n" << std::endl;

    // ---- Part 1: Simple program (5 instructions: a = x*x + y*y + z*z) ----
    std::cout << "[1] Simple program: a = x*x + y*y + z*z (5 instructions)\n" << std::endl;
    
    auto simple_prog = build_simple_program();
    
    std::cout << "    Dependency graph:\n";
    std::cout << "    I0(mul x*x)  I1(mul y*y)  I2(mul z*z)\n";
    std::cout << "          \\          /            |\n";
    std::cout << "           I3(add)                |\n";
    std::cout << "               \\                /\n";
    std::cout << "                I4(add -> result)\n" << std::endl;

    int critical = compute_critical_path(simple_prog);
    int total_instr = static_cast<int>(simple_prog.size());
    std::cout << "    Total instructions: " << total_instr << "\n";
    std::cout << "    Critical path length: " << critical << "\n";
    std::cout << "    Maximum ILP: " << std::fixed << std::setprecision(1) 
              << static_cast<double>(total_instr) / critical << "\n" << std::endl;

    // Simulate with different issue widths
    for (int width : {1, 2, 3, 4}) {
        auto result = superscalar_schedule(simple_prog, width);
        std::cout << "    Issue width = " << width << ": " 
                  << result.total_cycles << " cycles, IPC = " 
                  << std::fixed << std::setprecision(2) << result.avg_ipc 
                  << std::endl;
    }
    std::cout << std::endl;

    // Show detailed schedule for 2-wide superscalar
    std::cout << "    Detailed schedule (2-wide superscalar):\n";
    auto sched2 = superscalar_schedule(simple_prog, 2);
    print_schedule(sched2, simple_prog);
    std::cout << std::endl;

    // Show detailed schedule for 3-wide superscalar
    std::cout << "    Detailed schedule (3-wide superscalar):\n";
    auto sched3 = superscalar_schedule(simple_prog, 3);
    print_schedule(sched3, simple_prog);

    // ---- Part 2: Diminishing returns of wider superscalar ----
    std::cout << "\n[2] Diminishing returns of wider superscalar execution\n" << std::endl;
    
    auto complex_prog = build_complex_program();
    int total_complex = static_cast<int>(complex_prog.size());
    
    std::cout << "    Program: a=2, b=4, tmp2=a+b, tmp3=tmp2+a, tmp4=b+b,\n"
              << "             tmp5=b*b, tmp6=tmp2+tmp4, tmp7=tmp5+tmp6\n" << std::endl;
    std::cout << "    Total instructions: " << total_complex << "\n";
    
    critical = compute_critical_path(complex_prog);
    std::cout << "    Critical path: " << critical << "\n";
    std::cout << "    Max ILP: " << static_cast<double>(total_complex) / critical << "\n" << std::endl;

    std::cout << "    " << std::setw(14) << "Issue Width" 
              << std::setw(12) << "Cycles" 
              << std::setw(10) << "IPC"
              << std::setw(12) << "Speedup" << std::endl;
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

    // ---- Part 3: Power Wall Explanation ----
    std::cout << "\n[3] The Power Wall\n" << std::endl;
    std::cout << "    Power ∝ Capacitance × Voltage² × Frequency\n" << std::endl;
    std::cout << "    Key observations:\n";
    std::cout << "    - Dynamic power grows quadratically with voltage\n";
    std::cout << "    - Increasing frequency requires increasing voltage\n";
    std::cout << "    - High power → high heat → thermal limits\n";
    std::cout << "    - This ended the era of free frequency scaling (~2005)\n" << std::endl;

    // Calculate power scaling for different frequencies
    std::cout << "    Relative power vs frequency (simplified model):\n";
    double base_freq = 1.0;
    std::cout << "    " << std::setw(10) << "Freq(GHz)" 
              << std::setw(14) << "Relative V"
              << std::setw(14) << "Rel. Power" << std::endl;
    std::cout << "    " << std::string(38, '-') << std::endl;
    for (double f : {1.0, 2.0, 3.0, 4.0, 5.0}) {
        // Simplified: voltage scales roughly linearly with frequency
        double voltage_ratio = f / base_freq;
        double power_ratio = voltage_ratio * voltage_ratio * f / base_freq;
        std::cout << "    " << std::setw(10) << std::fixed << std::setprecision(1) << f
                  << std::setw(14) << std::setprecision(2) << voltage_ratio << "x"
                  << std::setw(14) << std::setprecision(1) << power_ratio << "x"
                  << std::endl;
    }

    // ---- Part 4: Summary ----
    std::cout << "\n[4] Key Takeaways\n" << std::endl;
    std::cout << "    - Superscalar processors execute independent instructions in parallel\n";
    std::cout << "    - ILP is limited by instruction dependencies (critical path)\n";
    std::cout << "    - Most available ILP exploited by ~4-wide superscalar\n";
    std::cout << "    - Power wall ended frequency scaling (~2005)\n";
    std::cout << "    - Result: shift to multi-core architectures\n";
    std::cout << "    - Software must be parallelized to see performance gains\n";

    return 0;
}
