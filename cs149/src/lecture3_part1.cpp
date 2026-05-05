// lecture3_part1.cpp - Latency vs Bandwidth + Pipeline Simulation
// =============================================================================
// Key concepts from CS149 Lecture 3:
//  - Latency: time to complete one operation (e.g., driving SF → Stanford: 0.5h)
//  - Bandwidth: rate of completing operations (e.g., 4 cars/hour on 4-lane highway)
//  - Analogy: drive faster (reduce latency) vs. build more lanes (increase bandwidth)
//  - Memory bandwidth: rate at which memory can provide data (e.g., 900 GB/s V100)
//  - Bandwidth-limited computation: processors request data faster than memory can supply
//  - Instruction pipeline: IF → D → EX → WB (latency 4 cycles, throughput 1/cycle)
//  - Laundry pipelining: overlap wash, dry, fold stages
//  - Pipe analogy: max flow = bottleneck (min of connected pipes)
//
// Compile: g++ -std=c++17 -O2 lecture3_part1.cpp -o lecture3_part1
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
// Highway (car) analogy for latency vs bandwidth
// ---------------------------------------------------------------------------
void demo_highway_analogy() {
    std::cout << "[1] Highway Analogy: Latency vs. Bandwidth\n" << std::endl;
    std::cout << "    Distance: San Francisco → Stanford (~50 km)\n\n";

    struct Scenario {
        std::string name;
        double speed_kmph;   // car velocity
        int lanes;            // number of lanes
        double spacing_km;    // spacing between cars
    };

    std::vector<Scenario> scenarios = {
        {"Baseline", 100.0, 1, 50.0},
        {"Drive faster", 200.0, 1, 50.0},
        {"More lanes", 100.0, 4, 50.0},
        {"Closer spacing", 100.0, 1, 1.0},
        {"Closer + more lanes", 100.0, 4, 1.0},
    };

    std::cout << "    " << std::left << std::setw(22) << "Scenario"
              << std::setw(12) << "Latency(h)"
              << std::setw(16) << "Throughput(c/h)"
              << "Notes\n";
    std::cout << "    " << std::string(70, '-') << std::endl;

    const double distance = 50.0; // km

    for (const auto& s : scenarios) {
        double latency = distance / s.speed_kmph;
        double throughput = s.lanes * s.speed_kmph / s.spacing_km;

        std::cout << "    " << std::left << std::setw(22) << s.name
                  << std::setw(12) << std::fixed << std::setprecision(2) << latency
                  << std::setw(16) << std::setprecision(1) << throughput;
        
        if (s.name == "Baseline") std::cout << "(1 car on highway at a time)";
        else if (s.name == "Drive faster") std::cout << "(2x speed, same lanes)";
        else if (s.name == "More lanes") std::cout << "(4x lanes, same speed)";
        else if (s.name == "Closer spacing") std::cout << "(1 km spacing → 1 car/36 sec)";
        std::cout << std::endl;
    }

    std::cout << "\n    Key insight: improving throughput is NOT the same as\n"
              << "    reducing latency. Building more lanes increases throughput\n"
              << "    without reducing the time for any one car.\n" << std::endl;
}

// ---------------------------------------------------------------------------
// Laundry pipelining analogy
// Stage 1: Wash (45 min), Stage 2: Dry (60 min), Stage 3: Fold (15 min)
// ---------------------------------------------------------------------------
void demo_laundry_pipeline() {
    std::cout << "[2] Laundry Pipeline Analogy\n" << std::endl;

    struct Stage { std::string name; int minutes; };
    std::vector<Stage> stages = {
        {"Wash", 45},
        {"Dry",  60},
        {"Fold", 15}
    };

    // Sequential: 3 loads, no pipelining
    int seq_total = 0;
    for (int load = 0; load < 3; load++) {
        for (const auto& s : stages) seq_total += s.minutes;
    }

    // Pipelined: stages run in parallel for different loads
    // Bottleneck is the longest stage (dryer = 60 min)
    int bottleneck = 60; // minutes (the dryer)
    int pipeline_latency = 45 + 60 + 15; // first load: 120 min
    int pipeline_total = pipeline_latency + (3 - 1) * bottleneck;

    std::cout << "    Operation: 3 loads of laundry\n";
    std::cout << "    Stages: Wash(45min) → Dry(60min) → Fold(15min)\n\n";

    std::cout << "    Sequential: " << seq_total << " min total\n";
    std::cout << "    Pipelined:  " << pipeline_total << " min total (first load: "
              << pipeline_latency << " min)\n";
    std::cout << "    Speedup: " << std::fixed << std::setprecision(1) 
              << static_cast<double>(seq_total) / pipeline_total << "x\n" << std::endl;

    std::cout << "    Pipelining timeline:\n";
    std::cout << "    " << std::setw(8) << "Time" 
              << std::setw(14) << "Washer" 
              << std::setw(14) << "Dryer" 
              << std::setw(14) << "Folder" << std::endl;
    std::cout << "    " << std::string(50, '-') << std::endl;

    // Show timeline for 3 loads
    struct Job { int load_id; int stage; int start; int end; };
    std::vector<Job> timeline;
    int washer_available = 0, dryer_available = 0, folder_available = 0;

    for (int load = 0; load < 3; load++) {
        int wash_start = washer_available;
        int wash_end = wash_start + 45;
        washer_available = wash_end;
        timeline.push_back({load, 0, wash_start, wash_end});

        int dry_start = std::max(wash_end, dryer_available);
        int dry_end = dry_start + 60;
        dryer_available = dry_end;
        timeline.push_back({load, 1, dry_start, dry_end});

        int fold_start = std::max(dry_end, folder_available);
        int fold_end = fold_start + 15;
        folder_available = fold_end;
        timeline.push_back({load, 2, fold_start, fold_end});
    }

    int max_time = 0;
    for (const auto& j : timeline) max_time = std::max(max_time, j.end);
    // Round up to nearest 15 min
    max_time = ((max_time + 14) / 15) * 15;

    const char* names[] = {"Wash", "Dry", "Fold"};
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
                std::cout << std::setw(14) << ("  Load " + std::to_string(load_id + 1));
            } else {
                std::cout << std::setw(14) << "  idle";
            }
        }
        std::cout << std::endl;
    }

    std::cout << "\n    Key: bottleneck stage (Dry, 60 min) determines throughput = 1 load/hr\n";
    std::cout << "    Latency of 1 load = 120 min. Throughput = 1 load per 60 min.\n" << std::endl;
}

// ---------------------------------------------------------------------------
// Instruction Pipeline Simulation (IF → D → EX → WB)
// ---------------------------------------------------------------------------
void demo_instruction_pipeline() {
    std::cout << "[3] Instruction Pipeline (4-stage)\n" << std::endl;

    const int STAGES = 4;
    const char* stage_names[] = {"IF", "D ", "EX", "WB"};
    const int NUM_INSTRS = 6;

    std::cout << "    Pipeline stages: IF(etch) → D(ecode) → EX(ecute) → WB(write back)\n";
    std::cout << "    Each stage: 1 cycle. Total latency: 4 cycles per instruction.\n";
    std::cout << "    Throughput: 1 instruction per cycle (after pipeline fill).\n\n";

    // Draw timeline
    std::cout << "    ";
    for (int i = 0; i < NUM_INSTRS; i++) {
        std::cout << " instr" << std::setw(2) << i << " ";
    }
    std::cout << "\n    ";
    for (int i = 0; i < NUM_INSTRS; i++) {
        for (int s = 0; s < STAGES; s++) std::cout << "---";
    }
    std::cout << std::endl;

    // Each instruction occupies one slot per stage
    std::vector<std::vector<int>> schedule(NUM_INSTRS + STAGES - 1, 
                                            std::vector<int>(STAGES, -1));

    for (int instr = 0; instr < NUM_INSTRS; instr++) {
        for (int stage = 0; stage < STAGES; stage++) {
            schedule[instr + stage][stage] = instr;
        }
    }

    // Print cycles as rows
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

    std::cout << "\n    Total cycles for " << NUM_INSTRS << " instructions: " 
              << (NUM_INSTRS + STAGES - 1) << "\n";
    std::cout << "    Sequential (1 instr/cycle): " << (NUM_INSTRS * STAGES) << " cycles\n";
    std::cout << "    Pipeline speedup: " << std::fixed << std::setprecision(2)
              << static_cast<double>(NUM_INSTRS * STAGES) / (NUM_INSTRS + STAGES - 1) << "x\n" 
              << std::endl;

    // Large N case: throughput asymptotically = 1 instruction/cycle
    std::cout << "    For large N: throughput → 1 instruction/cycle\n";
    std::cout << "    (4x improvement over non-pipelined)\n";
    std::cout << "    Modern CPUs: ~20 stage pipelines for some instructions\n" << std::endl;
}

// ---------------------------------------------------------------------------
// Bandwidth-limited computation simulation
// From the lecture: vector element-wise multiplication
// 3 memory ops (12 bytes) for every 1 MUL → bandwidth limited
// NVIDIA V100: 5120 fp32 MULs/clock → needs ~98 TB/s, but has 900 GB/s
// ---------------------------------------------------------------------------
void demo_bandwidth_bound() {
    std::cout << "[4] Bandwidth-Limited Computation\n" << std::endl;

    // V100 specs
    double v100_sms = 80;
    double v100_alu_per_sm = 64;       // fp32 ALUs per SM
    double v100_clock = 1.6e9;         // Hz
    double v100_bandwidth = 900e9;     // bytes/second (HBM2)
    double bytes_per_mul = 12;         // 3 memory ops × 4 bytes each
    double mul_per_clock = v100_sms * v100_alu_per_sm;

    double required_bw = mul_per_clock * v100_clock * bytes_per_mul;
    double efficiency = v100_bandwidth / required_bw * 100.0;

    std::cout << "    Task: element-wise vector multiplication (A[i] * B[i])\n";
    std::cout << "    Memory ops per MUL: 3 (load A, load B, store C) = 12 bytes\n\n";

    std::cout << "    NVIDIA V100:\n";
    std::cout << "    - " << v100_sms << " SMs × " << v100_alu_per_sm 
              << " fp32 ALUs = " << mul_per_clock << " ALUs\n";
    std::cout << "    - Clock: " << v100_clock / 1e9 << " GHz\n";
    std::cout << "    - Peak compute: " << std::fixed << std::setprecision(0) 
              << (mul_per_clock * v100_clock / 1e12) << " TFLOPs\n";
    std::cout << "    - Memory bandwidth: " << v100_bandwidth / 1e9 << " GB/s (HBM2)\n\n";

    std::cout << "    Required bandwidth: " << std::setprecision(1) 
              << required_bw / 1e12 << " TB/s\n";
    std::cout << "    Available: " << v100_bandwidth / 1e12 << " TB/s\n";
    std::cout << "    GPU efficiency on this computation: < " << std::setprecision(0) 
              << efficiency << "%\n" << std::endl;

    // Compare with CPU
    double cpu_cores = 8;
    double cpu_clock = 3.2e9;
    double cpu_bw = 76e9; // bytes/s
    double cpu_alus = cpu_cores * 8 * 2; // 8-core, AVX2 (8-wide), 2 FMA units
    double cpu_required_bw = cpu_alus * cpu_clock * bytes_per_mul;
    double cpu_efficiency = cpu_bw / cpu_required_bw * 100.0;

    std::cout << "    8-core Xeon E5v4 (3.2 GHz, 76 GB/s bus):\n";
    std::cout << "    - Required bandwidth: " << std::setprecision(1) 
              << cpu_required_bw / 1e12 << " TB/s\n";
    std::cout << "    - Efficiency: ~" << std::setprecision(0) << cpu_efficiency << "%\n" 
              << std::endl;

    std::cout << "    Key insight: this computation is bandwidth limited!\n";
    std::cout << "    → Processors request data faster than memory can supply\n";
    std::cout << "    → Must reuse data (temporal locality) or share across threads\n";
    std::cout << "    → In modern computing, bandwidth is the critical resource\n" << std::endl;
}

// =============================================================================
int main() {
    std::cout << "=== CS149 Lecture 3: Latency, Bandwidth & Pipelining ===\n" << std::endl;

    demo_highway_analogy();
    demo_laundry_pipeline();
    demo_instruction_pipeline();
    demo_bandwidth_bound();

    // ---- Summary ----
    std::cout << "[5] Key Takeaways\n" << std::endl;
    std::cout << "    - Latency: time to complete one operation (reducing is hard)\n";
    std::cout << "    - Bandwidth: rate of completing operations (scale with parallelism)\n";
    std::cout << "    - Pipelining: improve throughput without reducing latency\n";
    std::cout << "    - Bottleneck determines max throughput (weakest link in pipe)\n";
    std::cout << "    - Memory bandwidth is often the limiting factor in modern computing\n";
    std::cout << "    - Strategies: reuse data, share across threads, do more math per load\n";
    std::cout << "    - Pipeline fill time: initial latency before steady-state throughput\n";

    return 0;
}
