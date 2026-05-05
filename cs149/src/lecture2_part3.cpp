// lecture2_part3.cpp - Hardware Multi-Threading & Latency Hiding
// =============================================================================
// Key concepts from CS149 Lecture 2:
//  - Memory access latency (DRAM: ~248 cycles) causes processor stalls
//  - Multi-threading: interleave processing of multiple threads on the same core
//    to hide stalls (when one thread stalls, work on another)
//  - Interleaved multi-threading: each clock, core chooses one thread
//  - Simultaneous multi-threading (SMT): multiple threads per clock (e.g., Intel HT)
//  - Trade-off: more threads = better latency hiding but less per-thread storage
//  - Throughput computing: potentially increase per-thread latency to increase
//    overall system throughput
//  - More arithmetic per memory access → fewer threads needed
//  - NVIDIA V100: 80 SMs, 64 warps per SM, 32-wide SIMD → 163,840 concurrent
//    data items for maximal latency hiding
//
// Compile: g++ -std=c++17 -O2 -pthread lecture2_part3.cpp -o lecture2_part3
// =============================================================================

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <queue>
#include <cassert>

using namespace std::chrono;

// ---------------------------------------------------------------------------
// Simulate a multi-threaded core execution
// Models: instructions execute in cycles, memory loads have latency
// ---------------------------------------------------------------------------
class MultiThreadedCore {
public:
    struct ThreadState {
        int id;
        int pc = 0;                          // program counter (next instruction)
        bool stalled = false;
        int stall_remaining = 0;
        int instructions_completed = 0;
    };

    struct Config {
        int num_threads;
        int memory_latency;    // cycles for a memory load
        int math_per_load;     // number of arithmetic instructions per load
        bool verbose = false;
    };

    MultiThreadedCore(const Config& cfg) : config_(cfg) {
        for (int i = 0; i < cfg.num_threads; i++) {
            threads_.push_back({i, 0, false, 0, 0});
        }
    }

    // Run simulation for a given number of cycles
    void run(int max_cycles) {
        int cycle = 0;
        while (cycle < max_cycles) {
            // Check if any stalled threads are ready
            for (auto& t : threads_) {
                if (t.stalled) {
                    t.stall_remaining--;
                    if (t.stall_remaining <= 0) {
                        t.stalled = false;
                    }
                }
            }

            // Choose a thread to execute (round-robin among ready threads)
            bool any_progress = false;
            int start_search = round_robin_next_;
            int checked = 0;

            while (checked < config_.num_threads) {
                int idx = (start_search + checked) % config_.num_threads;
                auto& t = threads_[idx];

                if (!t.stalled) {
                    t.instructions_completed++;

                    // Check if this instruction is a load
                    if (t.instructions_completed % (config_.math_per_load + 1) == 0 &&
                        t.instructions_completed > 0) {
                        // Memory load: next instructions stall
                        t.stalled = true;
                        t.stall_remaining = config_.memory_latency;
                        
                        if (config_.verbose) {
                            std::cout << "    [cycle " << cycle << "] T" << t.id 
                                      << ": LOAD (stalls for " << t.stall_remaining 
                                      << " cycles), completed=" << t.instructions_completed 
                                      << std::endl;
                        }
                    } else {
                        if (config_.verbose) {
                            std::cout << "    [cycle " << cycle << "] T" << t.id 
                                      << ": MATH, completed=" << t.instructions_completed 
                                      << std::endl;
                        }
                    }

                    round_robin_next_ = (idx + 1) % config_.num_threads;
                    any_progress = true;
                    break;
                }
                checked++;
            }

            if (!any_progress && config_.verbose) {
                std::cout << "    [cycle " << cycle << "] STALL (all threads waiting)\n";
            }

            // Track busy cycles
            if (any_progress) busy_cycles_++;

            cycle++;
        }
        total_cycles_ = cycle;
    }

    void print_stats() const {
        std::cout << "    " << std::left << std::setw(16) << "Threads"
                  << std::setw(14) << "Math/Load"
                  << std::setw(14) << "Mem Latency"
                  << std::setw(14) << "Utilization"
                  << std::setw(16) << "Total Instr." 
                  << std::setw(16) << "Instr/Thread" << std::endl;
        std::cout << "    " << std::string(90, '-') << std::endl;

        int total_instr = 0;
        for (const auto& t : threads_) {
            total_instr += t.instructions_completed;
        }
        double util = static_cast<double>(busy_cycles_) / total_cycles_ * 100.0;

        // Same stats display: one row per unique thread count already handled by caller
        int avg_per_thread = total_instr / std::max(1, config_.num_threads);

        std::cout << "    " << std::left << std::setw(16) << config_.num_threads
                  << std::setw(14) << config_.math_per_load
                  << std::setw(14) << config_.memory_latency
                  << std::setw(14) << std::fixed << std::setprecision(1) << util << "%"
                  << std::setw(16) << total_instr
                  << std::setw(16) << avg_per_thread << std::endl;
    }

    double utilization() const {
        return static_cast<double>(busy_cycles_) / total_cycles_ * 100.0;
    }

    int total_instructions() const {
        int t = 0;
        for (const auto& th : threads_) t += th.instructions_completed;
        return t;
    }

private:
    Config config_;
    std::vector<ThreadState> threads_;
    int round_robin_next_ = 0;
    int total_cycles_ = 0;
    int busy_cycles_ = 0;
};

// ---------------------------------------------------------------------------
// Theoretical: how many threads needed for 100% utilization?
// Formula: threads_needed = ceil(1 + latency / math_per_load)
// Because during the memory stall of one thread, we must have enough other
// threads' math instructions to keep the core busy
// ---------------------------------------------------------------------------
int theoretical_threads_needed(int memory_latency, int math_per_load) {
    // During the memory latency period, we need at least latency/math_per_load
    // other threads to fill every cycle. Plus 1 for the original thread.
    return static_cast<int>(std::ceil(1.0 + static_cast<double>(memory_latency) / math_per_load));
}

// ---------------------------------------------------------------------------
// GPU-style: extreme multi-threading
// NVIDIA V100: 80 SMs × 64 warps × 32 threads/warp = 163,840 threads
// ---------------------------------------------------------------------------
void demo_gpu_style_multithreading() {
    std::cout << "[2] GPU-style Extreme Multi-Threading (V100)\n" << std::endl;

    std::cout << "    NVIDIA V100 Streaming Multiprocessor (SM):\n";
    std::cout << "    ┌──────────────────────────────────────────────────┐\n";
    std::cout << "    │ 64 warp execution contexts per SM                │\n";
    std::cout << "    │ Each warp = 32 threads (SIMD width = 32)         │\n";
    std::cout << "    │ 64 × 32 = 2048 concurrent data items per SM      │\n";
    std::cout << "    │ 80 SMs on V100                                   │\n";
    std::cout << "    │ Total: 80 × 2048 = 163,840 concurrent items      │\n";
    std::cout << "    └──────────────────────────────────────────────────┘\n" << std::endl;

    // Simulate: 1 SM with 64 warps, SIMD width 32
    // Simplified: each warp executes independently for our simulation
    const int WARPS = 64;
    const int SIMD_WIDTH = 32;
    const int mem_lat = 200;  // GPU memory latency in cycles
    const int math_per_load = 10;

    int needed = theoretical_threads_needed(mem_lat, math_per_load);
    std::cout << "    With " << mem_lat << "-cycle memory latency and " 
              << math_per_load << " math ops per load:\n";
    std::cout << "    Threads needed for 100% utilization: " << needed << "\n";
    std::cout << "    Available warps: " << WARPS << " (well beyond needed)\n";
    std::cout << "    → GPU can hide massive memory latency\n" << std::endl;
}

// ---------------------------------------------------------------------------
// CPU cache hierarchy reference
// ---------------------------------------------------------------------------
void demo_cache_latency() {
    std::cout << "[3] Memory Latency Context\n" << std::endl;
    std::cout << "    " << std::left << std::setw(20) << "Cache Level" 
              << std::setw(14) << "Latency"
              << std::setw(16) << "Typical Size" << std::endl;
    std::cout << "    " << std::string(50, '-') << std::endl;
    std::cout << "    " << std::setw(20) << "L1 Cache" 
              << std::setw(14) << "~4 cycles"
              << std::setw(16) << "32 KB" << std::endl;
    std::cout << "    " << std::setw(20) << "L2 Cache" 
              << std::setw(14) << "~12 cycles"
              << std::setw(16) << "256 KB" << std::endl;
    std::cout << "    " << std::setw(20) << "L3 Cache" 
              << std::setw(14) << "~38 cycles"
              << std::setw(16) << "8-20 MB" << std::endl;
    std::cout << "    " << std::setw(20) << "DRAM" 
              << std::setw(14) << "~248 cycles"
              << std::setw(16) << "GBs" << std::endl;
    std::cout << "    " << std::setw(20) << "GPU HBM2 (V100)" 
              << std::setw(14) << "~350-500 cyc"
              << std::setw(16) << "16 GB" << std::endl;
    std::cout << "\n    At 4 GHz, 248 cycles = 62 ns for DRAM access\n";
    std::cout << "    At 1.6 GHz (V100), 350 cycles = 219 ns for HBM2\n" << std::endl;
}

// =============================================================================
int main() {
    std::cout << "=== CS149 Lecture 2: Hardware Multi-Threading & Latency Hiding ===\n\n";

    // ---- Part 1: Latency hiding with multi-threading ----
    std::cout << "[1] Latency Hiding Through Multi-Threading\n" << std::endl;
    std::cout << "    Scenario: 3 math instructions, then 1 load (12-cycle latency)\n" << std::endl;

    std::cout << "    " << std::left << std::setw(16) << "Threads"
              << std::setw(14) << "Math/Load"
              << std::setw(14) << "Mem Latency"
              << std::setw(14) << "Utilization"
              << std::setw(16) << "Total Instr."
              << std::setw(16) << "Instr/Thread" << std::endl;
    std::cout << "    " << std::string(90, '-') << std::endl;

    // Simulate with 1, 2, 3, 4, 5 threads (matching the lecture slides)
    for (int num_threads : {1, 2, 3, 4, 5}) {
        MultiThreadedCore::Config cfg;
        cfg.num_threads = num_threads;
        cfg.memory_latency = 12;
        cfg.math_per_load = 3;
        cfg.verbose = false;

        MultiThreadedCore core(cfg);
        core.run(35); // run for 35 cycles (matches lecture slide timeline)
        core.print_stats();
    }

    int needed = theoretical_threads_needed(12, 3);
    std::cout << "\n    Theoretical threads needed for 100%: " << needed << "\n";
    std::cout << "    Formula: ceil(1 + latency / math_per_load)\n" << std::endl;

    // ---- Part 2: Varying math-to-memory ratio ----
    std::cout << "[1b] Effect of Arithmetic Intensity on Thread Requirements\n" << std::endl;

    // More math per load → fewer threads needed
    std::cout << "    " << std::left << std::setw(16) << "Math/Load"
              << std::setw(20) << "Threads needed"
              << std::setw(20) << "Threads for 100%" << std::endl;
    std::cout << "    " << std::string(56, '-') << std::endl;

    for (int mpl : {1, 3, 6, 12}) {
        int needed_t = theoretical_threads_needed(12, mpl);
        std::cout << "    " << std::setw(16) << mpl
                  << std::setw(20) << needed_t;
        
        // Verify with simulation
        MultiThreadedCore::Config cfg;
        cfg.num_threads = needed_t;
        cfg.memory_latency = 12;
        cfg.math_per_load = mpl;
        cfg.verbose = false;

        MultiThreadedCore core(cfg);
        core.run(50);
        std::cout << std::setw(20) << std::fixed << std::setprecision(1) 
                  << core.utilization() << "%" << std::endl;
    }

    // ---- Part 3: GPU Extreme Multi-threading ----
    demo_gpu_style_multithreading();

    // ---- Part 4: Cache Latency Reference ----
    demo_cache_latency();

    // ---- Part 5: Context Storage Trade-off ----
    std::cout << "[4] Execution Context Storage Trade-off\n" << std::endl;
    std::cout << "    ┌──────────────────────────────────────────────────────┐\n";
    std::cout << "    │ Many small contexts:                                  │\n";
    std::cout << "    │   + Excellent latency hiding (many threads to swap)  │\n";
    std::cout << "    │   - Limited per-thread working set (small registers) │\n";
    std::cout << "    │   - More pressure on caches                          │\n";
    std::cout << "    ├──────────────────────────────────────────────────────┤\n";
    std::cout << "    │ Few large contexts:                                   │\n";
    std::cout << "    │   + Large per-thread working set                     │\n";
    std::cout << "    │   + Better cache locality per thread                 │\n";
    std::cout << "    │   - Less latency hiding ability                      │\n";
    std::cout << "    └──────────────────────────────────────────────────────┘\n" << std::endl;

    // ---- Part 6: Key Takeaways ----
    std::cout << "[5] Key Takeaways from Lecture 2 (Multi-Threading)\n" << std::endl;
    std::cout << "    - Memory access latency (~hundreds of cycles) causes stalls\n";
    std::cout << "    - Multi-threading hides stalls by running other threads' instructions\n";
    std::cout << "    - Interleaved multi-threading: 1 thread per clock (round-robin)\n";
    std::cout << "    - Simultaneous multi-threading: multiple threads per clock (SMT)\n";
    std::cout << "    - More arithmetic per memory access → fewer threads needed\n";
    std::cout << "    - GPU: extreme multi-threading (1000s of concurrent contexts)\n";
    std::cout << "    - Throughput trade-off: individual thread slower, but overall throughput higher\n";
    std::cout << "    - Application needs: sufficient parallel work + arithmetic intensity\n";

    return 0;
}
