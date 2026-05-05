// lecture3_part3.cpp - Memory Bandwidth Pipeline + Data Movement Analysis
// =============================================================================
// Key concepts from CS149 Lecture 3:
//  - Multi-threaded core bandwidth analysis: load instructions + math ops
//  - Memory bandwidth-bound execution: core stalls when data not ready
//  - Rate of math ops limited by memory bandwidth in steady state
//  - Steady-state underutilization depends on instruction and memory throughput,
//    NOT on memory latency or number of outstanding requests
//  - "Math is free": arithmetic is cheap, data movement is expensive
//  - Program should access memory infrequently to utilize processors efficiently
//  - ISPC foreach: raise abstraction, think about iterations not instances
//  - Collection-oriented programming (map, NumPy-style)
//
// Compile: g++ -std=c++17 -O2 lecture3_part3.cpp -o lecture3_part3
// =============================================================================

#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>

// ---------------------------------------------------------------------------
// Simulate the multi-threaded core bandwidth analysis from Lecture 3
// 
// Processor: 1 math op/clock, co-issues loads with math, 8 bytes/clock from mem
// Thread sequence: Load 64 bytes, add x+x, add x+y
// With enough threads to hide latency, the steady state behavior depends on
// instruction throughput vs. memory bandwidth
// ---------------------------------------------------------------------------
void demo_memory_bandwidth_pipeline() {
    std::cout << "[1] Memory Bandwidth Pipeline Analysis" << std::endl;
    std::cout << "    (Lecture 3: multi-threaded core example)" << std::endl;

    // System parameters
    double math_ops_per_clock = 1.0;
    double bytes_per_clock_from_mem = 8.0;
    int load_size = 64;
    int math_per_thread = 2;                // add x+x, add x+y
    int clocks_to_transfer = load_size / (int)bytes_per_clock_from_mem;

    std::cout << "    System specs:" << std::endl;
    std::cout << "    - ALU: " << math_ops_per_clock << " math op/clock" << std::endl;
    std::cout << "    - Memory: " << bytes_per_clock_from_mem << " bytes/clock" << std::endl;
    std::cout << "    - Each load: " << load_size << " bytes (takes " 
              << clocks_to_transfer << " clocks)" << std::endl;
    std::cout << "    - Thread: Load " << load_size 
              << "B, add x+x, add x+y" << std::endl;

    // Memory bandwidth determines max throughput:
    // each thread needs load_size bytes, memory provides bytes_per_clock
    double max_threads_per_clock = bytes_per_clock_from_mem / load_size;
    double threads_per_100_clocks = max_threads_per_clock * 100;

    std::cout << std::endl;
    std::cout << "    Max thread completion rate: " << std::fixed 
              << std::setprecision(3) << max_threads_per_clock << " per clock" << std::endl;
    std::cout << "    In 100 clocks: " << std::setprecision(1) 
              << threads_per_100_clocks << " threads complete" << std::endl;
    std::cout << "    In 100 clocks, ALU can do: " << 100 * math_ops_per_clock 
              << " math ops" << std::endl;
    std::cout << "    But only " << threads_per_100_clocks * math_per_thread 
              << " useful math ops complete (memory-limited)" << std::endl;

    double utilization = (threads_per_100_clocks * math_per_thread) 
                         / (100 * math_ops_per_clock) * 100;
    std::cout << "    Core utilization: " << std::setprecision(1) 
              << utilization << "%" << std::endl;

    std::cout << std::endl;
    std::cout << "    Key insight from lecture: in steady state, core underutilization" << std::endl;
    std::cout << "    depends ONLY on instruction throughput and memory throughput," << std::endl;
    std::cout << "    NOT on memory latency or outstanding request count." << std::endl;
    std::cout << std::endl;

    // Compare: more math per memory access improves utilization
    std::cout << "    Effect of arithmetic intensity on utilization:" << std::endl;
    std::cout << "    " << std::setw(16) << "Math per Load"
              << std::setw(18) << "Mem BW Util"
              << std::setw(16) << "Core Util" << std::endl;
    std::cout << "    " << std::string(50, '-') << std::endl;

    for (int mpl : {1, 2, 4, 8, 16}) {
        int total_bytes = load_size;  // fixed load size
        double threads_per_clock = bytes_per_clock_from_mem / total_bytes;
        double core_util = (threads_per_clock * mpl) / math_ops_per_clock * 100;
        double mem_util = 100.0; // memory is 100% busy (bottleneck)

        std::cout << "    " << std::setw(16) << mpl
                  << std::setw(18) << std::fixed << std::setprecision(1) << mem_util << "%"
                  << std::setw(16) << std::setprecision(1) << std::min(core_util, 100.0) << "%"
                  << std::endl;
    }
    std::cout << std::endl;
}

// ---------------------------------------------------------------------------
// "Math is free" demonstration: comparing arithmetic vs. memory energy cost
// Data from Lecture 1 slide on data movement energy
// ---------------------------------------------------------------------------
void demo_math_is_free() {
    std::cout << "[2] Math is Free -- Arithmetic vs. Data Movement Cost" << std::endl;

    // Ballpark numbers from lecture (pJ = picojoules)
    double int_op_cost = 1.0;        // pJ
    double fp_op_cost = 20.0;        // pJ
    double sram_read_64b = 26.0;     // pJ (on-chip SRAM, 1mm away)
    double dram_read_64b = 1200.0;   // pJ (mobile LPDDR)

    std::cout << "    Energy costs (ballpark, in picojoules):" << std::endl;
    std::cout << "    Integer op:           ~" << int_op_cost << " pJ" << std::endl;
    std::cout << "    FP op:                ~" << fp_op_cost << " pJ" << std::endl;
    std::cout << "    Read 64b from SRAM:   ~" << sram_read_64b << " pJ" << std::endl;
    std::cout << "    Read 64b from DRAM:   ~" << dram_read_64b << " pJ" << std::endl;
    std::cout << std::endl;

    // Example: computing a value vs. loading from memory
    std::cout << "    Scenario: need a value. Compute it or load it?" << std::endl;
    std::cout << "    Cost to compute (100 FP ops): " << (100 * fp_op_cost) 
              << " pJ" << std::endl;
    std::cout << "    Cost to load 64b from DRAM: " << dram_read_64b 
              << " pJ" << std::endl;
    std::cout << "    Loading ONE 64b value from DRAM costs as much as " 
              << std::fixed << std::setprecision(0) 
              << dram_read_64b / fp_op_cost << " FP operations!" << std::endl;
    std::cout << std::endl;

    std::cout << "    Implication: it is often more energy-efficient to" << std::endl;
    std::cout << "    RE-COMPUTE a value than to STORE and RELOAD it." << std::endl;
    std::cout << "    This is why modern programs should favor arithmetic over" << std::endl;
    std::cout << "    memory access whenever possible." << std::endl;
    std::cout << std::endl;

    // Bandwidth energy calculation
    double bw_gb_per_sec = 10.0;
    double reads_per_sec = bw_gb_per_sec * 1e9 / 8.0; // 10 GB/s in 64-bit reads
    double power_watts = reads_per_sec * dram_read_64b * 1e-12;

    std::cout << "    Reading " << bw_gb_per_sec << " GB/sec from DRAM:" << std::endl;
    std::cout << "    Power: " << std::setprecision(1) << power_watts 
              << " watts" << std::endl;
    std::cout << "    iPhone battery: ~7 watt-hours" << std::endl;
    std::cout << "    At " << bw_gb_per_sec << " GB/s: battery lasts ~" 
              << std::setprecision(1) << 7.0 / power_watts << " hours" << std::endl;
    std::cout << "    This is why mobile GPUs target ~1W total budget" << std::endl;
    std::cout << std::endl;
}

// ---------------------------------------------------------------------------
// Collection-oriented programming model (NumPy-style)
// From the lecture: "Don't even allow array indexing"
// Programmer writes no loops, performs no data indexing
// ---------------------------------------------------------------------------
void demo_collection_programming() {
    std::cout << "[3] Collection-Oriented Parallel Programming" << std::endl;

    // NumPy-style: X + Y operates on entire arrays
    // map(f, collection) applies f to every element

    const int N = 16;
    std::vector<int> X(N), Y(N), Z(N);

    // Initialize
    for (int i = 0; i < N; i++) {
        X[i] = i;
        Y[i] = i;
    }

    // Collection operation: Z = X + Y (element-wise, implicit parallelism)
    for (int i = 0; i < N; i++) {
        Z[i] = X[i] + Y[i];
    }

    std::cout << "    NumPy-style: Z = X + Y (element-wise vector addition)" << std::endl;
    std::cout << "    X: [";
    for (int i = 0; i < N; i++) std::cout << std::setw(3) << X[i];
    std::cout << " ]" << std::endl;
    std::cout << "    Y: [";
    for (int i = 0; i < N; i++) std::cout << std::setw(3) << Y[i];
    std::cout << " ]" << std::endl;
    std::cout << "    Z: [";
    for (int i = 0; i < N; i++) std::cout << std::setw(3) << Z[i];
    std::cout << " ]" << std::endl;
    std::cout << std::endl;

    // map(f, collection): apply function to each element
    auto addOne = [](int x) { return x + 1; };
    std::vector<int> Zplus1(N);
    for (int i = 0; i < N; i++) Zplus1[i] = addOne(Z[i]);

    std::cout << "    map(addOne, Z):" << std::endl;
    std::cout << "    Zplus1: [";
    for (int i = 0; i < N; i++) std::cout << std::setw(3) << Zplus1[i];
    std::cout << " ]" << std::endl;
    std::cout << std::endl;

    std::cout << "    Key abstraction: programmer writes no loops," << std::endl;
    std::cout << "    performs no data indexing. The runtime/compiler" << std::endl;
    std::cout << "    handles parallelism automatically." << std::endl;
    std::cout << "    This is the model behind NumPy, PyTorch, etc." << std::endl;
    std::cout << std::endl;
}

// ---------------------------------------------------------------------------
// ISPC foreach abstraction: thinking about iterations, not instances
// From the lecture: foreach declares parallel loop iterations
// Programmer says: "these are the iterations the entire gang must perform"
// ISPC implementation takes responsibility for assigning iterations to instances
// ---------------------------------------------------------------------------
void demo_foreach_abstraction() {
    std::cout << "[4] foreach Abstraction (ISPC-style)" << std::endl;

    std::cout << "    ISPC foreach semantics:" << std::endl;
    std::cout << "    foreach (i = 0 ... N) {" << std::endl;
    std::cout << "        // body -- programmer thinks about iteration i" << std::endl;
    std::cout << "    }" << std::endl;
    std::cout << std::endl;

    std::cout << "    Four possible foreach implementations:" << std::endl;
    std::cout << "    1. Single instance runs ALL iterations (sequential)" << std::endl;
    std::cout << "    2. Interleaved: loop_i += programCount, idx = loop_i + programIndex" << std::endl;
    std::cout << "    3. Blocked: each instance gets N/programCount contiguous items" << std::endl;
    std::cout << "    4. Dynamic: atomic counter assigns iterations on-demand" << std::endl;
    std::cout << std::endl;

    std::cout << "    The foreach abstraction allows:" << std::endl;
    std::cout << "    - Programmer to write nearly sequential-looking code" << std::endl;
    std::cout << "    - Compiler/runtime to choose the best scheduling strategy" << std::endl;
    std::cout << "    - Clean separation of correctness (what) from performance (how)" << std::endl;
    std::cout << std::endl;
}

// =============================================================================
int main() {
    std::cout << "=== CS149 Lecture 3: Bandwidth Pipeline + Data Movement Logic ===" << std::endl;
    std::cout << std::endl;

    demo_memory_bandwidth_pipeline();
    demo_math_is_free();
    demo_collection_programming();
    demo_foreach_abstraction();

    // ---- Summary ----
    std::cout << "[5] Key Takeaways from Lecture 3" << std::endl;
    std::cout << "    - Memory bandwidth, not latency, is the critical resource" << std::endl;
    std::cout << "    - Bandwidth-bound: ALUs idle waiting for data (low utilization)" << std::endl;
    std::cout << "    - Steady-state utilization: function of Math/Mem throughput ratio" << std::endl;
    std::cout << "    - Math is cheap (1-20 pJ), data movement is expensive (1200 pJ)" << std::endl;
    std::cout << "    - Recompute > Store/Reload when arithmetic is cheaper than memory" << std::endl;
    std::cout << "    - Collection-oriented programming: no loops, no indexing" << std::endl;
    std::cout << "    - foreach: think about iterations, let compiler handle instances" << std::endl;
    std::cout << "    - Abstraction != Implementation (key theme throughout CS149)" << std::endl;

    return 0;
}
