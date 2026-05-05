// lecture12_part1.cpp
// Distributed Matrix Multiply with Collective Communication
// Models: Tensor parallel GEMM with Reduce-Scatter / AllReduce
// Stanford CS149, Fall 2025 - Lecture 12: Mapping AI to the AI Datacenter

#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>

// Simulated timing for communication primitives
struct CommConfig {
    double bandwidth_GBs;         // Interconnect bandwidth (GB/s)
    double latency_us;            // Fixed per-message latency (microseconds)
    int numRanks;                 // Number of ranks (GPUs/RDUs)
};

// Distributed matrix: each rank holds a slice
struct DistributedMatrix {
    int globalRows, globalCols;
    int localRows, localCols;
    int rank;
    int numRanks;
    std::vector<std::vector<double>> localData;

    DistributedMatrix(int gRows, int gCols, int rank, int numRanks, bool splitCols = true)
        : globalRows(gRows), globalCols(gCols), rank(rank), numRanks(numRanks) {
        if (splitCols) {
            localRows = gRows;
            localCols = (gCols + numRanks - 1) / numRanks;
        } else {
            localRows = (gRows + numRanks - 1) / numRanks;
            localCols = gCols;
        }

        // Initialize with rank-specific data
        localData.resize(localRows, std::vector<double>(localCols, 0.0));
        for (int i = 0; i < localRows; ++i)
            for (int j = 0; j < localCols; ++j)
                localData[i][j] = (rank + 1) * 0.1;  // Simple initialization
    }
};

// Simulation of collective communication operations
class CollectiveComm {
public:
    CollectiveComm(const CommConfig& cfg) : cfg_(cfg) {}

    // Time for Reduce-Scatter: reduce partial results and scatter
    // Each rank sends (size/numRanks) bytes, receives same
    double reduceScatterTime(size_t totalBytes) const {
        // Ring algorithm: numRanks-1 steps, each sends bytesPerRank
        size_t bytesPerStep = totalBytes / cfg_.numRanks;
        double transferTime = bytesPerStep / (cfg_.bandwidth_GBs * 1e9) * 1e6;  // us
        return (cfg_.numRanks - 1) * (cfg_.latency_us + transferTime);
    }

    // Time for All-Gather: gather chunks from all ranks
    double allGatherTime(size_t totalBytes) const {
        // Same complexity as Reduce-Scatter (symmetric)
        return reduceScatterTime(totalBytes);
    }

    // Time for All-Reduce (Reduce-Scatter + All-Gather)
    double allReduceTime(size_t totalBytes) const {
        return reduceScatterTime(totalBytes) + allGatherTime(totalBytes);
    }

    // Time for All-to-All: each rank sends to every other rank
    double allToAllTime(size_t bytesPerRank) const {
        double transferTime = bytesPerRank / (cfg_.bandwidth_GBs * 1e9) * 1e6;
        return (cfg_.numRanks - 1) * (cfg_.latency_us + transferTime);
    }

    // Time for point-to-point send/recv (pipeline parallel)
    double sendRecvTime(size_t bytes) const {
        return cfg_.latency_us + bytes / (cfg_.bandwidth_GBs * 1e9) * 1e6;
    }

private:
    CommConfig cfg_;
};

// Distributed GEMM simulation
class DistributedGEMM {
public:
    DistributedGEMM(int M, int N, int K, int numRanks, const CommConfig& comm)
        : M_(M), N_(N), K_(K), numRanks_(numRanks), comm_(comm) {}

    void simulate() {
        std::cout << "══════════════════════════════════════════════════════════════\n";
        std::cout << "Distributed GEMM: A[" << M_ << "x" << K_ << "] * B[" << K_ << "x" << N_
                  << "] → C[" << M_ << "x" << N_ << "]\n";
        std::cout << "Ranks: " << numRanks_ << " | Split: K dimension\n";
        std::cout << "══════════════════════════════════════════════════════════════\n\n";

        // Step 1: Local GEMM computation
        double flopsPerRank = 2.0 * M_ * (K_ / numRanks_) * N_;
        double totalFlops = flopsPerRank * numRanks_;
        double computeTime_us = flopsPerRank / (989e12) * 1e6;  // Assume 989 TFLOPS (H100 tensor)

        std::cout << "Step 1: Local GEMM (per rank)\n";
        std::cout << "  FLOPs/rank: " << std::scientific << std::setprecision(2)
                  << flopsPerRank << "\n";
        std::cout << "  Compute time/rank: " << std::fixed << std::setprecision(1)
                  << computeTime_us << " us\n\n";

        // Step 2: Reduce-Scatter to combine partial results
        // Each rank has [M x N] partial result → reduce-scatter → [M x N/numRanks] final
        size_t resultBytes = M_ * N_ * 4;  // fp32 = 4 bytes
        double rsTime = comm_.reduceScatterTime(resultBytes);

        std::cout << "Step 2: Reduce-Scatter\n";
        std::cout << "  Data size: " << resultBytes / 1e6 << " MB\n";
        std::cout << "  RS time: " << rsTime << " us\n\n";

        // Step 3: All-Gather to distribute final result
        double agTime = comm_.allGatherTime(resultBytes);

        std::cout << "Step 3: All-Gather (optional, if full result needed)\n";
        std::cout << "  AG time: " << agTime << " us\n\n";

        // Total for AllReduce path
        double arTime = rsTime + agTime;
        double totalTime = computeTime_us + arTime;

        std::cout << "Summary:\n";
        std::cout << "  Compute:     " << std::setw(10) << computeTime_us << " us\n";
        std::cout << "  AllReduce:   " << std::setw(10) << arTime << " us\n";
        std::cout << "  Total:       " << std::setw(10) << totalTime << " us\n";
        std::cout << "  Utilization: " << std::setw(10) << std::setprecision(1)
                  << (100.0 * computeTime_us / totalTime) << "%\n\n";
    }

private:
    int M_, N_, K_, numRanks_;
    CollectiveComm comm_;
};

// Analyze compute-communication overlap (key RDU advantage)
void analyzeOverlap() {
    std::cout << "══════════════════════════════════════════════════════════════\n";
    std::cout << "Compute-Communication Overlap Analysis\n";
    std::cout << "(Based on lecture data: BS=16, M=24576, K=131074, N=8192)\n";
    std::cout << "══════════════════════════════════════════════════════════════\n\n";

    struct OverlapCase {
        int numSockets;
        double totalTFLOPS;
        double computeRoofline_ms;   // @100% utilization
        double reduceScatter_ms;     // @100% link utilization
        double theoreticalNoOverlap; // % utilization without overlap
        double measuredWithOverlap;  // % utilization with overlap (from lecture)
    };

    std::vector<OverlapCase> cases = {
        {8,  12744, 66.3, 8.6,  88.5, 72.0},
        {16, 25488, 33.1, 9.7,  77.0, 75.0},
        {32, 50976, 16.5, 15.0, 52.0, 79.0},
    };

    std::cout << std::left
              << std::setw(14) << "Sockets"
              << std::setw(16) << "Total TFLOPS"
              << std::setw(20) << "Roofline (ms)"
              << std::setw(18) << "RS Time (ms)"
              << std::setw(22) << "No Overlap Util%"
              << std::setw(18) << "With Overlap%"
              << "Gain\n";
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

    std::cout << "\nKey insight: Without overlap, utilization drops from 88% to 52%\n";
    std::cout << "as we scale from 8 to 32 sockets.\n";
    std::cout << "With overlap (RDU), utilization stays 70-79% across all scales.\n";
    std::cout << "Overlap: AllReduce fully overlapped with weight load + compute.\n\n";
}

// Parallelism strategy analyzer
void analyzeParallelismStrategies() {
    std::cout << "══════════════════════════════════════════════════════════════\n";
    std::cout << "Parallelism Strategies for AI Training\n";
    std::cout << "══════════════════════════════════════════════════════════════\n\n";

    struct Strategy {
        std::string name;
        std::string splitDim;
        std::string commPrimitive;
        double commVolume_GB;        // GB per step for a large model
        std::string scalingNote;
    };

    std::vector<Strategy> strategies = {
        {"Data Parallel (DP)",    "Batch",        "Reduce-Scatter + All-Gather", 2.0,  "Linear in model size"},
        {"Tensor Parallel (TP)",  "Hidden dim",    "Reduce-Scatter + All-Gather", 8.0,  "High comm, within node"},
        {"Pipeline Parallel (PP)","Layers",       "Send-Recv (P2P)",            1.0,  "Low comm, bubbles"},
        {"Expert Parallel (EP)",  "MoE experts",  "All-to-All",                 4.0,  "Sparse, selective"},
        {"Sequence Parallel (SP)","Seq length",   "Reduce-Scatter",             0.5,  "Merged with TP"},
        {"Context Parallel (CP)", "Context tokens","All-Reduce",                1.0,  "Long context"},
    };

    std::cout << std::left
              << std::setw(22) << "Strategy"
              << std::setw(16) << "Split Dim"
              << std::setw(30) << "Communication"
              << std::setw(15) << "Comm Vol (GB)"
              << "Notes\n";
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

// Analyze scaling from lecture data (TP, PP, DP combinations)
void analyzeScalingTable() {
    std::cout << "══════════════════════════════════════════════════════════════\n";
    std::cout << "Scaling Table (from lecture: sequence length 2048)\n";
    std::cout << "══════════════════════════════════════════════════════════════\n\n";

    struct ScalingEntry {
        double params_B;
        int attentionHeads;
        int hiddenSize;
        int numLayers;
        int tpSize, ppSize, mpSize, dpSize, numGPUs;
        int batchSize;
        double peakFlopsPct;
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
              << std::setw(10) << "Params"
              << std::setw(8)  << "Heads"
              << std::setw(10) << "Hidden"
              << std::setw(8)  << "Layers"
              << std::setw(8)  << "TP"
              << std::setw(8)  << "PP"
              << std::setw(8)  << "MP"
              << std::setw(8)  << "DP"
              << std::setw(10) << "GPUs"
              << std::setw(12) << "% Peak\n";
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

    std::cout << "\nObservations:\n";
    std::cout << "  - Utilization plateaus around 41-49% for large models\n";
    std::cout << "  - TP increases with model size (hidden dim grows)\n";
    std::cout << "  - PP needed for largest models (530B+: 35-64 stages)\n";
    std::cout << "  - DP ~ batch size / micro-batch; total GPUs = TP × PP × DP\n\n";
}

int main() {
    std::cout << "=== Lecture 12: Distributed AI Computation ===\n";
    std::cout << "Stanford CS149 - Mapping AI to the AI Datacenter\n\n";

    // Part 1: Distributed GEMM with communication
    CommConfig nvlink = {
        900.0,   // 900 GB/s NVLink bidirectional
        1.0,     // 1 us latency
        8        // 8 GPUs in a DGX node
    };

    DistributedGEMM dgemm(24576, 8192, 131072, 8, nvlink);
    dgemm.simulate();

    // Part 2: Overlap analysis
    analyzeOverlap();

    // Part 3: Parallelism strategies
    analyzeParallelismStrategies();

    // Part 4: Scaling table
    analyzeScalingTable();

    // Summary
    std::cout << "══════════════════════════════════════════════════════════════\n";
    std::cout << "Key Takeaways:\n";
    std::cout << "1. Communication can dominate without compute-communication overlap\n";
    std::cout << "2. RDU advantage: AllReduce fully overlapped, no HBM consumed\n";
    std::cout << "3. TP + PP + DP combine for model scaling; TP grows with hidden dim\n";
    std::cout << "4. 100x fewer kernel calls on RDU (3 vs 800 per token)\n";
    std::cout << "5. Dataflow fusion eliminates GBs of off-chip intermediate traffic\n";
    std::cout << "══════════════════════════════════════════════════════════════\n";

    return 0;
}
