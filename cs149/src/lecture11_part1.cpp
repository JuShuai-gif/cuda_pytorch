// lecture11_part1.cpp
// Asynchronous Pipeline Execution Simulation
// Models producer-consumer pipeline with overlapping compute and memory access
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

// Simulated time units (cycles)
using Cycles = long long;

// Tile descriptor (like TMA copy descriptor / ThunderKittens tile)
struct Tile {
    int id;
    int rows, cols;
    Cycles loadTime;       // time to load from HBM to shared memory
    Cycles computeTime;    // time to compute (e.g., MMA on tensor cores)
    Cycles storeTime;      // time to store back to HBM
};

// Pipeline stage timing
struct PipelineConfig {
    Cycles tmaLoadCycles;      // TMA async load latency
    Cycles tensorCoreCycles;   // Tensor core MMA compute
    Cycles storeCycles;        // Store to HBM
    int numTiles;              // Total tiles to process
    int pipelineDepth;         // Number of stages in flight (TMA depth / warp groups)
};

// Synchronous (blocking) execution: LD → AO → ST → LD → AO → ST ...
class SynchronousExecutor {
public:
    Cycles execute(const PipelineConfig& cfg) {
        Cycles total = 0;
        for (int t = 0; t < cfg.numTiles; ++t) {
            total += cfg.tmaLoadCycles;       // Load tile
            total += cfg.tensorCoreCycles;    // Compute
            total += cfg.storeCycles;         // Store result
        }
        return total;
    }
};

// Asynchronous execution with producer-consumer pipeline
// Models ThunderKittens-style pipeline: producer loads tiles via TMA,
// consumer computes using tensor cores, with overlap
class AsynchronousExecutor {
public:
    AsynchronousExecutor(int pipelineDepth)
        : pipelineDepth_(pipelineDepth) {}

    int pipelineDepth() const { return pipelineDepth_; }

    Cycles execute(const PipelineConfig& cfg) {
        Cycles total = 0;
        Cycles computeDone = 0;
        Cycles loadDone = 0;

        // First tile: load must complete before compute starts
        loadDone = cfg.tmaLoadCycles;

        // Pipeline steady state: compute and load overlap
        for (int t = 0; t < cfg.numTiles; ++t) {
            // Compute tile t (can start if tile loaded)
            Cycles computeStart = std::max(loadDone, computeDone);
            computeDone = computeStart + cfg.tensorCoreCycles;

            // Start loading next tile(s) — async, non-blocking
            if (t + pipelineDepth_ < cfg.numTiles) {
                loadDone = std::max(loadDone, computeStart) + cfg.tmaLoadCycles;
            }
        }

        // Final store
        computeDone += cfg.storeCycles;

        // Wait for all loads to finish
        total = std::max(computeDone, loadDone);
        return total;
    }

private:
    int pipelineDepth_;
};

// Tile Processing Pipeline (full 3-stage: Load → Compute → Store)
// Models the H100/B100 pipeline with TMA + Tensor Cores
class FullPipeline {
public:
    FullPipeline(int pipelineDepth) : pipelineDepth_(pipelineDepth) {}

    Cycles execute(const PipelineConfig& cfg) {
        // Simulate timestamps for each stage
        // producer: TMA loads tiles into shared memory buffers
        // consumer: tensor cores read from shared memory, compute, write to registers

        struct Buffer {
            Cycles loadedAt = -1;  // When data became available
            Cycles consumedAt = -1;
        };

        std::vector<Buffer> buffers(cfg.numTiles);
        std::vector<Cycles> computeComplete(cfg.numTiles, 0);
        std::vector<Cycles> storeComplete(cfg.numTiles, 0);

        Cycles tmaBusyUntil = 0;
        Cycles tcBusyUntil = 0;
        Cycles storeBusyUntil = 0;

        int nextLoad = 0;
        int nextCompute = 0;
        int nextStore = 0;

        while (nextStore < cfg.numTiles) {
            // Producer: TMA load if buffer slot available
            if (nextLoad < cfg.numTiles &&
                nextLoad - nextCompute < pipelineDepth_) {
                Cycles loadStart = tmaBusyUntil;
                buffers[nextLoad].loadedAt = loadStart + cfg.tmaLoadCycles;
                tmaBusyUntil = buffers[nextLoad].loadedAt;
                ++nextLoad;
            }

            // Consumer: compute if data is ready
            if (nextCompute < cfg.numTiles &&
                nextCompute < nextLoad &&
                buffers[nextCompute].loadedAt <= tcBusyUntil) {
                Cycles compStart = std::max(tcBusyUntil, buffers[nextCompute].loadedAt);
                computeComplete[nextCompute] = compStart + cfg.tensorCoreCycles;
                tcBusyUntil = computeComplete[nextCompute];
                ++nextCompute;
            }

            // Store result
            if (nextStore < nextCompute &&
                computeComplete[nextStore] <= storeBusyUntil) {
                Cycles stStart = std::max(storeBusyUntil, computeComplete[nextStore]);
                storeComplete[nextStore] = stStart + cfg.storeCycles;
                storeBusyUntil = storeComplete[nextStore];
                ++nextStore;
            }

            // Advance time to next event if nothing can proceed
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
              << std::setw(18) << "Pipeline Depth" << ": " << cfg.pipelineDepth << "\n"
              << std::setw(18) << "Num Tiles" << ": " << cfg.numTiles << "\n"
              << std::setw(18) << "TMA Latency" << ": " << cfg.tmaLoadCycles << " cycles\n"
              << std::setw(18) << "TC Compute" << ": " << cfg.tensorCoreCycles << " cycles\n"
              << std::setw(18) << "Store Latency" << ": " << cfg.storeCycles << " cycles\n\n";

    std::cout << std::left
              << std::setw(22) << "Execution Mode"
              << std::setw(18) << "Total Cycles"
              << "Speedup\n";
    std::cout << std::string(55, '-') << "\n";

    std::cout << std::left
              << std::setw(22) << "Synchronous"
              << std::setw(18) << syncTime
              << "1.00x (baseline)\n";

    std::cout << std::left
              << std::setw(22) << "Async (simple)"
              << std::setw(18) << asyncTime
              << std::fixed << std::setprecision(2) << asyncSpeedup << "x\n";

    std::cout << std::left
              << std::setw(22) << "Full Pipeline"
              << std::setw(18) << pipeTime
              << std::fixed << std::setprecision(2) << pipeSpeedup << "x\n\n";
}

int main() {
    std::cout << "=== Lecture 11: Asynchronous Pipeline Execution ===\n";
    std::cout << "Stanford CS149 - Programming Specialized Hardware for AI\n";
    std::cout << "Models: Synchronous vs TMA+TensorCore pipeline\n\n";

    // Scenario 1: Compute-heavy tiles (like GEMM)
    {
        std::cout << "--- Scenario 1: Compute-Heavy (GEMM-like) ---\n";
        PipelineConfig cfg;
        cfg.tmaLoadCycles = 100;      // TMA load is fast relative to compute
        cfg.tensorCoreCycles = 1000;  // Tensor core compute dominates
        cfg.storeCycles = 50;
        cfg.numTiles = 32;
        cfg.pipelineDepth = 4;        // 4-stage pipeline (like ThunderKittens)
        printComparison(cfg);

        std::cout << "Analysis: GPU compute bound. Async helps but limited overlap.\n";
        std::cout << "  TMA load is 10% of compute → most time spent computing.\n\n";
    }

    // Scenario 2: Memory-heavy tiles (like attention, bandwidth-bound)
    {
        std::cout << "--- Scenario 2: Memory-Heavy (Attention-like) ---\n";
        PipelineConfig cfg;
        cfg.tmaLoadCycles = 800;      // Loading from HBM dominates
        cfg.tensorCoreCycles = 200;   // Quick compute
        cfg.storeCycles = 100;
        cfg.numTiles = 64;
        cfg.pipelineDepth = 8;        // Deeper pipeline to hide load latency
        printComparison(cfg);

        std::cout << "Analysis: HBM bandwidth bound. Async critical for hiding load latency.\n";
        std::cout << "  TMA load is 4x compute → overlapping is essential.\n";
        std::cout << "  ThunderKittens: default 4-stage input pipeline, 8 consumer warps.\n\n";
    }

    // Scenario 3: Balanced (typical transformer layer)
    {
        std::cout << "--- Scenario 3: Balanced Pipeline ---\n";
        PipelineConfig cfg;
        cfg.tmaLoadCycles = 500;
        cfg.tensorCoreCycles = 500;
        cfg.storeCycles = 200;
        cfg.numTiles = 16;
        cfg.pipelineDepth = 4;
        printComparison(cfg);

        std::cout << "Analysis: Near-perfect overlap possible when load ≈ compute.\n\n";
    }

    // TPU comparison
    {
        std::cout << "══════════════════════════════════════════════════════════════\n";
        std::cout << "TPU vs GPU Comparison:\n";
        std::cout << "══════════════════════════════════════════════════════════════\n\n";

        std::cout << "Google TPU v1 (Systolic Array):\n";
        std::cout << "  - No async instructions needed (dataflow execution)\n";
        std::cout << "  - Weights pre-loaded: weight-stationary\n";
        std::cout << "  - Inputs stream through: spatial + temporal locality\n";
        std::cout << "  - Key instructions: read_weights, matrix_multiply, activate\n";
        std::cout << "  - ~30% of chip area = arithmetic (vs ~5% on CPU)\n\n";

        std::cout << "NVIDIA H100/B100:\n";
        std::cout << "  - TMA: async tensor loads, hardware address gen\n";
        std::cout << "  - Tensor cores: warp-group MMA, 16x16 tiles\n";
        std::cout << "  - Requires careful pipeline management (ThunderKittens DSL)\n";
        std::cout << "  - B100: single-thread MMA, no warps, tcgen05 instructions\n\n";

        std::cout << "SambaNova SN40L (Dataflow):\n";
        std::cout << "  - No instructions → no fetch/decode overhead\n";
        std::cout << "  - Metapipelining: hierarchical coarse-grained pipeline\n";
        std::cout << "  - Token-controlled dataflow: no lock-based synchronization\n";
        std::cout << "  - 520 MB on-chip SRAM vs H100's 100 MB\n\n";
    }

    return 0;
}
