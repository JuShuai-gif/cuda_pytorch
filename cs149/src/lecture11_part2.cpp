// lecture11_part2.cpp
// Metapipelining Simulation
// Hierarchical coarse-grained pipeline ("pipeline of pipelines")
// Models SambaNova metapipelining and streaming dataflow execution
// Stanford CS149, Fall 2025 - Lecture 11: Programming Specialized Hardware for AI

#include <iostream>
#include <vector>
#include <queue>
#include <iomanip>
#include <string>
#include <cassert>

// Time unit (cycles or ns)
using Time = long long;

// A tile of data flowing through the pipeline
struct DataTile {
    int id;
    int m_idx;   // M dimension tile index
    int n_idx;   // N dimension tile index
    Time readyTime = 0;
};

// Pipeline stage: processes tiles and passes to next stage
class PipelineStage {
public:
    PipelineStage(const std::string& name, Time latency, int capacity = 1)
        : name_(name), latency_(latency), capacity_(capacity), busyUntil_(0) {}

    // Process a tile: returns when output is ready
    Time process(Time inputReady, Time currentTime) {
        Time start = std::max(inputReady, std::max(busyUntil_, currentTime));
        busyUntil_ = start + latency_;
        return busyUntil_;
    }

    bool canAccept(Time currentTime) const {
        return busyUntil_ <= currentTime;
    }

    const std::string& name() const { return name_; }
    Time latency() const { return latency_; }

private:
    std::string name_;
    Time latency_;
    int capacity_;
    Time busyUntil_;
};

// Double Buffer: allows concurrent read and write
class DoubleBuffer {
public:
    DoubleBuffer(const std::string& name, int size)
        : name_(name), size_(size),
          writeBuffer_(0), readBuffer_(1),
          writeReady_(0), readReady_(0),
          writeBusy_(0), readBusy_(0) {}

    // Producer writes to buffer
    Time write(Time dataReady) {
        Time start = std::max(dataReady, writeBusy_);
        writeBusy_ = start + 1;  // 1 cycle to swap
        writeReady_ = writeBusy_;
        std::swap(writeBuffer_, readBuffer_);
        return writeReady_;
    }

    // Consumer reads from buffer
    Time read(Time requestTime) {
        Time start = std::max(requestTime, std::max(readReady_, readBusy_));
        readBusy_ = start + 1;  // 1 cycle to swap
        return readReady_;
    }

    const std::string& name() const { return name_; }

private:
    std::string name_;
    int size_;
    int writeBuffer_, readBuffer_;
    Time writeReady_, readReady_;
    Time writeBusy_, readBusy_;
};

// Metapipeline: hierarchical pipeline with nested stages
// Models: METAPIPE(M/MM) { METAPIPE(N/NN) { ... } }
class MetaPipeline {
public:
    MetaPipeline(int M, int N, int K,
                 int tileM, int tileN,
                 Time loadLatency, Time computeLatency, Time storeLatency)
        : M_(M), N_(N), K_(K),
          tileM_(tileM), tileN_(tileN),
          loadLatency_(loadLatency),
          computeLatency_(computeLatency),
          storeLatency_(storeLatency),
          numM_Tiles_(M / tileM),
          numN_Tiles_(N / tileN),
          totalTime_(0) {}

    // Simulate metapipeline execution and return total time
    Time execute() {
        // METAPIPE over M dimension (outer loop)
        std::vector<Time> a_load_complete(numM_Tiles_, 0);

        for (int m = 0; m < numM_Tiles_; ++m) {
            // LOAD_TILE A: load a_tile for this M iteration
            Time aLoadStart = (m > 0) ? a_load_complete[m-1] : 0;
            a_load_complete[m] = aLoadStart + loadLatency_;

            // METAPIPE over N dimension (inner loop)
            std::vector<Time> b_load_complete(numN_Tiles_, 0);
            std::vector<Time> compute_complete(numN_Tiles_, 0);
            std::vector<Time> store_complete(numN_Tiles_, 0);

            for (int n = 0; n < numN_Tiles_; ++n) {
                // LOAD_TILE B: async load b_tile
                Time bLoadStart = std::max(
                    a_load_complete[m],
                    (n > 0) ? b_load_complete[n-1] : 0
                );
                b_load_complete[n] = bLoadStart + loadLatency_;

                // MAT_MUL: compute C = A * B (starts when both tiles ready)
                Time compStart = std::max(
                    b_load_complete[n],
                    (n > 0) ? compute_complete[n-1] : 0
                );
                compute_complete[n] = compStart + computeLatency_;

                // BUFFER + STORE_TILE: write result
                Time storeStart = std::max(
                    compute_complete[n],
                    (n > 0) ? store_complete[n-1] : 0
                );
                store_complete[n] = storeStart + storeLatency_;
            }

            totalTime_ = std::max(totalTime_, store_complete[numN_Tiles_ - 1]);
        }

        return totalTime_;
    }

    // Naive sequential execution (no pipelining)
    Time executeSequential() {
        Time totalPerTile = loadLatency_ + computeLatency_ + storeLatency_;
        // Load A once per M tile, then for each N tile: load B + compute + store
        Time seqTime = 0;
        for (int m = 0; m < numM_Tiles_; ++m) {
            seqTime += loadLatency_;  // Load A tile
            for (int n = 0; n < numN_Tiles_; ++n) {
                seqTime += loadLatency_;      // Load B tile
                seqTime += computeLatency_;   // Compute
                seqTime += storeLatency_;     // Store
            }
        }
        return seqTime;
    }

    // Ideal time (fully pipelined, no stalls)
    Time executeIdeal() {
        // First tile: full latency
        Time firstTile = loadLatency_ + computeLatency_ + storeLatency_;
        // Subsequent tiles: max(latency) per tile (bottleneck stage)
        Time stageLatency = loadLatency_;
        if (computeLatency_ > stageLatency) stageLatency = computeLatency_;
        if (storeLatency_ > stageLatency) stageLatency = storeLatency_;
        Time nTotal = numM_Tiles_ * numN_Tiles_;
        return firstTile + (nTotal - 1) * stageLatency;
    }

    void printConfig() const {
        std::cout << "Matrix: " << M_ << " x " << K_ << " * " << K_ << " x " << N_ << "\n";
        std::cout << "Tiling: " << tileM_ << " x " << tileN_ << "\n";
        std::cout << "Tiles: " << numM_Tiles_ << " (M) x " << numN_Tiles_ << " (N) = "
                  << (numM_Tiles_ * numN_Tiles_) << " total\n";
        std::cout << "Latencies: Load=" << loadLatency_ << "  Compute=" << computeLatency_
                  << "  Store=" << storeLatency_ << "\n";
    }

private:
    int M_, N_, K_;
    int tileM_, tileN_;
    Time loadLatency_, computeLatency_, storeLatency_;
    int numM_Tiles_, numN_Tiles_;
    Time totalTime_;
};

// FlashAttention-style metapipeline for attention computation
// Models: QK^T → Scale → Mask → Softmax → ×V, tiled across heads
class AttentionMetaPipeline {
public:
    AttentionMetaPipeline(int seqLen, int headDim, int numHeads,
                          Time matmulLatency, Time softmaxLatency, Time loadLatency)
        : seqLen_(seqLen), headDim_(headDim), numHeads_(numHeads),
          matmulLatency_(matmulLatency), softmaxLatency_(softmaxLatency),
          loadLatency_(loadLatency) {}

    void simulate() {
        int tileSize = 16;  // 16x16 tiles
        int numSeqTiles = seqLen_ / tileSize;

        std::cout << "FlashAttention Metapipeline Simulation\n";
        std::cout << "  SeqLen=" << seqLen_ << ", HeadDim=" << headDim_
                  << ", Heads=" << numHeads_ << "\n\n";

        // For each head, pipeline: QK^T → Scale → Mask → Softmax → ×V
        for (int h = 0; h < std::min(numHeads_, 4); ++h) {
            std::cout << "  Head " << h << " pipeline:\n";

            Time qkTime = 0, scaleTime = 0, softmaxTime = 0, pvTime = 0;
            Time totalHead = 0;

            for (int t = 0; t < numSeqTiles; ++t) {
                // QK^T matmul
                Time qkStart = (t > 0) ? qkTime : 0;
                qkTime = qkStart + matmulLatency_;

                // Scale (element-wise, fast)
                scaleTime = std::max(qkTime, scaleTime) + 5;

                // Softmax (across K dimension)
                softmaxTime = std::max(scaleTime, softmaxTime) + softmaxLatency_;

                // PV matmul
                Time pvStart = std::max(softmaxTime, pvTime);
                pvTime = pvStart + matmulLatency_;
            }

            totalHead = qkTime;
            if (softmaxTime > totalHead) totalHead = softmaxTime;
            if (pvTime > totalHead) totalHead = pvTime;

            std::cout << "    Tiles: " << numSeqTiles << "\n";
            std::cout << "    QK^T matmul time:  " << qkTime << "\n";
            std::cout << "    Softmax time:      " << softmaxTime << "\n";
            std::cout << "    PV matmul time:    " << pvTime << "\n";
            std::cout << "    Total head time:   " << totalHead << "\n\n";
        }

        // Kernel fusion benefit
        Time tilesPerHead = numSeqTiles;
        Time fusedTime = tilesPerHead * (matmulLatency_ + softmaxLatency_ +
                         matmulLatency_);
        Time unfusedTime = tilesPerHead * (matmulLatency_ + softmaxLatency_ +
                           matmulLatency_ + 3 * loadLatency_);  // 3 extra loads w/o fusion

        std::cout << "Kernel Fusion Benefit:\n";
        std::cout << "  Without fusion (separate kernels): " << unfusedTime << " cycles\n";
        std::cout << "  With fusion (dataflow, no off-chip): " << fusedTime << " cycles\n";
        std::cout << "  Savings: " << (unfusedTime - fusedTime) << " cycles ("
                  << std::fixed << std::setprecision(1)
                  << (100.0 * (unfusedTime - fusedTime) / unfusedTime) << "%)\n\n";

        std::cout << "RDU advantage: 520 MB on-chip SRAM enables aggressive fusion.\n";
        std::cout << "GPU limitation: 100 MB → intermediate results spill to HBM.\n";
    }

private:
    int seqLen_, headDim_, numHeads_;
    Time matmulLatency_, softmaxLatency_, loadLatency_;
};

int main() {
    std::cout << "=== Lecture 11: Metapipelining Simulation ===\n";
    std::cout << "Stanford CS149 - Programming Specialized Hardware for AI\n\n";

    // Part 1: Basic matmul metapipeline
    {
        std::cout << "══════════════════════════════════════════════════════════════\n";
        std::cout << "Part 1: Matmul Metapipeline\n";
        std::cout << "METAPIPE(M/MM) { METAPIPE(N/NN) { MAT_MUL } }\n";
        std::cout << "══════════════════════════════════════════════════════════════\n\n";

        // Matrix: 1024x1024 * 1024x1024, tiled 256x64
        MetaPipeline mp(1024, 1024, 1024, 256, 64,
                        100,   // Load latency (AGCU → PMU)
                        500,   // Compute latency (PCU systolic)
                        50);   // Store latency (PMU → AGCU)

        mp.printConfig();

        Time seqTime = mp.executeSequential();
        Time mpTime = mp.execute();
        Time idealTime = mp.executeIdeal();

        std::cout << "\nResults:\n";
        std::cout << "  Sequential (no pipelining):  " << seqTime << " cycles\n";
        std::cout << "  Metapipelining:              " << mpTime << " cycles\n";
        std::cout << "  Ideal (fully overlapped):    " << idealTime << " cycles\n";
        std::cout << "  Speedup (vs sequential):     " << std::fixed << std::setprecision(2)
                  << (double)seqTime / mpTime << "x\n";
        std::cout << "  Efficiency (vs ideal):       " << std::setprecision(1)
                  << (100.0 * idealTime / mpTime) << "%\n\n";

        std::cout << "Key insight: metapipelining converts nested loops into\n";
        std::cout << "streaming pipelines. Each stage executes in parallel.\n";
        std::cout << "Intermediate data stored in double buffers.\n";
        std::cout << "Works with tiling and kernel fusion.\n\n";
    }

    // Part 2: Compare different pipeline depths
    {
        std::cout << "══════════════════════════════════════════════════════════════\n";
        std::cout << "Part 2: Pipeline Depth Comparison\n";
        std::cout << "══════════════════════════════════════════════════════════════\n\n";

        struct Config {
            int M, N, K, tileM, tileN;
            Time loadL, compL, storeL;
            std::string label;
        };

        std::vector<Config> configs = {
            {256, 256, 256, 64, 64, 100, 500, 50, "Small (256x256), compute-heavy"},
            {4096, 4096, 4096, 256, 64, 100, 500, 50, "Large (4096x4096)"},
            {8192, 8192, 8192, 256, 64, 200, 500, 100, "XL (8192x8192), memory-heavy"},
        };

        std::cout << std::left
                  << std::setw(40) << "Configuration"
                  << std::setw(15) << "Sequential"
                  << std::setw(15) << "Metapipe"
                  << "Speedup\n";
        std::cout << std::string(85, '-') << "\n";

        for (const auto& c : configs) {
            MetaPipeline mp(c.M, c.N, c.K, c.tileM, c.tileN,
                           c.loadL, c.compL, c.storeL);
            Time seq = mp.executeSequential();
            Time met = mp.execute();

            std::cout << std::left
                      << std::setw(40) << c.label
                      << std::setw(15) << seq
                      << std::setw(15) << met
                      << std::fixed << std::setprecision(2) << (double)seq / met << "x\n";
        }
        std::cout << "\n";
    }

    // Part 3: FlashAttention-style metapipeline
    {
        std::cout << "══════════════════════════════════════════════════════════════\n";
        std::cout << "Part 3: FlashAttention Metapipeline\n";
        std::cout << "══════════════════════════════════════════════════════════════\n\n";

        AttentionMetaPipeline attn(2048, 64, 32,
                                   200,   // matmul latency
                                   100,   // softmax latency
                                   150);  // load latency
        attn.simulate();
    }

    // Part 4: Summary - ThunderKittens vs Metapipelining
    {
        std::cout << "══════════════════════════════════════════════════════════════\n";
        std::cout << "Part 4: Programming Model Comparison\n";
        std::cout << "══════════════════════════════════════════════════════════════\n\n";

        std::cout << "ThunderKittens (GPU H100/B100):\n";
        std::cout << "   - Embedded CUDA DSL with 16x16 tile primitive\n";
        std::cout << "   - Producer-consumer pipeline: TMA loads + MMA compute\n";
        std::cout << "   - Warp groups: 8 consumer warps, 4 producer warps\n";
        std::cout << "   - mbarrier synchronization for async coordination\n";
        std::cout << "   - B100: single-thread MMA, no warps, tcgen05\n\n";

        std::cout << "Metapipelining (SambaNova SN40L):\n";
        std::cout << "   - Hierarchical coarse-grained pipeline\n";
        std::cout << "   - Data-parallel patterns: Map, Zip, Reduce, GEMM\n";
        std::cout << "   - Double buffering for intermediate data\n";
        std::cout << "   - Token-controlled dataflow (no locks!)\n";
        std::cout << "   - Aggressive kernel fusion: 100x fewer kernel calls\n";
        std::cout << "   - 520 MB on-chip SRAM enables entire decoder in one kernel\n\n";

        std::cout << "Both models achieve asynchrony, but with different approaches:\n";
        std::cout << "   GPU: hardware-managed async (TMA) + software DSL (TK)\n";
        std::cout << "   RDU: compiler-managed spatial scheduling + metapipelining\n";
    }

    return 0;
}
