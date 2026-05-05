// lecture7_part3.cpp
// Stanford CS149, Lecture 7: GPU Architecture & CUDA Programming
// Part 3: GPU Memory Hierarchy Simulation
//
// Simulates the three distinct GPU memory types:
//   - Global memory   (DRAM): large, slow, shared by all threads
//   - Shared memory    (SRAM): per-block, fast, on-chip
//   - Registers        (RF): per-thread, fastest
//
// Also demonstrates:
//   - Warp-based execution (32 threads per warp, SIMD-style)
//   - Thread block scheduling with resource constraints
//   - Atomic operations on global and shared memory
//
// Compile: g++ -std=c++17 -pthread lecture7_part3.cpp -o lecture7_part3
// Run: ./lecture7_part3

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <iomanip>
#include <algorithm>
#include <cstring>

// ============================================================================
// Simulated GPU architecture parameters (V100-inspired)
// ============================================================================

constexpr int WARP_SIZE          = 32;
constexpr int MAX_WARPS_PER_SM   = 64;
constexpr int MAX_THREADS_PER_SM = MAX_WARPS_PER_SM * WARP_SIZE;  // 2048
constexpr int SHARED_MEM_PER_SM  = 128 * 1024;  // 128 KB (shared + L1)
constexpr int REGISTERS_PER_SM   = 256 * 1024;   // 256 KB total registers
constexpr int NUM_SM             = 4;   // Simulated 4-SM GPU

// ============================================================================
// Simulated GPU memory
// ============================================================================

// Global memory (DRAM) — slow, shared by all SMs
class GlobalMemory {
public:
    std::vector<int> data;

    GlobalMemory(size_t size) : data(size, 0) {}

    int read(size_t addr) {
        // Simulate DRAM latency: count access (in real GPU, ~200-400 cycles)
        access_count++;
        return data[addr];
    }

    void write(size_t addr, int value) {
        access_count++;
        data[addr] = value;
    }

    void atomicAdd(size_t addr, int value) {
        access_count += 2;  // Atomic ops are more expensive
        std::lock_guard<std::mutex> lock(gmem_mutex);
        data[addr] += value;
    }

    long long getAccessCount() const { return access_count; }

private:
    std::mutex gmem_mutex;
    std::atomic<long long> access_count{0};
};

// ============================================================================
// Streaming Multiprocessor (SM) simulation
// Each SM has:
//   - Shared memory (per-block)
//   - Register file (per-thread)
//   - Execution units (simulated as CPU threads)
// ============================================================================

struct SM {
    int id;
    int sharedMemUsed  = 0;
    int registersUsed  = 0;
    int activeWarps    = 0;
    int activeThreads  = 0;
    int maxSharedMem   = SHARED_MEM_PER_SM;
    int maxRegisters   = REGISTERS_PER_SM;
    int maxThreads     = MAX_THREADS_PER_SM;

    // Simulated shared memory (per-SM, partitioned among blocks)
    std::vector<int> sharedMem;

    SM(int _id) : id(_id), sharedMem(SHARED_MEM_PER_SM / sizeof(int), 0) {}

    bool canFitBlock(int blockThreads, int blockSharedBytes, int blockRegsPerThread) {
        if (activeThreads + blockThreads > maxThreads) return false;
        if (sharedMemUsed + blockSharedBytes > maxSharedMem) return false;
        if (registersUsed + blockThreads * blockRegsPerThread > maxRegisters) return false;
        return true;
    }

    void allocateBlock(int blockThreads, int blockSharedBytes, int blockRegsPerThread) {
        activeThreads  += blockThreads;
        sharedMemUsed  += blockSharedBytes;
        registersUsed  += blockThreads * blockRegsPerThread;
        activeWarps    = (activeThreads + WARP_SIZE - 1) / WARP_SIZE;
    }

    void deallocateBlock(int blockThreads, int blockSharedBytes, int blockRegsPerThread) {
        activeThreads  -= blockThreads;
        sharedMemUsed  -= blockSharedBytes;
        registersUsed  -= blockThreads * blockRegsPerThread;
        activeWarps    = (activeThreads + WARP_SIZE - 1) / WARP_SIZE;
    }

    void printStatus() const {
        std::cout << "  SM[" << id << "]: "
                  << activeThreads << "/" << maxThreads << " threads, "
                  << activeWarps << " warps, "
                  << sharedMemUsed << "/" << maxSharedMem << " bytes shared, "
                  << registersUsed << "/" << maxRegisters << " bytes regs\n";
    }
};

// ============================================================================
// GPU Work Scheduler
// Distributes thread blocks to SMs based on resource availability
// ============================================================================

class GPUWorkScheduler {
public:
    std::vector<SM> sms;
    GlobalMemory gmem;

    GPUWorkScheduler(int numSMs, size_t globalMemSize)
        : gmem(globalMemSize)
    {
        for (int i = 0; i < numSMs; i++) {
            sms.emplace_back(i);
        }
    }

    // Schedule a thread block to the first SM that has room
    int scheduleBlock(int blockIdx, int blockThreads,
                      int blockSharedBytes, int blockRegsPerThread)
    {
        for (auto& sm : sms) {
            if (sm.canFitBlock(blockThreads, blockSharedBytes, blockRegsPerThread)) {
                sm.allocateBlock(blockThreads, blockSharedBytes, blockRegsPerThread);
                return sm.id;
            }
        }
        return -1;  // No SM available — must wait
    }

    void completeBlock(int smId, int blockThreads,
                       int blockSharedBytes, int blockRegsPerThread)
    {
        sms[smId].deallocateBlock(blockThreads, blockSharedBytes, blockRegsPerThread);
    }

    void printStatus() const {
        std::cout << "GPU Status:\n";
        for (const auto& sm : sms) {
            sm.printStatus();
        }
    }

    long long getGlobalAccessCount() const {
        return gmem.getAccessCount();
    }
};

// ============================================================================
// Histogram computation using atomics on global memory
// (Lecture 7 example: atomicAdd on shared variable in global memory)
// ============================================================================

void computeHistogram(GPUWorkScheduler& gpu,
                      const std::vector<int>& data,
                      int numBins)
{
    std::cout << "\n--- Histogram Computation (Atomics on Global Memory) ---\n";
    std::cout << "Input data size: " << data.size() << ", bins: " << numBins << "\n";

    // Allocate histogram bins in global memory
    int binBase = 1000;  // Address offset for histogram bins

    constexpr int HIST_THREADS_PER_BLK = 64;
    constexpr int HIST_SHARED_BYTES    = 0;     // No shared memory needed
    constexpr int HIST_REGS_PER_THREAD = 4 * 4; // ~4 int registers

    int numBlocks = (data.size() + HIST_THREADS_PER_BLK - 1) / HIST_THREADS_PER_BLK;

    std::vector<std::thread> blockThreads;

    for (int blk = 0; blk < numBlocks; blk++) {
        int smId = gpu.scheduleBlock(blk, HIST_THREADS_PER_BLK,
                                     HIST_SHARED_BYTES, HIST_REGS_PER_THREAD);
        if (smId < 0) {
            std::cout << "  Block " << blk << " cannot be scheduled (resources full)\n";
            continue;
        }

        int startIdx = blk * HIST_THREADS_PER_BLK;

        blockThreads.emplace_back([&gpu, &data, startIdx, binBase, numBins]() {
            for (int t = 0; t < HIST_THREADS_PER_BLK; t++) {
                int idx = startIdx + t;
                if (idx >= static_cast<int>(data.size())) break;

                int bin = data[idx] % numBins;
                if (bin < 0) bin += numBins;
                // atomicAdd(&counts[bin], 1)
                gpu.gmem.atomicAdd(binBase + bin, 1);
            }
        });

        gpu.completeBlock(smId, HIST_THREADS_PER_BLK,
                         HIST_SHARED_BYTES, HIST_REGS_PER_THREAD);
    }

    for (auto& t : blockThreads) t.join();

    // Print results
    std::cout << "Histogram:\n";
    for (int b = 0; b < numBins; b++) {
        std::cout << "  bin[" << b << "]: " << gpu.gmem.read(binBase + b) << "\n";
    }
    std::cout << "Global memory accesses: " << gpu.getGlobalAccessCount() << "\n";
}

// ============================================================================
// Demonstrate warp-based execution model
// 32 threads form a warp; all threads in a warp execute same instruction
// ============================================================================

void simulateWarpExecution()
{
    std::cout << "\n--- Warp Execution Simulation (SIMD) ---\n";

    constexpr int WARP_COUNT = 4;
    int perWarpData[WARP_SIZE];

    std::cout << "Launching " << WARP_COUNT * WARP_SIZE
              << " threads in " << WARP_COUNT << " warps:\n";

    // Each warp is launched as a group of 32 threads
    std::vector<std::thread> warpThreads;

    for (int w = 0; w < WARP_COUNT; w++) {
        // In hardware: warp selector picks one warp per clock
        // Each warp has its own set of registers for 32 threads
        warpThreads.emplace_back([w, &perWarpData]() {
            // Initialize per-thread data (in registers)
            for (int lane = 0; lane < WARP_SIZE; lane++) {
                // Each "thread" has its own register values
                perWarpData[lane] = w * 100 + lane;
            }

            // SIMD operation: all 32 threads execute the same multiply
            for (int lane = 0; lane < WARP_SIZE; lane++) {
                perWarpData[lane] = perWarpData[lane] * 2 + 1;
            }
        });
    }

    for (auto& t : warpThreads) t.join();

    std::cout << "Each warp executed 32-way SIMD multiply-add on its data.\n";
    std::cout << "In real GPU: 16 ALUs per sub-core, so 32-thread warp ";
    std::cout << "takes 2 clocks per instruction.\n";
}

// ============================================================================
// Demonstrate divergent execution within a warp
// When threads take different branches, execution serializes
// ============================================================================

void simulateWarpDivergence()
{
    std::cout << "\n--- Warp Divergence Example ---\n";

    int results[WARP_SIZE];

    // Simulate a kernel with conditional branch
    for (int lane = 0; lane < WARP_SIZE; lane++) {
        if (lane % 2 == 0) {
            // Even threads take this path
            results[lane] = lane * 10;
        } else {
            // Odd threads take this path
            results[lane] = lane * 10 + 1000;
        }
    }

    std::cout << "Warp of 32 threads with divergent branches:\n";
    std::cout << "  Even threads: results[lane] = lane * 10\n";
    std::cout << "  Odd threads:  results[lane] = lane * 10 + 1000\n";
    std::cout << "Sample results: ";
    for (int i = 0; i < 8; i++) std::cout << results[i] << " ";
    std::cout << "\n";

    std::cout << "In real GPU: even threads execute first (masked), ";
    std::cout << "then odd threads execute (masked).\n";
    std::cout << "This serialization reduces performance by ~50% for this warp.\n";
}

// ============================================================================
// Memory latency comparison simulation
// ============================================================================

void simulateMemoryLatency()
{
    std::cout << "\n--- Memory Hierarchy Latency Comparison ---\n";

    constexpr int ITERATIONS = 1000;

    // Simulate access costs (in cycles)
    // Register:    1 cycle
    // Shared memory: ~20 cycles (on-chip SRAM)
    // Global memory: ~300-500 cycles (HBM DRAM)

    struct MemoryStats {
        const char* name;
        int latencyCycles;
        double bandwidthGBps;
    };

    MemoryStats memories[] = {
        {"Register File",   1,   8000.0},
        {"Shared Memory",   20,  10000.0},
        {"Global (HBM)",    400, 900.0},
    };

    std::cout << std::left << std::setw(18) << "Memory Type"
              << std::setw(14) << "Latency"
              << std::setw(16) << "Bandwidth"
              << "Relative Speed\n";
    std::cout << std::string(60, '-') << "\n";

    for (const auto& mem : memories) {
        double relativeSpeed = static_cast<double>(memories[2].latencyCycles)
                               / mem.latencyCycles;
        std::cout << std::left << std::setw(18) << mem.name
                  << "~" << std::setw(13) << (std::to_string(mem.latencyCycles) + " cycles")
                  << std::setw(15) << (std::to_string(static_cast<int>(mem.bandwidthGBps)) + " GB/s")
                  << std::fixed << std::setprecision(0) << relativeSpeed << "x faster\n";
    }

    std::cout << "\nKey insight: shared memory is ~20x faster than global memory.\n";
    std::cout << "Cooperative data loading into shared memory amortizes\n";
    std::cout << "the cost of global memory access across many threads.\n";
}

// ============================================================================
// main
// ============================================================================

int main()
{
    std::cout << "==================================================\n";
    std::cout << "Lecture 7 Part 3: GPU Memory Hierarchy Simulation\n";
    std::cout << "==================================================\n";

    // Initialize GPU simulator
    GPUWorkScheduler gpu(NUM_SM, 4096);

    std::cout << "\nGPU Configuration:\n";
    std::cout << "  Streaming Multiprocessors: " << NUM_SM << "\n";
    std::cout << "  Max threads per SM: " << MAX_THREADS_PER_SM << "\n";
    std::cout << "  Warp size: " << WARP_SIZE << "\n";
    std::cout << "  Shared memory per SM: " << SHARED_MEM_PER_SM / 1024 << " KB\n";
    std::cout << "  Registers per SM: " << REGISTERS_PER_SM / 1024 << " KB\n";

    gpu.printStatus();

    // 1. Demonstrate thread block scheduling with resource constraints
    std::cout << "\n--- Thread Block Scheduling ---\n";
    {
        // Simulate convolution block from lecture:
        // 128 threads, 520 bytes shared memory, 4 int regs per thread
        constexpr int CONV_THREADS   = 128;
        constexpr int CONV_SHARED    = 520;   // 130 floats = 520 bytes
        constexpr int CONV_REGS      = 4 * 4; // 4 int registers, 4 bytes each

        int blocksScheduled = 0;
        for (int blk = 0; blk < 10; blk++) {
            int smId = gpu.scheduleBlock(blk, CONV_THREADS,
                                         CONV_SHARED, CONV_REGS);
            if (smId >= 0) {
                std::cout << "  Block " << blk << " → SM[" << smId << "]\n";
                blocksScheduled++;
            } else {
                std::cout << "  Block " << blk << " → NO ROOM\n";
            }
        }
        std::cout << "  Total blocks scheduled: " << blocksScheduled << "\n";
        std::cout << "  (Each block: " << CONV_THREADS << " threads, "
                  << CONV_SHARED << "B shared, "
                  << CONV_REGS << "B regs/thread)\n";
    }
    gpu.printStatus();

    // 2. Histogram with atomics
    std::vector<int> testData(200);
    for (size_t i = 0; i < testData.size(); i++) {
        testData[i] = static_cast<int>(i);
    }
    computeHistogram(gpu, testData, 10);

    // 3. Warp execution
    simulateWarpExecution();

    // 4. Warp divergence
    simulateWarpDivergence();

    // 5. Memory latency
    simulateMemoryLatency();

    std::cout << "\n==================================================\n";
    std::cout << "Key concepts demonstrated:\n";
    std::cout << "  - Three GPU memory types: global, shared, registers\n";
    std::cout << "  - Resource-based thread block scheduling\n";
    std::cout << "  - Atomic operations on global memory\n";
    std::cout << "  - Warp-based SIMD execution (32 threads)\n";
    std::cout << "  - Branch divergence impact on performance\n";
    std::cout << "  - Memory hierarchy latency comparison\n";
    std::cout << "==================================================\n";

    return 0;
}
