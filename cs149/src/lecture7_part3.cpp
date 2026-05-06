// lecture7_part3.cpp
// Stanford CS149 第7讲：GPU 架构与 CUDA 编程
// 第三部分：GPU 内存层次结构模拟
//
// 模拟 GPU 的三种不同内存类型及其特性：
//   - Global Memory（全局内存 / DRAM）：容量大、速度慢、所有线程共享
//       位于 GPU 芯片外部（HBM/DRAM），延迟约 300~500 cycles，
//       bandwidth 约 900 GB/s（V100），通常几百 MB 到几十 GB。
//       通过 L1/L2 cache 缓存访问，所有 SM 均可访问。
//   - Shared Memory（共享内存 / SRAM）：per-block、速度快、片上
//       位于 SM 内部，延迟约 20 cycles，bandwidth 约 10 TB/s，
//       容量 48~128 KB/block（V100 上为 96 KB 可编程）。
//       仅同一 block 内的线程可以访问，用于线程间协作和数据复用。
//   - Registers（寄存器文件 / RF）：per-thread、速度最快
//       每个 thread 独占的寄存器，延迟仅 1 cycle，带宽最高。
//       寄存器数量有限（V100 上每个 SM 有 65536 个 32-bit 寄存器），
//       寄存器溢出（spilling）会导致数据被写入 local memory（实际在 DRAM）。
//
// 此外还演示了：
//   - Warp 执行模型：32 个线程组成一个 warp，以 SIMD 方式执行
//     （Single Instruction Multiple Data，单指令多数据）
//   - 基于资源约束的 thread block 调度
//   - 全局和共享内存上的原子操作
//
// GPU SM（Streaming Multiprocessor，流多处理器）资源限制（以 V100 为参考）：
//   - 每个 SM 最多驻留 2048 个线程（64 个 warp）
//   - 每个 SM 有约 96 KB 可编程 shared memory
//   - 每个 SM 有 65536 个 32-bit 寄存器
//   这些资源限制同时约束着能在一个 SM 上运行的 block 数量。
//
// 编译：g++ -std=c++17 -pthread lecture7_part3.cpp -o lecture7_part3
// 运行：./lecture7_part3

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <iomanip>
#include <algorithm>
#include <cstring>

// ============================================================================
// 模拟 GPU 架构参数（以 NVIDIA V100 为参考）
// ============================================================================

constexpr int WARP_SIZE          = 32;   // 每个 warp 包含 32 个线程，以 SIMD 方式执行
constexpr int MAX_WARPS_PER_SM   = 64;   // 每个 SM 最多 64 个 warp
constexpr int MAX_THREADS_PER_SM = MAX_WARPS_PER_SM * WARP_SIZE;  // 2048 线程/SM
constexpr int SHARED_MEM_PER_SM  = 128 * 1024;  // 128 KB（shared + L1 缓存，硬件可配置比例）
constexpr int REGISTERS_PER_SM   = 256 * 1024;  // 256 KB 总寄存器（65536 个 32-bit 寄存器）
constexpr int NUM_SM             = 4;   // 模拟 4 个 SM 的 GPU（小规模模拟）

// ============================================================================
// 模拟 GPU 全局内存（Global Memory / DRAM）
// 特点：容量大但延迟高，所有 SM 通过交叉开关（crossbar）共享访问
// ============================================================================

class GlobalMemory {
public:
    std::vector<int> data;

    GlobalMemory(size_t size) : data(size, 0) {}

    int read(size_t addr) {
        // 模拟 DRAM 延迟：记录访问次数（真实 GPU 中约 200~400 cycles）
        access_count++;
        return data[addr];
    }

    void write(size_t addr, int value) {
        access_count++;
        data[addr] = value;
    }

    void atomicAdd(size_t addr, int value) {
        // 原子操作比普通读写更昂贵（需要锁总线/缓存一致性协议）
        access_count += 2;
        std::lock_guard<std::mutex> lock(gmem_mutex);
        data[addr] += value;
    }

    long long getAccessCount() const { return access_count; }

private:
    std::mutex gmem_mutex;           // 模拟全局内存访问的互斥锁
    std::atomic<long long> access_count{0};  // 访问计数器
};

// ============================================================================
// 流多处理器（Streaming Multiprocessor, SM）模拟
// 每个 SM 拥有：
//   - 共享内存（per-block）：block 内线程共享的快速片上 SRAM
//   - 寄存器文件（per-thread）：每个 thread 独享的最快存储
//   - 执行单元（warp scheduler + CUDA cores）：按 warp 调度执行
//
// SM 的资源分配是一个关键约束：一个 SM 能同时驻留多少 block
// 取决于 block 需要的 shared memory、寄存器和线程数是否超出 SM 限制。
// ============================================================================

struct SM {
    int id;                      // SM 编号
    int sharedMemUsed  = 0;      // 当前已使用的 shared memory（字节）
    int registersUsed  = 0;      // 当前已使用的寄存器（字节）
    int activeWarps    = 0;      // 当前活跃的 warp 数量
    int activeThreads  = 0;      // 当前活跃的线程总数
    int maxSharedMem   = SHARED_MEM_PER_SM;   // 该 SM 的 shared memory 上限
    int maxRegisters   = REGISTERS_PER_SM;    // 该 SM 的寄存器上限
    int maxThreads     = MAX_THREADS_PER_SM;  // 该 SM 的线程数上限

    // 模拟的共享内存（per-SM，在不同 block 之间分配）
    std::vector<int> sharedMem;

    SM(int _id) : id(_id), sharedMem(SHARED_MEM_PER_SM / sizeof(int), 0) {}

    // 检查该 SM 是否有足够资源容纳一个新的 thread block
    // 需要同时满足三个约束：线程数、shared memory、寄存器数
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
        activeWarps    = (activeThreads + WARP_SIZE - 1) / WARP_SIZE;  // 上取整
    }

    void deallocateBlock(int blockThreads, int blockSharedBytes, int blockRegsPerThread) {
        activeThreads  -= blockThreads;
        sharedMemUsed  -= blockSharedBytes;
        registersUsed  -= blockThreads * blockRegsPerThread;
        activeWarps    = (activeThreads + WARP_SIZE - 1) / WARP_SIZE;
    }

    void printStatus() const {
        std::cout << "  SM[" << id << "]："
                  << activeThreads << "/" << maxThreads << " 线程，"
                  << activeWarps << " 个 warp，"
                  << sharedMemUsed << "/" << maxSharedMem << " 字节 shared，"
                  << registersUsed << "/" << maxRegisters << " 字节 regs\n";
    }
};

// ============================================================================
// GPU 工作调度器
// 根据资源可用性将 thread block 分配到各个 SM 上
// 这是 CUDA 运行时硬件调度逻辑的简化模拟
// ============================================================================

class GPUWorkScheduler {
public:
    std::vector<SM> sms;          // SM 集合
    GlobalMemory gmem;            // 全局内存

    GPUWorkScheduler(int numSMs, size_t globalMemSize)
        : gmem(globalMemSize)
    {
        for (int i = 0; i < numSMs; i++) {
            sms.emplace_back(i);
        }
    }

    // 将 thread block 调度到第一个有足够资源的 SM 上
    // 返回分配的 SM 编号，-1 表示当前无可用 SM（需要等待）
    int scheduleBlock(int blockIdx, int blockThreads,
                      int blockSharedBytes, int blockRegsPerThread)
    {
        for (auto& sm : sms) {
            if (sm.canFitBlock(blockThreads, blockSharedBytes, blockRegsPerThread)) {
                sm.allocateBlock(blockThreads, blockSharedBytes, blockRegsPerThread);
                return sm.id;
            }
        }
        return -1;  // 没有可用 SM — 必须等待资源释放
    }

    void completeBlock(int smId, int blockThreads,
                       int blockSharedBytes, int blockRegsPerThread)
    {
        sms[smId].deallocateBlock(blockThreads, blockSharedBytes, blockRegsPerThread);
    }

    void printStatus() const {
        std::cout << "GPU 状态：\n";
        for (const auto& sm : sms) {
            sm.printStatus();
        }
    }

    long long getGlobalAccessCount() const {
        return gmem.getAccessCount();
    }
};

// ============================================================================
// 使用全局内存上的原子操作计算直方图（Histogram）
// （第7讲示例：使用 atomicAdd 在 global memory 中对共享变量进行原子累加）
//
// 直方图计算是 GPU 上经典的竞争场景：多个线程可能同时更新同一个 bin。
// 解决方案是使用 atomicAdd 保证操作原子的（即不可分割的）完成，
// 不会出现读-改-写竞争导致的数据丢失。
// ============================================================================

void computeHistogram(GPUWorkScheduler& gpu,
                      const std::vector<int>& data,
                      int numBins)
{
    std::cout << "\n--- 直方图计算（Global Memory 上的原子操作）---\n";
    std::cout << "输入数据大小：" << data.size() << "，分桶数：" << numBins << "\n";

    // 在全局内存中分配直方图的 bin 数组
    int binBase = 1000;  // 直方图 bin 在全局内存中的起始地址偏移量

    constexpr int HIST_THREADS_PER_BLK = 64;   // 每个 block 64 个线程
    constexpr int HIST_SHARED_BYTES    = 0;    // 不需要共享内存
    constexpr int HIST_REGS_PER_THREAD = 4 * 4; // 约 4 个 int 寄存器（每个 4 字节）

    int numBlocks = (data.size() + HIST_THREADS_PER_BLK - 1) / HIST_THREADS_PER_BLK;

    std::vector<std::thread> blockThreads;

    for (int blk = 0; blk < numBlocks; blk++) {
        int smId = gpu.scheduleBlock(blk, HIST_THREADS_PER_BLK,
                                     HIST_SHARED_BYTES, HIST_REGS_PER_THREAD);
        if (smId < 0) {
            std::cout << "  Block " << blk << " 无法调度（资源已满）\n";
            continue;
        }

        int startIdx = blk * HIST_THREADS_PER_BLK;

        blockThreads.emplace_back([&gpu, &data, startIdx, binBase, numBins]() {
            for (int t = 0; t < HIST_THREADS_PER_BLK; t++) {
                int idx = startIdx + t;
                if (idx >= static_cast<int>(data.size())) break;

                // 计算该元素属于哪个 bin
                int bin = data[idx] % numBins;
                if (bin < 0) bin += numBins;
                // atomicAdd(&counts[bin], 1)
                // 原子累加保证多个线程同时更新同一 bin 时的正确性
                gpu.gmem.atomicAdd(binBase + bin, 1);
            }
        });

        gpu.completeBlock(smId, HIST_THREADS_PER_BLK,
                         HIST_SHARED_BYTES, HIST_REGS_PER_THREAD);
    }

    for (auto& t : blockThreads) t.join();

    // 输出直方图结果
    std::cout << "直方图结果：\n";
    for (int b = 0; b < numBins; b++) {
        std::cout << "  bin[" << b << "]：" << gpu.gmem.read(binBase + b) << "\n";
    }
    std::cout << "Global memory 访问次数：" << gpu.getGlobalAccessCount() << "\n";
}

// ============================================================================
// 演示基于 warp 的执行模型
// 32 个线程组成一个 warp；同一个 warp 中的所有线程执行同一条指令（SIMD）
//
// Warp 是 GPU 硬件调度的基本单位：
//   - 每个 warp 包含 32 个 thread（thread 0~31 称为 lane 0~31）
//   - Warp scheduler 每个时钟周期选择一个 warp 执行
//   - 同一 warp 内所有 lane 执行相同的指令（SIMD）
//   - NVIDIA 称其为 SIMT（Single Instruction Multiple Thread）
// ============================================================================

void simulateWarpExecution()
{
    std::cout << "\n--- Warp 执行模型模拟（SIMD 方式）---\n";

    constexpr int WARP_COUNT = 4;
    int perWarpData[WARP_SIZE];  // 每个 warp 中 32 个 lane 的数据（寄存器模拟）

    std::cout << "启动 " << WARP_COUNT * WARP_SIZE
              << " 个线程，共 " << WARP_COUNT << " 个 warp：\n";

    // 每个 warp 作为一组 32 个线程的集合被启动
    std::vector<std::thread> warpThreads;

    for (int w = 0; w < WARP_COUNT; w++) {
        // 在硬件中：warp selector 每个时钟周期选择一个 warp 执行
        // 每个 warp 有自己独立的寄存器集合（32 个 lane × 若干寄存器 = warp 寄存器堆）
        warpThreads.emplace_back([w, &perWarpData]() {
            // 初始化每个 lane（线程）的数据（存储在寄存器中）
            for (int lane = 0; lane < WARP_SIZE; lane++) {
                // 每个 "thread" 拥有自己的寄存器值
                // 不同 lane 的数据互不影响
                perWarpData[lane] = w * 100 + lane;
            }

            // SIMD 操作：所有 32 个线程执行相同的乘法和加法指令
            // 在真实 GPU 中，这 32 个操作同时完成（或分 2 个周期，取决于 ALU 数量）
            for (int lane = 0; lane < WARP_SIZE; lane++) {
                perWarpData[lane] = perWarpData[lane] * 2 + 1;
            }
        });
    }

    for (auto& t : warpThreads) t.join();

    std::cout << "每个 warp 对其数据执行了 32-way SIMD 乘加操作。\n";
    std::cout << "在真实 GPU 中：每个 sub-core 有 16 个 ALU，因此 32-thread warp\n";
    std::cout << "每条指令需要 2 个时钟周期来执行。\n";
}

// ============================================================================
// 演示 warp 内的分支发散（Branch Divergence）
// 当同一个 warp 内的线程走不同的分支时，执行会被序列化
//
// 由于 warp 内所有 lane 共享同一个程序计数器（PC），
// 因此它们必须执行同一条指令。当遇到条件分支时：
//   - 部分 lane 满足条件走 if 分支，其余 lane 被 mask（暂停）
//   - 然后满足 else 的 lane 执行 else 分支，if 的 lane 被 mask
//   - 结果：原本应并行的执行变成了串行，性能下降
//   - 极端情况：所有 32 个 lane 走不同分支 → 32x 性能退化
// ============================================================================

void simulateWarpDivergence()
{
    std::cout << "\n--- Warp 分支发散示例 ---\n";

    int results[WARP_SIZE];

    // 模拟带有条件分支的 kernel
    for (int lane = 0; lane < WARP_SIZE; lane++) {
        if (lane % 2 == 0) {
            // 偶数 lane 走这条路径（线程 0, 2, 4, ..., 30）
            results[lane] = lane * 10;
        } else {
            // 奇数 lane 走这条路径（线程 1, 3, 5, ..., 31）
            results[lane] = lane * 10 + 1000;
        }
    }

    std::cout << "包含分支发散的 32 线程 warp：\n";
    std::cout << "  偶数线程：results[lane] = lane * 10\n";
    std::cout << "  奇数线程：results[lane] = lane * 10 + 1000\n";
    std::cout << "样本结果：";
    for (int i = 0; i < 8; i++) std::cout << results[i] << " ";
    std::cout << "\n";

    std::cout << "在真实 GPU 中：偶数线程先执行（奇数被 mask），\n";
    std::cout << "然后奇数线程再执行（偶数被 mask）。\n";
    std::cout << "这种序列化导致该 warp 性能降低约 50%。\n";
}

// ============================================================================
// 内存延迟对比模拟
// 直观展示三种 GPU 内存层次的性能差距
// ============================================================================

void simulateMemoryLatency()
{
    std::cout << "\n--- 内存层次延迟对比 ---\n";

    constexpr int ITERATIONS = 1000;

    // 模拟各层级内存的访问延迟（以时钟周期计）
    // 寄存器文件（Register File）：    1 cycle    — 每个 thread 独享，带宽最高
    // 共享内存（Shared Memory/SRAM）：~20 cycles  — SM 片上，block 内共享
    // 全局内存（Global Memory/HBM）： ~300-500 cycles — 芯片外部 DRAM，延迟最高

    struct MemoryStats {
        const char* name;          // 内存类型名称
        int latencyCycles;         // 典型延迟（时钟周期）
        double bandwidthGBps;      // 典型带宽（GB/s，参考 V100）
    };

    MemoryStats memories[] = {
        {"寄存器文件 (RF)",   1,   8000.0},    // 寄存器：最快，带宽最高
        {"共享内存 (SRAM)",   20,  10000.0},   // 共享内存：比 global 快约 20x
        {"全局内存 (HBM)",    400, 900.0},     // 全局内存：最慢，容量最大
    };

    std::cout << std::left << std::setw(22) << "内存类型"
              << std::setw(14) << "延迟"
              << std::setw(16) << "带宽"
              << "相对速度\n";
    std::cout << std::string(60, '-') << "\n";

    for (const auto& mem : memories) {
        // 以全局内存为基准计算相对速度倍率
        double relativeSpeed = static_cast<double>(memories[2].latencyCycles)
                               / mem.latencyCycles;
        std::cout << std::left << std::setw(22) << mem.name
                  << "~" << std::setw(13) << (std::to_string(mem.latencyCycles) + " cycles")
                  << std::setw(15) << (std::to_string(static_cast<int>(mem.bandwidthGBps)) + " GB/s")
                  << std::fixed << std::setprecision(0) << relativeSpeed << "x 更快\n";
    }

    std::cout << "\n核心洞察：shared memory 比 global memory 快约 20 倍。\n";
    std::cout << "将数据协作加载到 shared memory 中，可以将 global memory\n";
    std::cout << "访问的开销分摊到多个线程上，从而显著提升性能。\n";
}

// ============================================================================
// 主函数
// ============================================================================

int main()
{
    std::cout << "==================================================\n";
    std::cout << "第7讲 第三部分：GPU 内存层次结构模拟\n";
    std::cout << "==================================================\n";

    // 初始化 GPU 模拟器
    GPUWorkScheduler gpu(NUM_SM, 4096);

    std::cout << "\nGPU 配置：\n";
    std::cout << "  流多处理器（SM）数量：" << NUM_SM << "\n";
    std::cout << "  每个 SM 最大线程数：" << MAX_THREADS_PER_SM << "\n";
    std::cout << "  Warp 大小：" << WARP_SIZE << "\n";
    std::cout << "  每个 SM 的 Shared Memory：" << SHARED_MEM_PER_SM / 1024 << " KB\n";
    std::cout << "  每个 SM 的寄存器：" << REGISTERS_PER_SM / 1024 << " KB\n";

    gpu.printStatus();

    // 1. 演示基于资源约束的 thread block 调度
    std::cout << "\n--- Thread Block 调度 ---\n";
    {
        // 模拟第7讲中的卷积 block：
        // 128 个线程，520 字节 shared memory，每线程 4 个 int 寄存器（16 字节）
        constexpr int CONV_THREADS   = 128;
        constexpr int CONV_SHARED    = 520;   // 130 个 float = 520 字节
        constexpr int CONV_REGS      = 4 * 4; // 4 个 int 寄存器，每个 4 字节

        int blocksScheduled = 0;
        for (int blk = 0; blk < 10; blk++) {
            int smId = gpu.scheduleBlock(blk, CONV_THREADS,
                                         CONV_SHARED, CONV_REGS);
            if (smId >= 0) {
                std::cout << "  Block " << blk << " → SM[" << smId << "]\n";
                blocksScheduled++;
            } else {
                std::cout << "  Block " << blk << " → 无可用资源\n";
            }
        }
        std::cout << "  总共调度了 " << blocksScheduled << " 个 block\n";
        std::cout << "  （每个 block：" << CONV_THREADS << " 个线程，"
                  << CONV_SHARED << "B shared，"
                  << CONV_REGS << "B regs/线程）\n";
    }
    gpu.printStatus();

    // 2. 使用原子操作的直方图计算
    std::vector<int> testData(200);
    for (size_t i = 0; i < testData.size(); i++) {
        testData[i] = static_cast<int>(i);
    }
    computeHistogram(gpu, testData, 10);

    // 3. Warp 执行模型
    simulateWarpExecution();

    // 4. Warp 分支发散
    simulateWarpDivergence();

    // 5. 内存延迟对比
    simulateMemoryLatency();

    std::cout << "\n==================================================\n";
    std::cout << "演示的核心概念：\n";
    std::cout << "  - 三种 GPU 内存类型：global、shared、registers\n";
    std::cout << "  - 基于资源约束的 thread block 调度机制\n";
    std::cout << "  - 全局内存上的原子操作（atomicAdd）\n";
    std::cout << "  - 基于 warp 的 SIMD 执行模型（32 个线程）\n";
    std::cout << "  - 分支发散（branch divergence）对性能的影响\n";
    std::cout << "  - 内存层次延迟对比分析\n";
    std::cout << "==================================================\n";

    return 0;
}
