// lecture2_part3.cpp - 硬件多线程与延迟隐藏
// =============================================================================
// CS149 第2讲核心概念：
//   - 内存访问延迟（DRAM：约 248 个周期）会导致处理器停顿（stall）
//     当 CPU 需要从主存加载数据时，在数据到达之前该指令流无法继续。
//     这些停顿周期中执行单元处于空闲状态 → 浪费计算能力。
//
//   - 多线程（Multi-threading）：在同一核心上交错处理多个线程
//     以隐藏停顿（当一个线程停顿时，切换到另一个线程工作）。
//     核心思想：用多个线程的可用工作来填充因内存等待产生的"气泡"。
//
//   - 交错多线程（Interleaved Multi-threading）：
//     每个时钟周期，核心选择一个就绪的线程执行一条指令。
//     使用轮询（round-robin）或优先级策略选择下一个线程。
//     优点是实现简单，缺点是单线程性能降低。
//
//   - 同时多线程（Simultaneous Multi-threading，SMT）：
//     每个时钟周期可执行多个线程的指令（如 Intel 超线程 HT）。
//     超标量发射宽度被多个线程共享 → 更好的执行单元利用率。
//
//   - 权衡：更多线程 = 更好的延迟隐藏能力，但每线程的存储资源更少
//     寄存器文件、缓存等需要在更多线程间共享。
//     GPU 的设计哲学：许多小上下文（小寄存器文件）→ 极致延迟隐藏。
//     CPU 的设计哲学：少量大上下文（大寄存器文件）→ 更好的单线程性能。
//
//   - 吞吐量计算（Throughput Computing）：
//     可能通过增加每线程的延迟来提高整体系统吞吐量。
//     这是一个反直觉的概念——单个线程可能变慢，
//     但系统整体完成的工作更多。
//
//   - 每次内存访问的算术运算越多 → 所需线程数越少
//     因为每个内存停顿之间有更多的算术指令可以填充周期。
//     这就是"算术强度"（Arithmetic Intensity）的概念。
//
//   - NVIDIA V100：80 个 SM，每个 SM 64 个 warp，32-wide SIMD
//     → 163,840 个并发数据项用于最大化延迟隐藏
//     这是极端的吞吐量导向设计：用海量线程淹没内存延迟。
//
// 编译: g++ -std=c++17 -O2 -pthread lecture2_part3.cpp -o lecture2_part3
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
// 模拟多线程核心的执行过程
//
// 模型说明：
//   - 指令按周期执行，内存加载具有延迟（模拟缓存未命中）
//   - 每个线程周期性地执行一定数量的算术指令，然后触发一次内存加载
//   - 内存加载期间，该线程停顿（stall），等待数据返回
//   - 调度器在就绪线程间轮询选择下一个执行的线程
//
// 这不是真实的 CPU 模拟，而是一个概念演示：
// 展示线程数、算术强度和内存延迟之间的关系。
// ---------------------------------------------------------------------------
class MultiThreadedCore {
public:
    struct ThreadState {
        int id;
        int pc = 0;                          // 程序计数器（下一条指令）
        bool stalled = false;
        int stall_remaining = 0;
        int instructions_completed = 0;
    };

    struct Config {
        int num_threads;
        int memory_latency;    // 内存加载所需的周期数
        int math_per_load;     // 每次内存加载前的算术指令数
        bool verbose = false;  // 是否打印详细调度信息
    };

    MultiThreadedCore(const Config& cfg) : config_(cfg) {
        for (int i = 0; i < cfg.num_threads; i++) {
            threads_.push_back({i, 0, false, 0, 0});
        }
    }

    // 运行模拟指定的周期数
    void run(int max_cycles) {
        int cycle = 0;
        while (cycle < max_cycles) {
            // 检查是否有停顿的线程已就绪
            for (auto& t : threads_) {
                if (t.stalled) {
                    t.stall_remaining--;
                    if (t.stall_remaining <= 0) {
                        t.stalled = false;
                    }
                }
            }

            // 选择一个线程执行（在就绪线程中轮询）
            bool any_progress = false;
            int start_search = round_robin_next_;
            int checked = 0;

            while (checked < config_.num_threads) {
                int idx = (start_search + checked) % config_.num_threads;
                auto& t = threads_[idx];

                if (!t.stalled) {
                    t.instructions_completed++;

                    // 检查本指令是否为加载指令
                    // 每 math_per_load 条算术指令后，执行一条加载指令
                    if (t.instructions_completed % (config_.math_per_load + 1) == 0 &&
                        t.instructions_completed > 0) {
                        // 内存加载：后续指令停顿
                        t.stalled = true;
                        t.stall_remaining = config_.memory_latency;
                        
                        if (config_.verbose) {
                            std::cout << "    [周期 " << cycle << "] T" << t.id 
                                      << "：LOAD（停顿 " << t.stall_remaining 
                                      << " 周期），已完成=" << t.instructions_completed 
                                      << std::endl;
                        }
                    } else {
                        if (config_.verbose) {
                            std::cout << "    [周期 " << cycle << "] T" << t.id 
                                      << "：MATH，已完成=" << t.instructions_completed 
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
                std::cout << "    [周期 " << cycle << "] STALL（所有线程都在等待）\n";
            }

            // 跟踪忙碌周期（有线程执行的周期）
            if (any_progress) busy_cycles_++;

            cycle++;
        }
        total_cycles_ = cycle;
    }

    void print_stats() const {
        std::cout << "    " << std::left << std::setw(16) << "线程数"
                  << std::setw(14) << "算术/加载"
                  << std::setw(14) << "内存延迟"
                  << std::setw(14) << "利用率"
                  << std::setw(16) << "总指令数" 
                  << std::setw(16) << "每线程指令数" << std::endl;
        std::cout << "    " << std::string(90, '-') << std::endl;

        int total_instr = 0;
        for (const auto& t : threads_) {
            total_instr += t.instructions_completed;
        }
        double util = static_cast<double>(busy_cycles_) / total_cycles_ * 100.0;

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
// 理论：需要多少线程才能达到 100% 利用率？
// 公式：threads_needed = ceil(1 + latency / math_per_load)
//
// 推导：
//   每个线程每 (math_per_load + 1) 条指令产生一次 latency 周期的停顿。
//   在停顿期间，需要有足够的其他线程的算术指令来填充每个周期。
//   所需线程数 = 1（原线程）+ ceil(latency / math_per_load)（填充线程）
//
// 直觉：
//   - 延迟越大 → 需要更多线程
//   - 算术强度越高 → 需要更少线程（因为有更多本地工作做）
// ---------------------------------------------------------------------------
int theoretical_threads_needed(int memory_latency, int math_per_load) {
    // 在内存延迟期间，我们需要至少 latency/math_per_load 个
    // 其他线程来填充每个周期。再加 1 是原线程。
    return static_cast<int>(std::ceil(1.0 + static_cast<double>(memory_latency) / math_per_load));
}

// ---------------------------------------------------------------------------
// GPU 风格：极致多线程
// NVIDIA V100：80 个 SM × 64 warp × 32 线程/warp = 163,840 个并发线程
//
// GPU 的设计哲学：
//   - 不追求低延迟，而是追求高吞吐量
//   - 通过海量并发线程来隐藏内存延迟
//   - 每个线程的寄存器文件很小（但线程多 → 总能找到就绪的线程）
//   - 没有缓存层次结构那么深 → 依靠线程切换而非缓存命中来隐藏延迟
// ---------------------------------------------------------------------------
void demo_gpu_style_multithreading() {
    std::cout << "[2] GPU 风格的极致多线程（V100）\n" << std::endl;

    std::cout << "    NVIDIA V100 流式多处理器（SM）：\n";
    std::cout << "    ┌──────────────────────────────────────────────────┐\n";
    std::cout << "    │ 每个 SM 有 64 个 warp 执行上下文                 │\n";
    std::cout << "    │ 每个 warp = 32 个线程（SIMD 宽度 = 32）          │\n";
    std::cout << "    │ 64 × 32 = 每个 SM 有 2048 个并发数据项           │\n";
    std::cout << "    │ V100 上有 80 个 SM                               │\n";
    std::cout << "    │ 总计：80 × 2048 = 163,840 个并发数据项           │\n";
    std::cout << "    └──────────────────────────────────────────────────┘\n" << std::endl;

    // 模拟：1 个 SM，64 个 warp，SIMD 宽度 32
    // 简化：在我们的模拟中每个 warp 独立执行
    const int WARPS = 64;
    const int SIMD_WIDTH = 32;
    const int mem_lat = 200;  // GPU 内存延迟（周期数）
    const int math_per_load = 10;

    int needed = theoretical_threads_needed(mem_lat, math_per_load);
    std::cout << "    在 " << mem_lat << " 周期内存延迟和 " 
              << math_per_load << " 次算术/加载的条件下：\n";
    std::cout << "    达到 100% 利用率所需线程数：" << needed << "\n";
    std::cout << "    可用 warp 数：" << WARPS << "（远超所需）\n";
    std::cout << "    → GPU 可以隐藏巨大的内存延迟\n" << std::endl;
}

// ---------------------------------------------------------------------------
// CPU 缓存层次结构延迟参考
//
// 这些数字来源于实际的 Kaby Lake CPU 测量数据：
//   L1：4 周期——最快，容量最小
//   L2：12 周期——比 L1 大 8 倍，慢 3 倍
//   L3：38 周期——共享的最后一级缓存，慢约 10 倍
//   DRAM：248 周期——没有缓存命中时的代价，慢约 60 倍
// ---------------------------------------------------------------------------
void demo_cache_latency() {
    std::cout << "[3] 内存延迟背景\n" << std::endl;
    std::cout << "    " << std::left << std::setw(20) << "缓存层级" 
              << std::setw(14) << "延迟"
              << std::setw(16) << "典型大小" << std::endl;
    std::cout << "    " << std::string(50, '-') << std::endl;
    std::cout << "    " << std::setw(20) << "L1 缓存" 
              << std::setw(14) << "~4 周期"
              << std::setw(16) << "32 KB" << std::endl;
    std::cout << "    " << std::setw(20) << "L2 缓存" 
              << std::setw(14) << "~12 周期"
              << std::setw(16) << "256 KB" << std::endl;
    std::cout << "    " << std::setw(20) << "L3 缓存" 
              << std::setw(14) << "~38 周期"
              << std::setw(16) << "8-20 MB" << std::endl;
    std::cout << "    " << std::setw(20) << "DRAM" 
              << std::setw(14) << "~248 周期"
              << std::setw(16) << "数 GB" << std::endl;
    std::cout << "    " << std::setw(20) << "GPU HBM2（V100）" 
              << std::setw(14) << "~350-500 c"
              << std::setw(16) << "16 GB" << std::endl;
    std::cout << "\n    在 4 GHz 下，248 周期 = 62 ns（DRAM 访问延迟）\n";
    std::cout << "    在 1.6 GHz（V100）下，350 周期 = 219 ns（HBM2 延迟）\n" << std::endl;
}

// =============================================================================
int main() {
    std::cout << "=== CS149 第2讲：硬件多线程与延迟隐藏 ===\n\n";

    // ---- 第一部分：通过多线程隐藏延迟 ----
    std::cout << "[1] 通过多线程隐藏延迟\n" << std::endl;
    std::cout << "    场景：3 条算术指令，然后 1 次加载（12 周期延迟）\n" << std::endl;

    std::cout << "    " << std::left << std::setw(16) << "线程数"
              << std::setw(14) << "算术/加载"
              << std::setw(14) << "内存延迟"
              << std::setw(14) << "利用率"
              << std::setw(16) << "总指令数"
              << std::setw(16) << "每线程指令数" << std::endl;
    std::cout << "    " << std::string(90, '-') << std::endl;

    // 模拟 1、2、3、4、5 个线程（与课程幻灯片一致）
    for (int num_threads : {1, 2, 3, 4, 5}) {
        MultiThreadedCore::Config cfg;
        cfg.num_threads = num_threads;
        cfg.memory_latency = 12;
        cfg.math_per_load = 3;
        cfg.verbose = false;

        MultiThreadedCore core(cfg);
        core.run(35); // 运行 35 个周期（与课程幻灯片的时间线匹配）
        core.print_stats();
    }

    int needed = theoretical_threads_needed(12, 3);
    std::cout << "\n    理论达到 100% 利用率所需线程数：" << needed << "\n";
    std::cout << "    公式：ceil(1 + latency / math_per_load)\n" << std::endl;

    // ---- 第二部分：不同算术强度的影响 ----
    std::cout << "[1b] 算术强度对线程需求的影响\n" << std::endl;

    // 每次加载前执行更多算术 → 所需线程数减少
    // 因为每个线程停顿期间能产生更多有用的算术周期
    std::cout << "    " << std::left << std::setw(16) << "算术/加载"
              << std::setw(20) << "所需线程数"
              << std::setw(20) << "100% 时利用率" << std::endl;
    std::cout << "    " << std::string(56, '-') << std::endl;

    for (int mpl : {1, 3, 6, 12}) {
        int needed_t = theoretical_threads_needed(12, mpl);
        std::cout << "    " << std::setw(16) << mpl
                  << std::setw(20) << needed_t;
        
        // 用模拟验证理论结果
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

    // ---- 第三部分：GPU 极致多线程 ----
    demo_gpu_style_multithreading();

    // ---- 第四部分：缓存延迟参考 ----
    demo_cache_latency();

    // ---- 第五部分：上下文存储的权衡 ----
    std::cout << "[4] 执行上下文存储的权衡\n" << std::endl;
    std::cout << "    ┌──────────────────────────────────────────────────────┐\n";
    std::cout << "    │ 许多小上下文：                                        │\n";
    std::cout << "    │   + 优秀的延迟隐藏（有很多线程可切换）               │\n";
    std::cout << "    │   - 每线程工作集有限（寄存器文件小）                  │\n";
    std::cout << "    │   - 对缓存压力更大                                   │\n";
    std::cout << "    ├──────────────────────────────────────────────────────┤\n";
    std::cout << "    │ 少量大上下文：                                        │\n";
    std::cout << "    │   + 每线程工作集较大                                  │\n";
    std::cout << "    │   + 每线程缓存局部性更好                              │\n";
    std::cout << "    │   - 延迟隐藏能力较弱                                  │\n";
    std::cout << "    └──────────────────────────────────────────────────────┘\n" << std::endl;

    // ---- 第六部分：核心要点 ----
    std::cout << "[5] 第2讲核心要点（多线程部分）\n" << std::endl;
    std::cout << "    - 内存访问延迟（数百周期）导致处理器停顿\n";
    std::cout << "    - 多线程通过执行其他线程的指令来隐藏停顿\n";
    std::cout << "    - 交错多线程：每周期 1 个线程（轮询调度）\n";
    std::cout << "    - 同时多线程：每周期多个线程（SMT，如 Intel HT）\n";
    std::cout << "    - 每次内存访问的算术运算越多 → 所需线程数越少\n";
    std::cout << "    - GPU：极致多线程（数千个并发上下文）\n";
    std::cout << "    - 吞吐量权衡：单个线程可能更慢，但整体吞吐量更高\n";
    std::cout << "    - 应用需求：足够的可并行工作 + 足够的算术强度\n";

    return 0;
}
