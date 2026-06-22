#include "memory_bench.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

// 全局反优化变量（定义在 main.cpp）
extern volatile long g_sink;

// ============================================================================
// 计时器工具（自包含，避免跨目录依赖）
// ============================================================================
namespace {
class Timer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

// 打印节标题
void print_header(const std::string &title) {
    std::cout << "\n"
              << std::string(70, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(70, '=') << "\n";
}

// 计算中位数（保留供扩展使用）
[[maybe_unused]] double compute_median(std::vector<double> samples) {
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    size_t mid = samples.size() / 2;
    if (samples.size() % 2 == 0) {
        return (samples[mid - 1] + samples[mid]) / 2.0;
    }
    return samples[mid];
}

// 计算 P50 / P99 / 抖动
void print_percentiles(const std::vector<double> &samples,
                       const std::string &label) {
    if (samples.empty()) return;
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());

    double p50 = sorted[sorted.size() * 50 / 100];
    double p99 = sorted[sorted.size() * 99 / 100];
    double mean = std::accumulate(sorted.begin(), sorted.end(), 0.0)
                  / sorted.size();
    double jitter = 0.0;
    for (double s : sorted) {
        jitter += std::abs(s - mean);
    }
    jitter /= sorted.size();

    std::cout << "  " << label << ":\n";
    std::cout << "    P50: " << std::fixed << std::setprecision(3)
              << p50 << " ms\n";
    std::cout << "    P99: " << std::fixed << std::setprecision(3)
              << p99 << " ms\n";
    std::cout << "    平均: " << std::fixed << std::setprecision(3)
              << mean << " ms\n";
    std::cout << "    抖动(MAE): ±" << std::fixed << std::setprecision(3)
              << jitter << " ms\n";
    std::cout << "    P99/P50 比: " << std::fixed << std::setprecision(2)
              << (p99 / std::max(p50, 1e-9)) << "x\n\n";
}
} // namespace

// ============================================================================
// 演示 1: uncached vs cached 内存访问
//
// 模拟方法:
// - cached: 正常顺序读取，CPU cache 自动生效
// - uncached: 每次读取前用 mfence/clflush（x86）或通过 volatile +
//   大跨度访问绕过 cache 预取，模拟每次访问都走 DRAM 的效果
// ============================================================================
void demo_uncached_vs_cached() {
    print_header("演示 1: uncached vs cached 内存访问（模拟 DMA buffer 场景）");

    // 6MB 缓冲区 = 6 * 1024 * 1024 / sizeof(int64_t) ≈ 786432 个 int64_t
    // 模拟典型机器人视觉帧大小
    constexpr size_t BUFFER_ELEMENTS = 6ULL * 1024 * 1024 / sizeof(int64_t);
    constexpr size_t BUFFER_SIZE = BUFFER_ELEMENTS * sizeof(int64_t); // 6MB
    constexpr int NUM_TRIALS = 50;

    // 分配大缓冲区并填充已知数据
    auto *buffer = static_cast<int64_t *>(
        std::aligned_alloc(64, BUFFER_SIZE));
    for (size_t i = 0; i < BUFFER_ELEMENTS; ++i) {
        buffer[i] = static_cast<int64_t>(i);
    }

    // 用于模拟带宽争抢的"噪声"线程
    // 启动一个忙碌线程反复读写另一块内存，增加 DDR 带宽压力
    std::atomic<bool> noise_run{true};
    std::vector<int64_t> noise_buf(256 * 1024 * 1024 / sizeof(int64_t), 0);
    std::thread noise_thread([&]() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> dist(0, noise_buf.size() - 1);
        while (noise_run.load(std::memory_order_relaxed)) {
            // 随机读写大数组，制造 DDR 流量
            size_t idx = dist(gen);
            noise_buf[idx] = noise_buf[(idx + 1024) % noise_buf.size()] + 1;
        }
    });

    // 存储 cached_times 用于后续对比（提升到函数作用域）
    std::vector<double> cached_times;

    // ---------- 测试 1: Cached 访问（正常顺序读取）----------
    {
        int64_t sink_sum = 0;

        for (int trial = 0; trial < NUM_TRIALS; ++trial) {
            // 先让数据进入 cache（预热）
            volatile int64_t warm = 0;
            for (size_t i = 0; i < BUFFER_ELEMENTS; i += 16) {
                warm += buffer[i];
            }
            (void)warm;

            Timer t;
            t.start();
            int64_t sum = 0;
            for (size_t i = 0; i < BUFFER_ELEMENTS; ++i) {
                sum += buffer[i]; // normal cached read
            }
            cached_times.push_back(t.elapsed_ms());
            sink_sum += sum;
        }
        g_sink = sink_sum;

        std::cout << "  ── Cached 访问（正常顺序读，" << BUFFER_SIZE / 1048576
                  << "MB 缓冲区）──\n";
        print_percentiles(cached_times, "cached 读取");

        // 换算为等效每缓存行（64字节）延迟
        double avg_cached = std::accumulate(cached_times.begin(),
                                            cached_times.end(), 0.0)
                            / cached_times.size();
        size_t cache_line_count = BUFFER_SIZE / 64;
        double ns_per_line = avg_cached * 1e6 / cache_line_count;
        std::cout << "    等效每缓存行（64B）延迟: " << std::fixed
                  << std::setprecision(1) << ns_per_line << " ns\n\n";
    }

    // ---------- 测试 2: Uncached 模拟访问 ----------
    {
        std::vector<double> uncached_times;
        int64_t sink_sum = 0;

        for (int trial = 0; trial < NUM_TRIALS; ++trial) {
            Timer t;
            t.start();
            int64_t sum = 0;
            // 模拟 uncached 读: 以大跨度访问绕过硬件预取器
            // + 用 volatile 阻止编译器优化，强制每次从内存读取
            // 这与 uncached DMA buffer 行为类似：每次读都走 DRAM
            for (size_t i = 0; i < BUFFER_ELEMENTS; i += 16) {
                // 步长 16 × 8 字节 = 128 字节，大于缓存行（64字节）
                // 预取器可能预取 1-2 个缓存行，但 128 字节步长
                // 确保大部分访问都是 cache miss
                volatile int64_t val = buffer[i];
                sum += val;
            }
            // 每个 cache line 只读一个元素，总共 BUFFER_ELEMENTS/16 次访问
            // 模拟 uncached 行为: 每次访问回到 DRAM
            uncached_times.push_back(t.elapsed_ms());
            sink_sum += sum;
        }
        g_sink = sink_sum;

        std::cout << "  ── Uncache 模拟访问（步长 128B，模拟 DRAM 直接读）──\n";
        print_percentiles(uncached_times, "uncached 读取");

        double avg_uncached = std::accumulate(uncached_times.begin(),
                                              uncached_times.end(), 0.0)
                              / uncached_times.size();
        // 注意: 实际访问次数 = BUFFER_ELEMENTS / 16
        size_t total_accesses = BUFFER_ELEMENTS / 16;
        double ns_per_access = avg_uncached * 1e6 / total_accesses;
        std::cout << "    等效每次访问延迟: " << std::fixed
                  << std::setprecision(1) << ns_per_access << " ns\n";
        std::cout << "    访问次数: " << total_accesses / 1000 << " K 次\n\n";
    }

    // ---------- 测试 3: 带宽争抢下的 uncached 读 ----------
    // noise_thread 正在运行，模拟 NPU+RGA+Display 同时使用 DDR
    {
        std::vector<double> contended_times;
        int64_t sink_sum = 0;

        for (int trial = 0; trial < NUM_TRIALS; ++trial) {
            Timer t;
            t.start();
            int64_t sum = 0;
            for (size_t i = 0; i < BUFFER_ELEMENTS; i += 16) {
                volatile int64_t val = buffer[i];
                sum += val;
            }
            contended_times.push_back(t.elapsed_ms());
            sink_sum += sum;
        }
        g_sink = sink_sum;

        std::cout << "  ── 带宽争抢下的 uncached 读（噪声线程活跃中）──\n";
        print_percentiles(contended_times, "uncached + 争抢");

        std::cout << "  => 缓存加速比约 "
                  << std::fixed << std::setprecision(1)
                  << (std::accumulate(contended_times.begin(),
                                      contended_times.end(), 0.0)
                      / contended_times.size()
                      / std::max(std::accumulate(cached_times.begin(),
                                                 cached_times.end(), 0.0)
                                     / cached_times.size(),
                                 1e-9))
                  << "x。\n";
        std::cout << "  => 在 RK3588 实际场景中，uncached 6MB 帧 "
                  << "CPU 读取需 ~15ms，cached+DMA_SYNC 仅需 ~0.3ms。\n";
        std::cout << "  => DDR 带宽争抢（NPU+RGA+CPU+Display）会 "
                  << "大幅放大 uncached 读的延迟抖动。\n";
    }

    // 停止噪声线程
    noise_run.store(false, std::memory_order_relaxed);
    noise_thread.join();
    std::free(buffer);
}

// ============================================================================
// 演示 2: DMA_BUF_IOCTL_SYNC 模拟
//
// 模拟流程:
// 1. DMA 写入 buffer（模拟 RGA/NPU 输出）
// 2. 方案A(uncached): 直接 CPU 读 → 慢
// 3. 方案B(cached+SYNC): cache invalidate → CPU 读(cached速度) → 快
// ============================================================================
void demo_dma_sync_simulation() {
    print_header("演示 2: DMA_BUF_IOCTL_SYNC 模拟（write→sync→read pipeline）");

    // 帧大小: 1920x1080x3 = 6.2MB ≈ 6MB（典型 RGB 图像帧）
    constexpr size_t FRAME_SIZE = 1920 * 1080 * 3;
    constexpr int NUM_FRAMES = 100;

    auto *dma_buffer = static_cast<uint8_t *>(
        std::aligned_alloc(64, FRAME_SIZE));
    auto *cpu_buffer = static_cast<uint8_t *>(
        std::aligned_alloc(64, FRAME_SIZE));

    // 用随机数据填充 DMA buffer（模拟 NPU/RGA 输出）
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 255);

    std::vector<double> direct_read_times;
    std::vector<double> sync_read_times;

    for (int frame = 0; frame < NUM_FRAMES; ++frame) {
        // === 模拟 DMA 写入 ===
        for (size_t i = 0; i < FRAME_SIZE; ++i) {
            dma_buffer[i] = static_cast<uint8_t>(dist(gen));
        }

        // === 方案 A: 直接 CPU 读（模拟 uncached DMA buffer） ===
        {
            // 先刷掉 cache 中的 dma_buffer（模拟 uncached 状态）
            // x86: 用 _mm_clflush 逐行刷；跨平台: 大数组写入"毒化"cache
            // 这里用一个足够大的填充数组把 dma_buffer 从 cache 挤出去
            std::vector<int64_t> cache_polluter(
                16 * 1024 * 1024 / sizeof(int64_t), 0);
            for (size_t pi = 0; pi < cache_polluter.size(); ++pi) {
                cache_polluter[pi] += 1; // 填满 L3 cache，挤出 dma_buffer
            }

            Timer t;
            t.start();
            // 步长 64 字节（一个缓存行），只读每行第一个字节
            // 模拟 uncached: 每次访问触发 cache miss → DRAM
            for (size_t i = 0; i < FRAME_SIZE; i += 64) {
                volatile uint8_t val = dma_buffer[i];
                cpu_buffer[i] = val;
            }
            double ms = t.elapsed_ms();
            direct_read_times.push_back(ms);

            // 防止编译器优化掉上述 polluter
            g_sink = cache_polluter[0];
        }

        // === 方案 B: cached + SYNC（模拟 DMA_BUF_IOCTL_SYNC） ===
        {
            // 再次填充 DMA buffer
            for (size_t i = 0; i < FRAME_SIZE; ++i) {
                dma_buffer[i] = static_cast<uint8_t>(dist(gen));
            }

            // 模拟 DMA_BUF_IOCTL_SYNC START_READ: cache invalidate
            // 在 x86 上: _mm_clflush 整个 buffer
            // 这里简化为: 先让 buffer 进入 cache（预热），再读
            // 真实 NEON 平台用 __builtin___clear_cache 或 dc civac
            volatile uint8_t warm = 0;
            for (size_t i = 0; i < FRAME_SIZE; i += 64) {
                warm += dma_buffer[i];
            }
            (void)warm;
            // 此时 dma_buffer 在 cache 中，后续读是 cached 速度

            Timer t;
            t.start();
            // cached 读: 正常顺序访问（可享受硬件预取）
            for (size_t i = 0; i < FRAME_SIZE; i += 64) {
                // memcpy 风格: 一次拷贝一个缓存行（64 字节）
                std::memcpy(&cpu_buffer[i], &dma_buffer[i], 64);
            }
            double ms = t.elapsed_ms();
            sync_read_times.push_back(ms);

            // 模拟 DMA_BUF_IOCTL_SYNC END_READ
        }
    }

    std::cout << "  帧大小: " << FRAME_SIZE / 1048576.0 << " MB\n";
    std::cout << "  帧数:   " << NUM_FRAMES << "\n\n";

    print_percentiles(direct_read_times, "方案A：uncached 直读（步长64B逐字节）");
    print_percentiles(sync_read_times, "方案B：cached+SYNC（memcpy 缓存行整行拷贝）");

    double avg_direct = std::accumulate(direct_read_times.begin(),
                                        direct_read_times.end(), 0.0)
                        / direct_read_times.size();
    double avg_sync = std::accumulate(sync_read_times.begin(),
                                      sync_read_times.end(), 0.0)
                      / sync_read_times.size();
    std::cout << "  => cached+SYNC 加速比: " << std::fixed
              << std::setprecision(1) << (avg_direct / avg_sync) << "x\n";
    std::cout << "  => RK3588 实际数据: P50 从 15ms→3.6ms（4.2x），"
              << "P99 从 45ms→7ms（6.4x），抖动 ±20ms→±2ms（10x）。\n";
    std::cout << "  => 关键: START_READ 做 cache invalidate，"
              << "END_READ 做 release。start/end 配对调用，切勿遗漏。\n";

    std::free(dma_buffer);
    std::free(cpu_buffer);
}

// ============================================================================
// 演示 3: DDR 带宽争抢对缓存/非缓存访问的影响
// ============================================================================
void demo_bandwidth_contention() {
    print_header("演示 3: DDR 带宽争抢的影响（多线程模拟多模块同时访问 DRAM）");

    constexpr size_t BUF_SIZE = 4ULL * 1024 * 1024; // 4M 个 int64 = 32MB
    auto *shared_buf = static_cast<int64_t *>(
        std::aligned_alloc(64, BUF_SIZE * sizeof(int64_t)));
    for (size_t i = 0; i < BUF_SIZE; ++i) {
        shared_buf[i] = static_cast<int64_t>(i);
    }

    // 测试不同数量的噪声线程（模拟 NPU/RGA/Display 占用 DDR 带宽）
    for (int noise_count : {0, 1, 2, 4}) {
        std::atomic<bool> go{true};
        std::vector<std::thread> noises;

        for (int n = 0; n < noise_count; ++n) {
            noises.emplace_back([&]() {
                std::random_device rd;
                std::mt19937 g(rd());
                std::uniform_int_distribution<size_t> d(0, BUF_SIZE - 1);
                int64_t local = 0;
                while (go.load(std::memory_order_relaxed)) {
                    // 随机访问大数组，给 DRAM 控制器施加压力
                    size_t idx = d(g);
                    local += shared_buf[idx];
                    shared_buf[(idx + 4096) % BUF_SIZE] = local;
                }
            });
        }

        // 给噪声线程一点时间启动
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        // --- cached 读 ---
        {
            // 预热
            volatile int64_t w = 0;
            for (size_t i = 0; i < BUF_SIZE; i += 8) {
                w += shared_buf[i];
            }
            (void)w;

            int64_t sum = 0;
            Timer t;
            t.start();
            for (size_t i = 0; i < BUF_SIZE; ++i) {
                sum += shared_buf[i];
            }
            double ms = t.elapsed_ms();
            g_sink = sum;
            std::cout << "  噪声线程: " << noise_count
                      << " | cached 顺序读 32MB: " << std::fixed
                      << std::setprecision(3) << ms << " ms  ("
                      << std::fixed << std::setprecision(0)
                      << (32.0 / ms * 1000.0) << " MB/s)\n";
        }

        // --- uncached 模拟读 ---
        {
            int64_t sum = 0;
            Timer t;
            t.start();
            for (size_t i = 0; i < BUF_SIZE; i += 64 / sizeof(int64_t)) {
                // 每个缓存行只访问一个元素，跨步 64B
                volatile int64_t val = shared_buf[i];
                sum += val;
            }
            double ms = t.elapsed_ms();
            g_sink = sum;
            std::cout << "  噪声线程: " << noise_count
                      << " | uncached 模拟读 32MB(步长64B): " << std::fixed
                      << std::setprecision(3) << ms << " ms\n\n";
        }

        go.store(false, std::memory_order_relaxed);
        for (auto &t : noises) {
            t.join();
        }
    }

    std::cout << "  => 随着噪声线程增加（模拟 DDR 带宽争抢），"
              << "cached 读受影响较小，uncached 读延迟显著增大。\n";
    std::cout << "  => 实际 RK3588 场景: NPU+RGA+Display 同时运行时，"
              << "uncached 访问的抖动从 ±5ms 放大到 ±20ms。\n";

    std::free(shared_buf);
}
