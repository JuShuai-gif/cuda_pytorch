#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <random>

// ============================================================================
//  Jitter 来源测试套件 — 6 类测试，逐一暴露不同的延迟不稳定来源
//  编译：g++ -std=c++17 -O2 -pthread -o test_jitter test_jitter.cpp
//
//  测试清单：
//    Test 1 — OS 调度抖动（CFS 时间片抢占）
//    Test 2 — 缓存 / TLB 未命中
//    Test 3 — 内存带宽竞争
//    Test 4 — 中断（IRQ）抢占检测
//    Test 5 — CPU 频率调节（DVFS / 热降频）
//    Test 6 — malloc 动态内存分配抖动
// ============================================================================

using Clock = std::chrono::high_resolution_clock;
using ns = std::chrono::nanoseconds;

static int64_t now_ns() {
    return std::chrono::duration_cast<ns>(Clock::now().time_since_epoch()).count();
}

// ---- Statistics helpers ----

struct Stats {
    double mean, stddev, p50, p99, min_val, max_val;
};

static Stats compute_stats(std::vector<int64_t> &samples) {
    std::sort(samples.begin(), samples.end());
    size_t n = samples.size();
    double sum = 0;
    for (auto v : samples) sum += v;
    double m = sum / n;

    double sq = 0;
    for (auto v : samples) sq += (v - m) * (v - m);
    double sd = std::sqrt(sq / n);

    auto pct = [&](int p) -> int64_t {
        size_t idx = static_cast<size_t>(std::ceil(p / 100.0 * n)) - 1;
        if (idx >= n) idx = n - 1;
        return samples[idx];
    };

    return {m, sd, (double)pct(50), (double)pct(99),
            (double)samples.front(), (double)samples.back()};
}

static void print_stats(const std::string &label, Stats s, int iterations) {
    std::cout << "\n--------------------------------------------------\n";
    std::cout << label << "  (" << iterations << " iterations)\n";
    std::cout << "  mean  : " << std::fixed << std::setprecision(0) << s.mean << " ns\n";
    std::cout << "  stddev: " << s.stddev << " ns\n";
    std::cout << "  p50   : " << s.p50 << " ns\n";
    std::cout << "  p99   : " << s.p99 << " ns\n";
    std::cout << "  min   : " << s.min_val << " ns\n";
    std::cout << "  max   : " << s.max_val << " ns\n";

    double jitter = s.p99 - s.p50;
    double ratio = s.mean > 0 ? s.p99 / s.mean : 0;
    std::cout << "  p99-p50 gap: " << jitter << " ns";
    if (ratio > 3.0)
        std::cout << "  ⚠ SEVERE tail latency (p99/mean=" << std::fixed << std::setprecision(1) << ratio << "x)";
    else if (ratio > 1.5)
        std::cout << "  ⚡ moderate jitter (p99/mean=" << std::fixed << std::setprecision(1) << ratio << "x)";
    else
        std::cout << "  ✓ stable";
    std::cout << "\n";
}

// ============================================================================
// Test 1: OS 调度抖动 — CFS 时间片抢占导致测量线程被踢出 CPU
// 方法：创建大量忙等噪声线程竞争 CPU，观察测量循环的超时幅度
// 判断：p99/mean > 3x → 调度器抢占严重
// 解决：绑核 + SCHED_FIFO + CPU 隔离（isolcpus）
// ============================================================================

void test_os_scheduling(int num_noisy_threads, int iterations) {
    std::atomic<bool> stop{false};
    std::atomic<int64_t> dummy{0};

    // Spawn noise threads that busy-spin (burn CPU, force scheduler to preempt)
    std::vector<std::thread> noise;
    for (int i = 0; i < num_noisy_threads; i++) {
        noise.emplace_back([&]() {
            while (!stop.load(std::memory_order_relaxed)) {
                // Busy work that thrashes L1 and forces context switching
                for (int k = 0; k < 1000; k++) {
                    dummy.fetch_add(k, std::memory_order_relaxed);
                }
            }
        });
    }

    // Measurement thread: record the latency of a spin-wait loop
    std::vector<int64_t> samples;
    samples.reserve(iterations);

    auto start = now_ns();
    auto deadline = start;
    for (int i = 0; i < iterations; i++) {
        deadline += 100000; // target 100us interval (10kHz)
        // Spin-wait until deadline (precision wait)
        while (now_ns() < deadline) {
            __builtin_ia32_pause(); // PAUSE hint to CPU
        }
        int64_t actual = now_ns();
        int64_t error_ns = actual - deadline; // positive = overshoot (jitter)
        samples.push_back(error_ns);
    }

    stop.store(true);
    for (auto &t : noise) t.join();

    print_stats("[OS Scheduling] Overshoot vs 100us deadline with "
                    + std::to_string(num_noisy_threads) + " noise threads",
                compute_stats(samples), iterations);
}

// ============================================================================
// Test 2: 缓存 / TLB 未命中 — 不同访问模式的单次延迟
// 方法：用不同大小的 buffer + stride 访问，RDTSC 测量每次访存延迟
// 判断：<5ns → L1/L2 hit；<30ns → L3 hit；>50ns → DRAM；>100ns → TLB miss
// 解决：数据重排（SoA）、prefetch、大页（huge pages）
// ============================================================================

static void test_cache_miss(int iterations) {
    constexpr size_t L1_SIZE = 32 * 1024;           // 32KB L1
    constexpr size_t L3_SIZE = 8 * 1024 * 1024;     // 8MB L3 (adjust for your CPU)
    constexpr size_t DRAM_SIZE = 128 * 1024 * 1024; // 128MB off-chip

    struct {
        const char *label;
        size_t size;
        int stride;
    } configs[] = {
        {"[Cache] Sequential L1-size (cache hit)", L1_SIZE, 1},
        {"[Cache] Sequential L3-size (L3 hit)", L3_SIZE, 1},
        {"[Cache] Sequential DRAM-size (cache miss)", DRAM_SIZE, 1},
        {"[Cache] Stride=16 (max cache line skip)", DRAM_SIZE, 16},
        {"[TLB]   Stride=512 (TLB miss every access)", DRAM_SIZE, 512},
    };

    volatile char sink = 0;
    for (auto &cfg : configs) {
        size_t num_elements = cfg.size / sizeof(char);
        std::vector<char> buffer(num_elements, 0);

        std::vector<int64_t> samples;
        samples.reserve(iterations);

        for (int iter = 0; iter < iterations; iter++) {
            // Use RDTSC for higher precision (sub-nanosecond on modern CPUs)
            uint64_t t0 = __builtin_ia32_rdtsc();
            for (size_t i = 0; i < num_elements; i += cfg.stride) {
                sink = buffer[i];
            }
            uint64_t t1 = __builtin_ia32_rdtsc();
            // Per-element average in cycles → convert to ns (assume ~3GHz → /3)
            int64_t total_cycles = static_cast<int64_t>(t1 - t0);
            int64_t num_accesses = static_cast<int64_t>(num_elements / cfg.stride + 1);
            int64_t per_access_cycles = total_cycles / num_accesses;
            int64_t per_access_ns = per_access_cycles / 3; // rough: 1 cycle ≈ 0.33ns @ 3GHz
            samples.push_back(per_access_ns);
        }

        auto s = compute_stats(samples);
        std::cout << "\n"
                  << cfg.label << "\n";
        std::cout << "  per-access mean: " << std::fixed << std::setprecision(1) << s.mean << " ns";
        if (s.mean < 5)
            std::cout << " → likely L1/L2 hit";
        else if (s.mean < 30)
            std::cout << " → likely L3 hit";
        else if (s.mean < 100)
            std::cout << " → likely DRAM access";
        else
            std::cout << " → possible TLB miss or major page fault";
        std::cout << "  (stddev: " << s.stddev << " ns)\n";
    }
    (void)sink;
}

// ============================================================================
// Test 3: 内存带宽竞争 — 多个核心同时访 DRAM 时的互相拖慢
// 方法：1 个测量线程扫描 64MB buffer，逐步加入竞争者线程同时扫描各自 buffer
// 判断：contended 延迟 > solo 的 2x → 带宽瓶颈；stddev 明显增大 → 竞争抖动
// 解决：减少并发访存线程、NUMA 感知分配、FP32→FP16 压缩数据
// ============================================================================

void test_memory_bandwidth(int num_contenders, int iterations) {
    constexpr size_t BUFFER_SIZE = 64 * 1024 * 1024; // 64MB per thread
    std::atomic<bool> start_flag{false};
    std::atomic<bool> stop{false};

    // Pre-allocate per-thread buffers to avoid malloc during measurement
    struct ThreadBuf {
        std::vector<char> buf;
        volatile int64_t sum = 0;
    };
    std::vector<ThreadBuf> bufs(num_contenders + 1);
    for (auto &b : bufs)
        b.buf.resize(BUFFER_SIZE, 0);

    // Spawn contender threads
    std::vector<std::thread> contenders;
    for (int t = 0; t < num_contenders; t++) {
        contenders.emplace_back([&, t]() {
            while (!start_flag.load(std::memory_order_acquire));
            auto &b = bufs[t + 1]; // offset: thread 0 is measurement
            while (!stop.load(std::memory_order_relaxed)) {
                for (size_t i = 0; i < BUFFER_SIZE; i += 64) { // 64-byte stride = cache line
                    b.sum += static_cast<int64_t>(b.buf[i]);
                }
            }
        });
    }

    // Warm up measurement thread
    {
        auto &b = bufs[0];
        for (size_t i = 0; i < BUFFER_SIZE; i += 64) b.sum += static_cast<int64_t>(b.buf[i]);
    }

    // Baseline: memory throughput alone
    std::vector<int64_t> samples_baseline, samples_contended;
    samples_baseline.reserve(iterations);
    samples_contended.reserve(iterations);

    // Phase 1: solo (no contention)
    for (int i = 0; i < iterations; i++) {
        auto &b = bufs[0];
        int64_t t0 = now_ns();
        for (size_t j = 0; j < BUFFER_SIZE; j += 64)
            b.sum += static_cast<int64_t>(b.buf[j]);
        int64_t t1 = now_ns();
        samples_baseline.push_back(t1 - t0);
    }

    // Phase 2: contended (launch contenders)
    start_flag.store(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(50)); // let contenders ramp up

    for (int i = 0; i < iterations; i++) {
        auto &b = bufs[0];
        int64_t t0 = now_ns();
        for (size_t j = 0; j < BUFFER_SIZE; j += 64)
            b.sum += static_cast<int64_t>(b.buf[j]);
        int64_t t1 = now_ns();
        samples_contended.push_back(t1 - t0);
    }

    stop.store(true);
    for (auto &t : contenders) t.join();

    auto s_solo = compute_stats(samples_baseline);
    auto s_cont = compute_stats(samples_contended);

    std::cout << "\n[Memory Bandwidth] " << num_contenders << " contender threads\n";
    std::cout << "  solo (mean):       " << std::fixed << std::setprecision(0)
              << s_solo.mean << " ns per 64MB scan\n";
    std::cout << "  contended (mean):   " << s_cont.mean << " ns per 64MB scan\n";
    std::cout << "  slow-down factor:   " << std::fixed << std::setprecision(2)
              << (s_cont.mean / s_solo.mean) << "x\n";
    std::cout << "  solo stddev:        " << s_solo.stddev << " ns\n";
    std::cout << "  contended stddev:   " << s_cont.stddev << " ns (increased = jitter from contention)\n";
}

// ============================================================================
// Test 4: 中断（IRQ）抢占检测 — 发现周期性的 >10us 延迟尖刺
// 方法：以 1us 为周期做精准忙等，记录任何 overshoot > 10us 的异常时刻
// 来源：定时器中断、网卡、磁盘 IO、内核 tick（250Hz/1000Hz）
// 判断：spike > 10us 频率 > 100/百万次 → 中断风暴；周期性出现 → 时钟 tick
// 解决：IRQ affinity 隔离 + nohz_full + 禁用无关设备
// ============================================================================

void test_irq_detection(int duration_sec) {
    std::cout << "\n[IRQ Detection] Monitoring for " << duration_sec
              << "s - long latency spikes → likely interrupt/scheduling\n";

    const int64_t target_ns = 1000; // 1us per iteration
    int64_t deadline = now_ns() + target_ns;
    int64_t end_time = now_ns() + duration_sec * 1'000'000'000LL;

    std::vector<int64_t> spikes; // latencies > 10x target
    int64_t total_iters = 0;
    int64_t max_seen = 0;

    while (now_ns() < end_time) {
        // Tight busy-wait until deadline
        while (now_ns() < deadline) {
            __builtin_ia32_pause();
        }
        int64_t actual = now_ns();
        int64_t overshoot = actual - deadline;
        total_iters++;
        if (overshoot > max_seen) max_seen = overshoot;

        // A normal overshoot is ~50-200ns (function call + timer precision).
        // Anything > 10us is an external disturbance (IRQ, preemption, etc.)
        if (overshoot > 10'000) { // 10us threshold
            spikes.push_back(overshoot);
        }

        deadline += target_ns;
        if (deadline < now_ns()) deadline = now_ns() + target_ns; // catch up
    }

    std::cout << "  total iterations: " << total_iters << "\n";
    std::cout << "  spikes > 10us:    " << spikes.size();
    if (spikes.empty())
        std::cout << " → system is quiet (few interrupts)\n";
    else {
        double spike_rate = spikes.size() * 1e6 / (duration_sec * 1'000'000.0);
        std::cout << " (" << std::fixed << std::setprecision(1)
                  << spike_rate << " per million iterations)\n";

        std::sort(spikes.begin(), spikes.end());
        std::cout << "  max spike:        " << *std::max_element(spikes.begin(), spikes.end()) / 1000.0
                  << " us\n";
        std::cout << "  p99 spike:        " << spikes[spikes.size() * 99 / 100] / 1000.0 << " us\n";

        // Print top 10 spikes for inspection
        std::cout << "  top 5 spikes (us):";
        size_t show = std::min(spikes.size(), size_t(5));
        for (size_t i = spikes.size() - show; i < spikes.size(); i++)
            std::cout << " " << spikes[i] / 1000.0;
        std::cout << "\n";

        if (spike_rate > 100)
            std::cout << "  ⚠ High interrupt rate! Check /proc/interrupts\n";
    }
    std::cout << "  overall max:      " << max_seen / 1000.0 << " us\n";
}

// ============================================================================
// Test 5: CPU 频率调节 / DVFS 抖动 — 频率不稳定导致周期数波动
// 方法：用 RDTSC 测量固定 ALU 工作的周期数，波动说明频率在变
// 判断：cycle variation > 5% → DVFS 或 thermal throttling 活跃
// 解决：governor → performance + 锁定最高 P-state + 加强散热
// ============================================================================

void test_frequency_scaling(int iterations) {
    std::vector<int64_t> samples;
    samples.reserve(iterations);

    // Use RDTSC to measure CPU cycles for a fixed amount of work
    // If freq is stable, cycle counts should be consistent
    volatile int sum = 0;
    for (int i = 0; i < iterations; i++) {
        uint64_t t0 = __builtin_ia32_rdtsc();
        // Do a fixed amount of ALU work (should take constant cycles regardless of DRAM)
        for (int k = 0; k < 100000; k++) {
            sum += k * 3 + 1;
        }
        uint64_t t1 = __builtin_ia32_rdtsc();
        samples.push_back(static_cast<int64_t>(t1 - t0));
    }

    auto s = compute_stats(samples);
    print_stats("[CPU Frequency] RDTSC cycles for fixed ALU work", s, iterations);

    double variation = s.mean > 0 ? (s.max_val - s.min_val) / s.mean * 100 : 0;
    std::cout << "  cycle variation: " << std::fixed << std::setprecision(1)
              << variation << "% ";
    if (variation > 5)
        std::cout << "⚠ DVFS / thermal throttling may be active\n";
    else
        std::cout << "→ frequency appears stable\n";
    (void)sum;
}

// ============================================================================
// Test 6: malloc 动态分配抖动 — new/delete 可能触发 page fault、brk、arena lock
// 方法：循环 malloc + touch + free，测量 4KB 和 1MB 两种大小的分配延迟
// 判断：p99/mean > 5x → malloc 内部锁或 page fault 造成尾延迟
// 解决：预分配 buffer 复用 + 线程局部内存池 + 禁止运行时 malloc
// ============================================================================

void test_malloc_jitter(int iterations) {
    std::vector<int64_t> samples;
    samples.reserve(iterations);

    // Pre-warm: force glibc to grow heap once
    {
        std::vector<char *> ptrs;
        for (int i = 0; i < 100; i++) ptrs.push_back(new char[1024]);
    }

    for (int i = 0; i < iterations; i++) {
        int64_t t0 = now_ns();
        char *p = new char[4096]; // one page
        p[0] = 42;                // touch to trigger actual page fault
        p[4095] = 43;
        int64_t t1 = now_ns();
        delete[] p;
        samples.push_back(t1 - t0);
    }

    auto s = compute_stats(samples);
    print_stats("[malloc] 4KB allocate + touch + free", s, iterations);

    // Also test large allocation (may trigger mmap)
    std::vector<int64_t> samples_large;
    for (int i = 0; i < iterations / 10; i++) {
        int64_t t0 = now_ns();
        char *p = new char[1024 * 1024]; // 1MB
        p[0] = 1;                        // touch first byte
        int64_t t1 = now_ns();
        delete[] p;
        samples_large.push_back(t1 - t0);
    }
    auto s2 = compute_stats(samples_large);
    print_stats("[malloc] 1MB allocate + touch + free", s2, samples_large.size());
}

// ============================================================================
// Main
// ============================================================================

static void print_header() {
    std::cout << "=================================================================\n";
    std::cout << "  Jitter Source Test Suite\n";
    std::cout << "  Each test reveals a different source of latency instability.\n";
    std::cout << "=================================================================\n";
    std::cout << "\nHow to diagnose jitter:\n";
    std::cout << "  1. Run ALL tests below → note which one shows high stddev/p99\n";
    std::cout << "  2. Use system tools to confirm (see jitter_diagnosis_guide.md)\n";
    std::cout << "  3. Apply the corresponding fix\n";
    std::cout << "\n";
}

int main(int argc, char **argv) {
    print_header();

    // Default parameters
    int noise_threads = std::thread::hardware_concurrency();
    int contention_threads = std::max(1, (noise_threads / 2));
    int duration_sec = 5;
    int iterations = 500;

    // Simple arg parsing
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--noise" && i + 1 < argc) noise_threads = std::stoi(argv[++i]);
        if (arg == "--contention" && i + 1 < argc) contention_threads = std::stoi(argv[++i]);
        if (arg == "--duration" && i + 1 < argc) duration_sec = std::stoi(argv[++i]);
        if (arg == "--iters" && i + 1 < argc) iterations = std::stoi(argv[++i]);
        if (arg == "--help") {
            std::cout << "Usage: test_jitter [options]\n";
            std::cout << "  --noise N       Num noise threads for scheduling test\n";
            std::cout << "  --contention N  Num contender threads for bandwidth test\n";
            std::cout << "  --duration N    Seconds for IRQ detection\n";
            std::cout << "  --iters N       Iterations for stat tests\n";
            return 0;
        }
    }

    std::cout << "Config: noise_threads=" << noise_threads
              << " contention_threads=" << contention_threads
              << " irq_duration=" << duration_sec << "s"
              << " iterations=" << iterations << "\n";

    // Run all 6 tests in order
    test_os_scheduling(noise_threads, iterations); // Test 1
    test_cache_miss(iterations);                   // Test 2

    if (noise_threads < 2) {
        std::cout << "\n[Memory Bandwidth] SKIPPED - need >= 2 cores for contention test\n";
    } else {
        test_memory_bandwidth(contention_threads, iterations / 10); // Test 3
    }

    test_irq_detection(duration_sec);   // Test 4
    test_frequency_scaling(iterations); // Test 5
    test_malloc_jitter(iterations);     // Test 6

    std::cout << "\n=================================================================\n";
    std::cout << "  All tests complete.\n";
    std::cout << "  See jitter_diagnosis_guide.md for how to use system tools\n";
    std::cout << "  to confirm each jitter source.\n";
    std::cout << "=================================================================\n";

    return 0;
}
