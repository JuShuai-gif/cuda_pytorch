#include "cache_bench.h"
#include "timer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

// ============================================================================
// 演示 1: 通过步长访问检测缓存行大小
// ============================================================================
void demo_cache_line_detection() {
    print_header("演示 1: 缓存行大小检测");

    constexpr int ARRAY_SIZE = 64 * 1024 * 1024; // 64M 个整数 = 256MB
    std::vector<int> data(ARRAY_SIZE, 1);

    std::cout << "  测量不同步长的访问时间...\n\n";
    std::cout << "  " << std::left << std::setw(12) << "步长(字节)"
              << std::setw(18) << "时间(ms)"
              << std::setw(18) << "纳秒/访问"
              << "  备注\n";
    std::cout << "  " << std::string(60, '-') << "\n";

    constexpr int STEPS = 64 * 1024 * 1024;
    std::vector<double> times;

    for (int stride_bytes : {4, 8, 16, 32, 48, 64, 80, 96, 128, 256, 512}) {
        int stride = stride_bytes / static_cast<int>(sizeof(int));
        if (stride == 0) stride = 1;

        Timer t;
        t.start();
        long sum = 0;
        for (size_t i = 0; i < STEPS; ++i) {
            sum += data[(i * static_cast<size_t>(stride)) % ARRAY_SIZE];
        }
        double ms = t.elapsed_ms();
        double ns = ms * 1e6 / STEPS;
        times.push_back(ns);
        g_sink = sum;

        std::string note;
        if (stride_bytes <= 16)
            note = "L1 命中";
        else if (stride_bytes == 64)
            note = "缓存行边界";
        else if (stride_bytes >= 128)
            note = "更多缓存未命中";

        std::cout << "  " << std::left << std::setw(12) << stride_bytes
                  << std::fixed << std::setprecision(3) << std::setw(18) << ms
                  << std::fixed << std::setprecision(2) << std::setw(18) << ns
                  << note << "\n";
    }

    std::cout << "\n  => 步长 > 64 时纳秒/访问较高，说明缓存行 "
              << "大小为 64 字节 (x86)。\n";
}

// ============================================================================
// 演示 2: 缓存命中 vs 未命中延迟
// ============================================================================
void demo_cache_hit_vs_miss() {
    print_header("演示 2: 缓存命中 vs 未命中（顺序访问 vs 随机访问）");

    // 小数组：可放入 L1 缓存 (32KB / sizeof(int64_t) = 4096 个元素)
    {
        constexpr size_t SMALL = 1024; // 8KB，可放入 L1
        constexpr size_t ITERS = 10'000'000;
        std::vector<int64_t> data(SMALL);

        // 顺序访问（缓存友好）
        Timer t;
        t.start();
        long sum = 0;
        for (size_t rep = 0; rep < ITERS / SMALL; ++rep) {
            for (size_t i = 0; i < SMALL; ++i) {
                sum += data[i];
            }
        }
        double seq_ms = t.elapsed_ms();
        g_sink = sum;

        // 随机访问（缓存不友好，但数据足够小，仍留在 L1/L2 中）
        std::vector<size_t> indices(SMALL);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(indices.begin(), indices.end(), gen);

        t.start();
        sum = 0;
        for (size_t rep = 0; rep < ITERS / SMALL; ++rep) {
            for (size_t i = 0; i < SMALL; ++i) {
                sum += data[indices[i]];
            }
        }
        double rnd_ms = t.elapsed_ms();
        g_sink = sum;

        double seq_ns = seq_ms * 1e6 / ITERS;
        double rnd_ns = rnd_ms * 1e6 / ITERS;

        std::cout << "  [小数组: " << (SMALL * sizeof(int64_t) / 1024)
                  << " KB - 可放入 L1]\n";
        std::cout << "    顺序访问: " << std::fixed << std::setprecision(2)
                  << seq_ms << " ms  (" << seq_ns << " ns/次访问)\n";
        std::cout << "    随机访问:     " << std::fixed << std::setprecision(2)
                  << rnd_ms << " ms  (" << rnd_ns << " ns/次访问)\n";
        std::cout << "    比率:      " << std::fixed << std::setprecision(1)
                  << (rnd_ms / seq_ms) << "x\n\n";
    }

    // 大数组：超出 L3 缓存
    {
        constexpr size_t LARGE = 16 * 1024 * 1024; // 128MB，超出 L3
        constexpr size_t STEPS = 1'000'000;
        std::vector<int64_t> data(LARGE);

        // 顺序访问
        Timer t;
        t.start();
        long sum = 0;
        for (size_t i = 0; i < STEPS; ++i) {
            sum += data[i];
        }
        double seq_ms = t.elapsed_ms();
        g_sink = sum;

        // 随机访问（预生成索引）
        std::vector<size_t> indices(STEPS);
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<size_t> dist(0, LARGE - 1);
        for (size_t i = 0; i < STEPS; ++i) {
            indices[i] = dist(gen);
        }

        t.start();
        sum = 0;
        for (size_t i = 0; i < STEPS; ++i) {
            sum += data[indices[i]];
        }
        double rnd_ms = t.elapsed_ms();
        g_sink = sum;

        double seq_ns = seq_ms * 1e6 / STEPS;
        double rnd_ns = rnd_ms * 1e6 / STEPS;

        std::cout << "  [大数组: " << (LARGE * sizeof(int64_t) / (1024 * 1024))
                  << " MB - 超出 L3]\n";
        std::cout << "    顺序访问: " << std::fixed << std::setprecision(2)
                  << seq_ms << " ms  (" << seq_ns << " ns/次访问)\n";
        std::cout << "    随机访问:     " << std::fixed << std::setprecision(2)
                  << rnd_ms << " ms  (" << rnd_ns << " ns/次访问)\n";
        std::cout << "    比率:      " << std::fixed << std::setprecision(1)
                  << (rnd_ms / seq_ms) << "x\n\n";
    }

    std::cout << "  => 当数据超出缓存时，随机访问显著变慢。\n"
              << "  顺序访问受益于硬件预取器。\n";
}

// ============================================================================
// 演示 3: 伪共享
// ============================================================================
struct UnpaddedCounter {
    int64_t counter_a = 0;
    int64_t counter_b = 0;
};

struct alignas(64) PaddedCounter {
    int64_t counter_a = 0;
    char pad_a[64 - sizeof(int64_t)];
    int64_t counter_b = 0;
    char pad_b[64 - sizeof(int64_t)];
};

void demo_false_sharing() {
    print_header("演示 3: 伪共享与缓存行填充");

    constexpr int64_t ITERATIONS = 100'000'000;
    constexpr int NUM_RUNS = 5;

    // 预热以分配页面
    {
        UnpaddedCounter warm;
        std::thread t1([&]() {
            for (int64_t i = 0; i < 1000; ++i) warm.counter_a++;
        });
        std::thread t2([&]() {
            for (int64_t i = 0; i < 1000; ++i) warm.counter_b++;
        });
        t1.join();
        t2.join();
    }

    // 未填充（存在伪共享）
    std::vector<double> unpadded_times;
    for (int run = 0; run < NUM_RUNS; ++run) {
        UnpaddedCounter c;
        Timer t;
        t.start();
        std::thread t1([&]() {
            for (int64_t i = 0; i < ITERATIONS; ++i) c.counter_a++;
        });
        std::thread t2([&]() {
            for (int64_t i = 0; i < ITERATIONS; ++i) c.counter_b++;
        });
        t1.join();
        t2.join();
        unpadded_times.push_back(t.elapsed_ms());
    }
    double unpadded_avg = std::accumulate(unpadded_times.begin(),
                                          unpadded_times.end(), 0.0)
                          / NUM_RUNS;

    // 已填充（无伪共享）
    std::vector<double> padded_times;
    for (int run = 0; run < NUM_RUNS; ++run) {
        PaddedCounter c;
        Timer t;
        t.start();
        std::thread t1([&]() {
            for (int64_t i = 0; i < ITERATIONS; ++i) c.counter_a++;
        });
        std::thread t2([&]() {
            for (int64_t i = 0; i < ITERATIONS; ++i) c.counter_b++;
        });
        t1.join();
        t2.join();
        padded_times.push_back(t.elapsed_ms());
    }
    double padded_avg = std::accumulate(padded_times.begin(),
                                        padded_times.end(), 0.0)
                        / NUM_RUNS;

    std::cout << "  " << std::left << std::setw(20) << "配置"
              << std::right << std::setw(15) << "平均时间(ms)"
              << std::setw(15) << "比率" << "\n";
    std::cout << "  " << std::string(50, '-') << "\n";
    std::cout << "  " << std::left << std::setw(20) << "未填充（存在伪共享）"
              << std::right << std::fixed << std::setprecision(2)
              << std::setw(15) << unpadded_avg
              << std::setw(15) << std::setprecision(1) << (unpadded_avg / padded_avg)
              << "x\n";
    std::cout << "  " << std::left << std::setw(20) << "已填充（缓存对齐）"
              << std::right << std::fixed << std::setprecision(2)
              << std::setw(15) << padded_avg
              << std::setw(15) << "1.0x\n";

    std::cout << "\n  => 伪共享导致 " << std::fixed
              << std::setprecision(1) << (unpadded_avg / padded_avg)
              << "x 的性能下降，原因是缓存行乒乓效应。\n";
}

// ============================================================================
// 演示 5: 行优先 vs 列优先遍历
// ============================================================================
void demo_row_vs_column() {
    print_header("演示 5: 行优先 vs 列优先 2D 数组遍历");

    constexpr int SIZE = 8192; // 8192x8192 = 67M 元素 = 536MB (double)
    std::vector<double> matrix(SIZE * SIZE, 0.0);

    // 初始化
    for (size_t i = 0; i < matrix.size(); ++i) {
        matrix[i] = static_cast<double>(i & 0xFF);
    }

    // 行优先：内层循环遍历列
    {
        Timer t;
        t.start();
        volatile double sum = 0.0;
        for (int row = 0; row < SIZE; ++row) {
            for (int col = 0; col < SIZE; ++col) {
                sum += matrix[row * SIZE + col];
            }
        }
        double ms = t.elapsed_ms();
        double val = sum;
        (void)val;

        std::cout << "  行优先遍历: " << std::fixed
                  << std::setprecision(2) << ms << " ms\n";
    }

    // 列优先：内层循环遍历行
    {
        Timer t;
        t.start();
        volatile double sum = 0.0;
        for (int col = 0; col < SIZE; ++col) {
            for (int row = 0; row < SIZE; ++row) {
                sum += matrix[row * SIZE + col];
            }
        }
        double ms = t.elapsed_ms();
        double val = sum;
        (void)val;

        std::cout << "  列优先遍历: " << std::fixed
                  << std::setprecision(2) << ms << " ms\n";
    }

    std::cout << "\n  => 列优先遍历更慢，因为每次访问跳过了 "
              << "SIZE*sizeof(double) 字节，\n"
              << "  一旦数据超出 L1 缓存，每次访问都会导致缓存未命中。\n";
}
