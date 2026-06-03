// 01_cache_line_alignment.cpp — Cache Line 对齐演示
// 演示: cache line 大小检测、alignas 用法、跨行访问性能差异

#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

// ===== 1. 检测当前平台的 cache line 大小 =====
// 方法: 通过 /sys 文件系统或 CPUID 指令
// 简化: 大多数 x86-64 平台为 64 字节
#ifdef __linux__
#include <cstdio>
size_t detect_cache_line_size() {
    long size = 0;
    FILE* fp = fopen(
        "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size", "r");
    if (fp) {
        fscanf(fp, "%ld", &size);
        fclose(fp);
    }
    return size > 0 ? static_cast<size_t>(size) : 64;
}
#else
size_t detect_cache_line_size() { return 64; }
#endif

// ===== 2. 未对齐 vs 对齐访问性能对比 =====
struct Unaligned {
    int counter;
    // 编译器可能在此插入 padding，但不能保证独占 cache line
};

struct alignas(64) Aligned {
    int counter;
    // 编译器确保整个结构体占用 64 字节，对齐到 64 字节边界
};

void demo_alignment_performance() {
    std::cout << "=== Cache Line 对齐性能对比 ===\n";
    const size_t kCacheLine = detect_cache_line_size();
    std::cout << "  检测到的 Cache Line 大小: " << kCacheLine << " 字节\n";

    // 验证对齐
    std::cout << "  sizeof(Unaligned) = " << sizeof(Unaligned)
              << " (align = " << alignof(Unaligned) << ")\n";
    std::cout << "  sizeof(Aligned)   = " << sizeof(Aligned)
              << " (align = " << alignof(Aligned) << ")\n";

    const int kIters = 10'000'000;

    // 测试: 多线程频繁写相邻的未对齐计数器
    {
        // 故意放在连续内存中（同一 cache line 概率高）
        std::vector<Unaligned> counters(4);

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::jthread> threads;
        for (int i = 0; i < 4; ++i) {
            threads.emplace_back([&counters, i, kIters]() {
                for (int j = 0; j < kIters; ++j) {
                    counters[i].counter++;
                }
            });
        }
        threads.clear();

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);
        std::cout << "  未对齐 (可能伪共享): "
                  << elapsed.count() << " ms\n";
    }

    // 测试: 对齐计数器
    {
        std::vector<Aligned> counters(4);

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::jthread> threads;
        for (int i = 0; i < 4; ++i) {
            threads.emplace_back([&counters, i, kIters]() {
                for (int j = 0; j < kIters; ++j) {
                    counters[i].counter++;
                }
            });
        }
        threads.clear();

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);
        std::cout << "  对齐 (无伪共享):     "
                  << elapsed.count() << " ms\n";
    }
}

// ===== 3. 跨 cache line 访问的性能影响 =====
void demo_cross_line_access() {
    std::cout << "\n=== 跨 Cache Line 访问的影响 ===\n";

    // 分配一块连续内存，测试跨行访问的额外开销
    const size_t kSize = 1024 * 1024; // 1M 个 int
    std::vector<int> data(kSize);

    // 步长测试: 不同步长意味着不同的 cache line 利用率
    for (int stride : {1, 16, 64, 256}) {
        long long sum = 0;
        auto start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < kSize; i += stride) {
            sum += data[i];
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start);

        std::cout << "  stride=" << stride
                  << " (访问 " << kSize / stride << " 个元素): "
                  << elapsed.count() << " us"
                  << " | sum=" << sum << "\n";
    }
    std::cout << "  结论: stride 越大(跨行越多)，性能越差\n";
}

// ===== 4. std::hardware_destructive_interference_size =====
void demo_standard_interference_size() {
    std::cout << "\n=== 标准干扰大小常量 ===\n";

#ifdef __cpp_lib_hardware_interference_size
    std::cout << "  hardware_destructive_interference_size = "
              << std::hardware_destructive_interference_size << "\n";
    std::cout << "  hardware_constructive_interference_size  = "
              << std::hardware_constructive_interference_size << "\n";
#else
    std::cout << "  hardware_destructive_interference_size = "
              << "不可用 (需要 GCC 12+ / Clang 15+)\n";
    std::cout << "  hardware_constructive_interference_size  = "
              << "不可用\n";
#endif
    std::cout << "  注意: 这两个值在某些编译器中为建议值(64)，"
              << "非强制对齐\n";
}

int main() {
    demo_alignment_performance();
    demo_cross_line_access();
    demo_standard_interference_size();

    return 0;
}
