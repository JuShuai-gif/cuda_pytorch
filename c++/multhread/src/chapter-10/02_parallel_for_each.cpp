/**
 * 02_parallel_for_each.cpp — 使用 C++17 parallel for_each
 *
 * std::for_each(std::execution::par, ...) 对每个元素并行应用函数。
 * 适用于独立的元素级操作 (无数据竞争)。
 *
 * 编译:
 *   GCC:  g++ -std=c++20 -O2 -pthread 02_parallel_for_each.cpp -ltbb -o parallel_for_each_stl
 */

#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>
#include <random>
#include <chrono>
#include <iomanip>
#include <thread>
#include <cmath>

#include <execution>
#ifdef HAS_TBB
    #define HAS_EXECUTION 1
#else
    #define HAS_EXECUTION 0
#endif

// ============================================================================
// 性能对比
// ============================================================================
template <typename Func>
double measure(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void benchmark() {
    std::cout << "=== C++17 并行 for_each 性能对比 ===\n\n";

    constexpr size_t N = 50'000'000;
    std::vector<double> original(N);
    std::iota(original.begin(), original.end(), 1.0);

    auto workload = [](double& x) {
        // 模拟重型计算
        x = std::sqrt(x) * std::log(x + 1.0) + std::sin(x * 0.001) * std::cos(x * 0.001);
    };

    std::cout << std::left
              << std::setw(20) << "策略"
              << std::setw(14) << "耗时"
              << std::setw(14) << "结果校验"
              << "\n";
    std::cout << std::string(48, '-') << "\n";

    // 顺序执行
    {
        auto data = original;
        double ms = measure([&]() {
            std::for_each(data.begin(), data.end(), workload);
        });

        // 校验: 计算和
        double sum = std::accumulate(data.begin(), data.end(), 0.0);
        std::cout << std::left
                  << std::setw(20) << "seq (顺序)"
                  << std::setw(14) << (std::to_string(static_cast<int>(ms)) + " ms")
                  << std::setw(14) << sum
                  << "\n";
    }

    // 并行执行
#if HAS_EXECUTION
    {
        auto data = original;
        double ms = measure([&]() {
            std::for_each(std::execution::par, data.begin(), data.end(), workload);
        });

        double sum = std::accumulate(data.begin(), data.end(), 0.0);
        std::cout << std::left
                  << std::setw(20) << "par (并行)"
                  << std::setw(14) << (std::to_string(static_cast<int>(ms)) + " ms")
                  << std::setw(14) << sum
                  << "\n";
    }

    // 并行+向量化
    {
        auto data = original;
        double ms = measure([&]() {
            std::for_each(std::execution::par_unseq, data.begin(), data.end(), workload);
        });

        double sum = std::accumulate(data.begin(), data.end(), 0.0);
        std::cout << std::left
                  << std::setw(20) << "par_unseq (并行+向量化)"
                  << std::setw(14) << (std::to_string(static_cast<int>(ms)) + " ms")
                  << std::setw(14) << sum
                  << "\n";
    }
#else
    std::cout << "  <execution> 不可用, 无法运行并行版本\n";
    std::cout << "  请安装 TBB: sudo apt install libtbb-dev && 链接 -ltbb\n";
#endif

    std::cout << "\n  硬件线程数: " << std::jthread::hardware_concurrency() << "\n";
}

// ============================================================================
// 使用场景: 图像处理模拟
// ============================================================================
void image_processing_demo() {
    std::cout << "\n=== 并行图像处理模拟 ===\n\n";

    // 模拟 1920x1080 图像像素
    constexpr size_t W = 1920, H = 1080, Channels = 3;
    constexpr size_t N = W * H * Channels;
    std::vector<uint8_t> image(N);

    // 填充模拟数据
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& p : image) p = static_cast<uint8_t>(dist(rng));

    // 亮度调节: 每个像素增加 20 (并行)
    constexpr uint8_t kBrightness = 20;

#if HAS_EXECUTION
    auto start = std::chrono::high_resolution_clock::now();

    std::for_each(std::execution::par, image.begin(), image.end(),
        [](uint8_t& p) {
            int val = p + kBrightness;
            p = static_cast<uint8_t>(val > 255 ? 255 : val); // clamp
        });

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "  图像尺寸: " << W << "x" << H << "x" << Channels << "\n";
    std::cout << "  像素数: " << N / 1000000.0 << "M\n";
    std::cout << "  亮度调节耗时: " << ms << " ms\n";
#else
    std::cout << "  <execution> 不可用, 跳过图像处理演示\n";
#endif
}

// ============================================================================
// main
// ============================================================================
int main() {
    benchmark();
    image_processing_demo();
    return 0;
}
