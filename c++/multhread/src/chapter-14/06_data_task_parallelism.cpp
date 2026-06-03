// 06_data_task_parallelism.cpp — 数据并行 vs 任务并行对比
// 演示: 同任务分块 vs 不同任务并行、混合模式

#include <barrier>
#include <chrono>
#include <functional>
#include <iostream>
#include <syncstream>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// ===== 1. 数据并行: 相同操作在不同数据块 =====
void demo_data_parallelism() {
    std::cout << "=== 数据并行 (Data Parallelism) ===\n";

    const size_t kDataSize = 1'000'000;
    std::vector<int> data(kDataSize);
    for (size_t i = 0; i < kDataSize; ++i) data[i] = static_cast<int>(i);

    auto compute_square = [](const std::vector<int>& input,
                              std::vector<int>& output,
                              size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            output[i] = input[i] * input[i];
        }
    };

    std::vector<int> result(kDataSize);
    const unsigned kNumThreads = std::thread::hardware_concurrency();
    const size_t kChunkSize = kDataSize / kNumThreads;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::jthread> threads;
    for (unsigned t = 0; t < kNumThreads; ++t) {
        size_t begin = t * kChunkSize;
        size_t end = (t == kNumThreads - 1) ? kDataSize : begin + kChunkSize;
        threads.emplace_back([&, begin, end]() {
            compute_square(data, result, begin, end);
        });
    }
    threads.clear();

    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);
    std::cout << "  " << kDataSize << " 个元素平方计算: "
              << elapsed.count() << " ms\n";
    std::cout << "  验证: data[100]^2 = " << data[100] * data[100]
              << ", result[100] = " << result[100] << "\n";
}

// ===== 2. 任务并行: 不同任务分配给不同线程 =====
void demo_task_parallelism() {
    std::cout << "\n=== 任务并行 (Task Parallelism) ===\n";

    auto parse = []() {
        std::osyncstream(std::cout) << "  [Parse]   解析中...\n";
        std::this_thread::sleep_for(50ms);
        std::osyncstream(std::cout) << "  [Parse]   完成\n";
    };

    auto validate = []() {
        std::osyncstream(std::cout) << "  [Validate] 验证中...\n";
        std::this_thread::sleep_for(30ms);
        std::osyncstream(std::cout) << "  [Validate] 完成\n";
    };

    auto save = []() {
        std::osyncstream(std::cout) << "  [Save]    存储中...\n";
        std::this_thread::sleep_for(40ms);
        std::osyncstream(std::cout) << "  [Save]    完成\n";
    };

    auto start = std::chrono::high_resolution_clock::now();

    std::jthread t1(parse);
    std::jthread t2(validate);
    std::jthread t3(save);
    t1.join();
    t2.join();
    t3.join();

    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);
    std::cout << "  总耗时: " << elapsed.count()
              << " ms (最慢任务: parse 50ms)\n";
}

// ===== 3. 混合模式: 数据并行 + 任务并行 =====
void demo_hybrid_parallelism() {
    std::cout << "\n=== 混合并行模式 ===\n";

    // 场景: 图像处理流水线
    // Stage 1: 数据并行加载多个图像
    // Stage 2: 任务并行处理过滤/分析
    // 简化: 两个阶段用 barrier 同步

    const int kNumImages = 8;
    std::vector<int> images(kNumImages);
    std::vector<int> processed(kNumImages);

    std::barrier sync(2);

    // 数据并行加载
    std::jthread loader([&]() {
        std::osyncstream(std::cout) << "  [Loader] 加载 " << kNumImages
                                    << " 张图像\n";
        for (int i = 0; i < kNumImages; ++i) {
            images[i] = i * 100;
        }
        std::this_thread::sleep_for(30ms);
        sync.arrive_and_wait(); // 等处理器就绪

        // 阶段2: 合并结果
        std::osyncstream(std::cout) << "  [Loader] 检查处理结果\n";
        sync.arrive_and_wait();
    });

    // 任务并行处理
    std::jthread processor([&]() {
        sync.arrive_and_wait(); // 等加载完成

        std::osyncstream(std::cout) << "  [Processor] 处理中...\n";
        for (int i = 0; i < kNumImages; ++i) {
            processed[i] = images[i] + 1;
        }
        std::this_thread::sleep_for(20ms);

        sync.arrive_and_wait();
    });

    loader.join();
    processor.join();

    std::cout << "  处理完成: images[0]=" << images[0]
              << ", processed[0]=" << processed[0] << "\n";
    std::cout << "  混合模式适合多阶段流水线并行\n";
}

int main() {
    demo_data_parallelism();
    demo_task_parallelism();
    demo_hybrid_parallelism();

    std::cout << "\n选择指南:\n";
    std::cout << "  大规模同构数据 → 数据并行\n";
    std::cout << "  独立异构任务   → 任务并行\n";
    std::cout << "  多阶段处理     → 混合/Pipeline\n";
    return 0;
}
