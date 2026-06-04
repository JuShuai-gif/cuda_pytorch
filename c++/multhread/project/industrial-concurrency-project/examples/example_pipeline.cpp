// Ch8.3 & Ch8.4：流水线并发示例
// 使用 TaskScheduler 实现 3 阶段 AI/ML 推理流水线：
//   阶段 1：预处理  （批量归一化、数据增强） - Ch8.2
//   阶段 2：推理    （模型前向传播）            - Ch8.3
//   阶段 3：后处理  （NMS、结果格式化）          - Ch8.4
// 演示：Ch8（工作划分）、Ch4.2（future 链式调用）、Ch3.2（共享数据）。

#include "task_scheduler/task_scheduler.hpp"
#include "task_scheduler/logger.hpp"
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <chrono>
#include "task_scheduler/format_compat.hpp"

using namespace task_scheduler;

// Ch8.2.1：一批模拟图像数据。
struct ImageBatch {
    int batch_id;
    std::vector<float> pixels; // 原始像素数据
};

// Ch8.2.2：模拟推理结果。
struct InferenceResult {
    int batch_id;
    std::vector<float> class_probabilities;
};

// Ch8.2.3：最终处理结果。
struct ProcessedResult {
    int batch_id;
    int predicted_class;
    float confidence;
};

// 阶段 1：预处理（Ch8.2.4：数据并行工作）。
// 预处理阶段：归一化像素数据
ImageBatch preprocess_stage(int batch_id, int width, int height) {
    Logger::instance().debug(TS_FORMAT("Pipeline: preprocessing batch {}", batch_id));

    ImageBatch batch;
    batch.batch_id = batch_id;
    batch.pixels.resize(width * height);

    // 模拟归一化
    std::mt19937 rng(batch_id);
    std::uniform_real_distribution<float> dist(0.0f, 255.0f);
    for (auto& p : batch.pixels) {
        p = (dist(rng) - 127.5f) / 127.5f; // 归一化到 [-1, 1]
    }

    // Ch8.4.5：模拟计算开销。
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return batch;
}

// 阶段 2：推理（Ch8.3.1：核心计算密集型阶段）。
// 推理阶段：模拟神经网络前向传播
InferenceResult inference_stage(const ImageBatch& batch) {
    Logger::instance().debug(TS_FORMAT("Pipeline: inference on batch {}", batch.batch_id));

    InferenceResult result;
    result.batch_id = batch.batch_id;
    result.class_probabilities.resize(10);

    // 模拟神经网络前向传播。
    float sum = std::accumulate(batch.pixels.begin(), batch.pixels.end(), 0.0f);
    for (size_t i = 0; i < 10; ++i) {
        result.class_probabilities[i] = std::abs(std::sin(sum + i * 0.1f));
    }

    // 模拟计算开销（推理是瓶颈）。
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    return result;
}

// 阶段 3：后处理（Ch8.4.1：结果聚合）。
// 后处理阶段：找出最可能的类别
ProcessedResult postprocess_stage(const InferenceResult& result) {
    Logger::instance().debug(TS_FORMAT("Pipeline: post-processing batch {}", result.batch_id));

    ProcessedResult final_result;
    final_result.batch_id = result.batch_id;

    // 找到最高概率的类别（模拟 softmax + argmax）。
    auto max_it = std::max_element(result.class_probabilities.begin(),
                                    result.class_probabilities.end());
    final_result.predicted_class = static_cast<int>(
        std::distance(result.class_probabilities.begin(), max_it));
    final_result.confidence = *max_it;

    return final_result;
}

// Ch8.3.2：为单个批次运行完整的流水线。
// 运行完整流水线：预处理 -> 推理 -> 后处理
ProcessedResult run_pipeline(TaskScheduler& scheduler, int batch_id,
                              int width, int height) {
    // Ch8.3.3：使用 TaskScheduler 流水线支持链接各阶段。
    // 阶段1 -> 阶段2 + 阶段3（延续传递，Ch4.2.3）。
    auto future = scheduler.submit_pipeline<ImageBatch, ProcessedResult>(
        TS_FORMAT("pipeline_batch_{}", batch_id),
        // 阶段 1：预处理
        [batch_id, width, height]() { return preprocess_stage(batch_id, width, height); },
        // 阶段 2+3：推理 + 后处理（为简化合并）
        [](ImageBatch batch) {
            auto inf_result = inference_stage(batch);
            return postprocess_stage(inf_result);
        }
    );

    return future.get();
}

int main() {
    Logger::instance().set_level(LogLevel::INFO);
    Logger::instance().info("=== 示例：AI/ML 推理流水线 ===");

    // Ch8.4.1：创建有足够线程的调度器以支持流水线并行。
    TaskScheduler scheduler(8);

    constexpr int num_batches = 20;
    constexpr int image_width = 64;
    constexpr int image_height = 64;

    std::cout << TS_FORMAT("Processing {} batches of {}x{} images...\n",
                              num_batches, image_width, image_height);

    auto start = std::chrono::steady_clock::now();

    // Ch8.4.2：提交所有批次用于并行流水线执行。
    std::vector<std::future<ProcessedResult>> futures;
    for (int i = 0; i < num_batches; ++i) {
        // 每个流水线在线程池的独立线程上运行。
        // 多个批次并行处理（Ch8.4.6：流水线并行）。
        futures.push_back(
            scheduler.submit(TaskPriority::HIGH, TS_FORMAT("batch_{}", i),
                [&scheduler, i] { return run_pipeline(scheduler, i, image_width, image_height); })
        );
    }

    // Ch4.2.4：收集所有结果。
    int correct = 0;
    for (auto& f : futures) {
        auto result = f.get();
        std::cout << TS_FORMAT("  Batch {}: class={}, confidence={:.2f}%\n",
                                  result.batch_id, result.predicted_class,
                                  result.confidence * 100.0f);
        if (result.confidence > 0.5f) correct++;
    }

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);

    // Ch8.4.7：报告吞吐量指标。
    std::cout << TS_FORMAT("\nPipeline Stats:\n");
    std::cout << TS_FORMAT("  Total batches: {}\n", num_batches);
    std::cout << TS_FORMAT("  High-confidence: {}/{}\n", correct, num_batches);
    std::cout << TS_FORMAT("  Total time: {}ms\n", elapsed.count());
    std::cout << TS_FORMAT("  Throughput: {:.1f} batches/sec\n",
                              num_batches * 1000.0 / elapsed.count());

    Logger::instance().info("=== 流水线示例完成 ===");
    return 0;
}
