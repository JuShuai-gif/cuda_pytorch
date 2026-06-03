// Chapter 8.3 & 8.4: Pipeline Concurrency Example
// Implements a 3-stage AI/ML inference pipeline using TaskScheduler:
//   Stage 1: Preprocessing  (batch normalization, data augmentation) - Ch8.2
//   Stage 2: Inference       (model forward pass)                      - Ch8.3
//   Stage 3: Post-processing (NMS, result formatting)                   - Ch8.4
// Demonstrates: Ch8 (work division), Ch4.2 (future chaining), Ch3.2 (shared data).

#include "task_scheduler/task_scheduler.hpp"
#include "task_scheduler/logger.hpp"
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <chrono>
#include "task_scheduler/format_compat.hpp"

using namespace task_scheduler;

// Ch8.2.1: Simulated image data for a batch.
struct ImageBatch {
    int batch_id;
    std::vector<float> pixels; // Raw pixel data
};

// Ch8.2.2: Simulated inference result.
struct InferenceResult {
    int batch_id;
    std::vector<float> class_probabilities;
};

// Ch8.2.3: Final processed result.
struct ProcessedResult {
    int batch_id;
    int predicted_class;
    float confidence;
};

// Stage 1: Preprocessing (Ch8.2.4: data-parallel work).
ImageBatch preprocess_stage(int batch_id, int width, int height) {
    Logger::instance().debug(TS_FORMAT("Pipeline: preprocessing batch {}", batch_id));

    ImageBatch batch;
    batch.batch_id = batch_id;
    batch.pixels.resize(width * height);

    // Simulate normalization
    std::mt19937 rng(batch_id);
    std::uniform_real_distribution<float> dist(0.0f, 255.0f);
    for (auto& p : batch.pixels) {
        p = (dist(rng) - 127.5f) / 127.5f; // Normalize to [-1, 1]
    }

    // Ch8.4.5: Simulate compute cost.
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return batch;
}

// Stage 2: Inference (Ch8.3.1: the core compute-bound stage).
InferenceResult inference_stage(const ImageBatch& batch) {
    Logger::instance().debug(TS_FORMAT("Pipeline: inference on batch {}", batch.batch_id));

    InferenceResult result;
    result.batch_id = batch.batch_id;
    result.class_probabilities.resize(10);

    // Simulate a neural network forward pass.
    float sum = std::accumulate(batch.pixels.begin(), batch.pixels.end(), 0.0f);
    for (size_t i = 0; i < 10; ++i) {
        result.class_probabilities[i] = std::abs(std::sin(sum + i * 0.1f));
    }

    // Simulate compute cost (inference is the bottleneck).
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    return result;
}

// Stage 3: Post-processing (Ch8.4.1: result aggregation).
ProcessedResult postprocess_stage(const InferenceResult& result) {
    Logger::instance().debug(TS_FORMAT("Pipeline: post-processing batch {}", result.batch_id));

    ProcessedResult final_result;
    final_result.batch_id = result.batch_id;

    // Find class with highest probability (simulated softmax + argmax).
    auto max_it = std::max_element(result.class_probabilities.begin(),
                                    result.class_probabilities.end());
    final_result.predicted_class = static_cast<int>(
        std::distance(result.class_probabilities.begin(), max_it));
    final_result.confidence = *max_it;

    return final_result;
}

// Ch8.3.2: Run the full pipeline for a single batch.
ProcessedResult run_pipeline(TaskScheduler& scheduler, int batch_id,
                              int width, int height) {
    // Ch8.3.3: Chain stages using TaskScheduler pipeline support.
    // Stage 1 -> Stage 2 -> Stage 3 (continuation passing, Ch4.2.3).
    auto future = scheduler.submit_pipeline<ImageBatch, ProcessedResult>(
        TS_FORMAT("pipeline_batch_{}", batch_id),
        // Stage 1: Preprocess
        [batch_id, width, height]() { return preprocess_stage(batch_id, width, height); },
        // Stage 2+3: Inference + Postprocess (combined for simplicity)
        [](ImageBatch batch) {
            auto inf_result = inference_stage(batch);
            return postprocess_stage(inf_result);
        }
    );

    return future.get();
}

int main() {
    Logger::instance().set_level(LogLevel::INFO);
    Logger::instance().info("=== Example: AI/ML Inference Pipeline ===");

    // Ch8.4.1: Create scheduler with enough threads for pipeline parallelism.
    TaskScheduler scheduler(8);

    constexpr int num_batches = 20;
    constexpr int image_width = 64;
    constexpr int image_height = 64;

    std::cout << TS_FORMAT("Processing {} batches of {}x{} images...\n",
                              num_batches, image_width, image_height);

    auto start = std::chrono::steady_clock::now();

    // Ch8.4.2: Submit all batches for parallel pipeline execution.
    std::vector<std::future<ProcessedResult>> futures;
    for (int i = 0; i < num_batches; ++i) {
        // Each pipeline runs on a separate thread from the pool.
        // Multiple batches are processed in parallel (Ch8.4.6: pipeline parallelism).
        futures.push_back(
            scheduler.submit(TaskPriority::HIGH, TS_FORMAT("batch_{}", i),
                [&scheduler, i] { return run_pipeline(scheduler, i, image_width, image_height); })
        );
    }

    // Ch4.2.4: Collect all results.
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

    // Ch8.4.7: Report throughput metrics.
    std::cout << TS_FORMAT("\nPipeline Stats:\n");
    std::cout << TS_FORMAT("  Total batches: {}\n", num_batches);
    std::cout << TS_FORMAT("  High-confidence: {}/{}\n", correct, num_batches);
    std::cout << TS_FORMAT("  Total time: {}ms\n", elapsed.count());
    std::cout << TS_FORMAT("  Throughput: {:.1f} batches/sec\n",
                              num_batches * 1000.0 / elapsed.count());

    Logger::instance().info("=== Pipeline Example Complete ===");
    return 0;
}
