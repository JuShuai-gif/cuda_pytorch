// Chapter 8.5: AI/ML Batch Inference Scheduling Example
// Simulates a real-world AI inference server with:
//   - Priority-based task scheduling (Ch6.3)
//   - Batch processing with dynamic batch sizing (Ch8.2)
//   - Result caching (Ch3.3.2)
//   - Periodic health checks (Ch9.2)
//   - Load metrics and monitoring (Ch11)

#include "task_scheduler/task_scheduler.hpp"
#include "task_scheduler/logger.hpp"
#include "task_scheduler/concurrent_cache.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "task_scheduler/format_compat.hpp"
#include <atomic>
#include <map>

using namespace task_scheduler;

// Ch8.5.1: Simulated ML model serving a batch of inputs.
class InferenceModel {
public:
    struct Request {
        int request_id;
        std::string input_data;
        TaskPriority priority;
    };

    struct Response {
        int request_id;
        std::string result;
        std::chrono::microseconds latency;
    };

    // Ch3.3.2: Use cache to avoid redundant inference.
    Response infer(const Request& req, ConcurrentCache<int, std::string>& cache) {
        // Ch3.3.2: Check cache first (shared_lock read).
        if (auto cached = cache.get(req.request_id)) {
            total_cache_hits_.fetch_add(1);
            return {req.request_id, *cached, std::chrono::microseconds(0)};
        }

        auto start = std::chrono::steady_clock::now();

        // Simulate model inference: hash-like operation.
        std::string result = TS_FORMAT("inference_result_{}", req.request_id);
        for (int i = 0; i < 500; ++i) {
            result += std::to_string(i % 10);
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start);

        // Ch3.3.1: Cache result (unique_lock write).
        cache.put(req.request_id, result);

        total_inferences_.fetch_add(1);
        return {req.request_id, result, elapsed};
    }

    [[nodiscard]] size_t cache_hits() const { return total_cache_hits_.load(); }
    [[nodiscard]] size_t inferences() const { return total_inferences_.load(); }

private:
    std::atomic<size_t> total_cache_hits_{0};
    std::atomic<size_t> total_inferences_{0};
};

// Ch8.5.3: Load monitoring with thread-safe counters (Ch5.3.3).
struct ServerMetrics {
    std::atomic<size_t> requests_submitted{0};
    std::atomic<size_t> requests_completed{0};
    std::atomic<size_t> requests_failed{0};
    std::atomic<size_t> high_priority_served{0};

    void print() const {
        std::cout << TS_FORMAT(
            "\nServer Metrics:\n"
            "  Submitted:  {}\n"
            "  Completed:  {}\n"
            "  Failed:     {}\n"
            "  High Pri:   {}\n",
            requests_submitted.load(),
            requests_completed.load(),
            requests_failed.load(),
            high_priority_served.load());
    }
};

int main() {
    Logger::instance().set_level(LogLevel::INFO);
    Logger::instance().info("=== Example: AI Inference Batch Scheduling ===");

    // Ch9.1: Create scheduler with worker threads matching hardware.
    TaskScheduler scheduler(std::thread::hardware_concurrency(), 512);
    InferenceModel model;
    ConcurrentCache<int, std::string> inference_cache(256);
    ServerMetrics metrics;

    // Ch8.5.1: Periodic metrics reporter.
    auto metrics_stop = scheduler.schedule_periodic(
        [&metrics, &model] {
            std::cout << TS_FORMAT(
                "\r[{}] Submitted: {}, Completed: {}, Cache Hits: {}, Inferences: {}  ",
                std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count() % 10000,
                metrics.requests_submitted.load(),
                metrics.requests_completed.load(),
                model.cache_hits(),
                model.inferences()) << std::flush;
        },
        std::chrono::milliseconds(100),
        TaskPriority::LOW,
        "metrics_reporter"
    );

    // Ch8.5.2: Generate random inference requests with varying priorities.
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> prio_dist(0, 3);

    constexpr int total_requests = 200;

    std::cout << TS_FORMAT("Submitting {} inference requests...\n", total_requests);

    // Ch8.2.1: Submit requests as batch tasks.
    auto batch_futures = scheduler.submit_batch(
        TaskPriority::NORMAL, "inference_request", total_requests,
        [&model, &inference_cache, &metrics, &rng, &prio_dist](int req_id) {
            metrics.requests_submitted.fetch_add(1);

            InferenceModel::Request req;
            req.request_id = req_id;
            req.input_data = TS_FORMAT("input_{}", req_id);
            req.priority = static_cast<TaskPriority>(prio_dist(rng));

            try {
                auto resp = model.infer(req, inference_cache);
                metrics.requests_completed.fetch_add(1);

                if (req.priority <= TaskPriority::HIGH) {
                    metrics.high_priority_served.fetch_add(1);
                }
            } catch (const std::exception& e) {
                metrics.requests_failed.fetch_add(1);
                Logger::instance().error(
                    TS_FORMAT("Request {} failed: {}", req_id, e.what()));
            }
        }, 0); // request_id starts from 0

    // Ch4.2.4: Wait for all requests to complete.
    for (auto& f : batch_futures) {
        f.get();
    }

    // Ch9.2.1: Stop periodic tasks.
    metrics_stop.request_stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    std::cout << "\n\n";
    metrics.print();

    Logger::instance().info("=== Inference Batch Example Complete ===");
    return 0;
}
