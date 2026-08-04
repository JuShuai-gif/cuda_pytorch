// Ch8.5：AI/ML 批量推理调度示例
// 模拟一个真实的 AI 推理服务器，包含：
//   - 基于优先级的任务调度（Ch6.3）
//   - 动态批量大小的批处理（Ch8.2）
//   - 结果缓存（Ch3.3.2）
//   - 定时健康检查（Ch9.2）
//   - 负载指标和监控（Ch11）

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

// Ch8.5.1：模拟的 ML 模型，服务一批输入。
// 推理模型类：封装推理请求和响应
class InferenceModel {
public:
    // 推理请求：包含请求ID、输入数据和优先级
    struct Request {
        int request_id;
        std::string input_data;
        TaskPriority priority;
    };

    // 推理响应：包含请求ID、结果和延迟
    struct Response {
        int request_id;
        std::string result;
        std::chrono::microseconds latency;
    };

    // Ch3.3.2：使用缓存避免冗余推理。
    // 执行推理：先查缓存，未命中则执行计算
    Response infer(const Request& req, ConcurrentCache<int, std::string>& cache) {
        // Ch3.3.2：先检查缓存（shared_lock 读取）。
        if (auto cached = cache.get(req.request_id)) {
            total_cache_hits_.fetch_add(1);
            return {req.request_id, *cached, std::chrono::microseconds(0)};
        }

        auto start = std::chrono::steady_clock::now();

        // 模拟模型推理：类哈希操作。
        std::string result = TS_FORMAT("inference_result_{}", req.request_id);
        for (int i = 0; i < 500; ++i) {
            result += std::to_string(i % 10);
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start);

        // Ch3.3.1：缓存结果（unique_lock 写入）。
        cache.put(req.request_id, result);

        total_inferences_.fetch_add(1);
        return {req.request_id, result, elapsed};
    }

    // 统计信息：缓存命中次数和推理次数
    [[nodiscard]] size_t cache_hits() const { return total_cache_hits_.load(); }
    [[nodiscard]] size_t inferences() const { return total_inferences_.load(); }

private:
    std::atomic<size_t> total_cache_hits_{0};   // 缓存命中计数
    std::atomic<size_t> total_inferences_{0};   // 实际推理计数
};

// Ch8.5.3：使用线程安全计数器进行负载监控（Ch5.3.3）。
// 服务器指标：各种原子计数器
struct ServerMetrics {
    std::atomic<size_t> requests_submitted{0};    // 已提交请求数
    std::atomic<size_t> requests_completed{0};    // 已完成请求数
    std::atomic<size_t> requests_failed{0};       // 失败请求数
    std::atomic<size_t> high_priority_served{0};  // 高优先级服务数

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
    Logger::instance().info("=== 示例：AI 推理批量调度 ===");

    // Ch9.1：创建与硬件匹配工作线程数的调度器。
    TaskScheduler scheduler(std::thread::hardware_concurrency(), 512);
    InferenceModel model;
    ConcurrentCache<int, std::string> inference_cache(256);
    ServerMetrics metrics;

    // Ch8.5.1：定时指标报告器。
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

    // Ch8.5.2：生成具有不同优先级的随机推理请求。
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> prio_dist(0, 3);

    constexpr int total_requests = 200;

    std::cout << TS_FORMAT("Submitting {} inference requests...\n", total_requests);

    // Ch8.2.1：将请求作为批量任务提交。
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
        }, 0); // request_id 从 0 开始

    // Ch4.2.4：等待所有请求完成。
    for (auto& f : batch_futures) {
        f.get();
    }

    // Ch9.2.1：停止定时任务。
    metrics_stop.request_stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    std::cout << "\n\n";
    metrics.print();

    Logger::instance().info("=== 推理批量示例完成 ===");
    return 0;
}
