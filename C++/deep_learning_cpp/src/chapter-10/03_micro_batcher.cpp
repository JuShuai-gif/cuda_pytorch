/*
 * 03_micro_batcher.cpp
 * Chapter 10: Model Deployment and Inference Optimization
 *
 * Micro-batch scheduler for production inference.
 *
 * Core design:
 *   - Bounded queue: rejects requests when full (fast-fail, no unbounded growth)
 *   - Dual trigger: execute when batch reaches max_batch OR window expires
 *   - Promise/Future: each request gets a future for its result
 *   - Single worker thread: sequential tensor stack -> forward -> split -> resolve
 *
 * Key parameters:
 *   max_batch (e.g. 16): maximum batch size per inference call
 *   max_delay_ms (e.g. 8ms): maximum wait time to collect a batch
 *   queue_capacity (e.g. 512): reject if queue reaches this size
 *
 * Production considerations:
 *   - Separate I/O threads from inference thread
 *   - One micro-batcher per GPU (or per model replica)
 *   - CUDA Graphs can eliminate kernel launch overhead for fixed shapes
 *   - Monitor: queue depth, batch size distribution, p50/p95/p99 latency
 */

#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <future>
#include <atomic>
#include <chrono>
#include <vector>

// ----------------------------------------------------------------
// Inference Task: a single request submitted by a client
// ----------------------------------------------------------------
struct InferenceTask {
    torch::Tensor sample; // preprocessed input (e.g. [C, H, W])
    std::promise<torch::Tensor> promise;
};

// ----------------------------------------------------------------
// Simple inference model (wraps a linear layer for demo purposes)
// ----------------------------------------------------------------
struct DemoModel : torch::nn::Module {
    torch::nn::Linear fc{nullptr};

    DemoModel(int input_dim, int output_dim) {
        fc = register_module("fc", torch::nn::Linear(input_dim, output_dim));
    }

    torch::Tensor forward(torch::Tensor x) {
        return fc->forward(x);
    }
};

// ----------------------------------------------------------------
// Micro-batch Scheduler
//
// Accepts individual requests, accumulates into batches, executes
// inference, and distributes results back via futures.
// ----------------------------------------------------------------
class MicroBatcher {
public:
    MicroBatcher(std::shared_ptr<DemoModel> model,
                 int max_batch,
                 int max_delay_ms,
                 size_t queue_capacity) : model_(model),
                                          max_batch_(max_batch),
                                          max_delay_(std::chrono::milliseconds(max_delay_ms)),
                                          cap_(queue_capacity),
                                          stop_(false) {
        model_->eval();
        worker_ = std::thread(&MicroBatcher::run, this);
    }

    ~MicroBatcher() {
        shutdown();
    }

    // Submit a single preprocessed sample. Returns a future for the result.
    // Returns an invalid future (throws exception) if the queue is full.
    std::future<torch::Tensor> submit(torch::Tensor sample) {
        std::promise<torch::Tensor> promise;
        auto future = promise.get_future();

        {
            std::lock_guard<std::mutex> lock(mu_);
            if (queue_.size() >= cap_) {
                // Fast-fail: set exception on promise so caller knows
                promise.set_exception(
                    std::make_exception_ptr(
                        std::runtime_error("Server busy: queue full")));
                return future;
            }
            queue_.push_back({sample, std::move(promise)});
        }
        cv_.notify_one();
        return future;
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

    // Metrics for observability
    struct Stats {
        size_t total_processed = 0;
        size_t total_batches = 0;
        double avg_batch_size = 0.0;
        size_t queue_depth = 0;
    };

    Stats stats() const {
        std::lock_guard<std::mutex> lock(mu_);
        return {total_processed_, total_batches_,
                total_batches_ > 0 ? (double)total_processed_ / total_batches_ : 0.0,
                queue_.size()};
    }

private:
    void run() {
        while (true) {
            std::deque<InferenceTask> batch;
            {
                std::unique_lock<std::mutex> lock(mu_);

                // Wait for at least one request or stop signal
                cv_.wait_for(lock, max_delay_, [this]() {
                    return !queue_.empty() || stop_;
                });

                if (stop_ && queue_.empty()) break;

                // Drain up to max_batch requests
                auto start = std::chrono::steady_clock::now();
                while (!queue_.empty() && batch.size() < (size_t)max_batch_) {
                    batch.push_back(std::move(queue_.front()));
                    queue_.pop_front();

                    // Check if time window expired (if batch > 1 already)
                    auto elapsed = std::chrono::steady_clock::now() - start;
                    if (batch.size() > 1 && elapsed >= std::chrono::milliseconds(max_delay_.count() / 2)) {
                        break;
                    }
                }
            }

            if (batch.empty()) continue;

            // Stack samples into a single tensor: [N, input_dim]
            std::vector<torch::Tensor> samples;
            samples.reserve(batch.size());
            for (auto &task : batch) {
                // Ensure each sample is [1, input_dim]
                samples.push_back(task.sample.unsqueeze(0));
            }
            auto stacked = torch::cat(samples, /*dim=*/0);

            // Inference (no_grad for efficiency)
            torch::Tensor out;
            {
                torch::NoGradGuard ng;
                out = model_->forward(stacked);
            }

            // Split results and resolve promises
            for (size_t i = 0; i < batch.size(); i++) {
                auto row = out[i]; // logits for this sample
                batch[i].promise.set_value(row);
            }

            // Update stats
            total_processed_ += batch.size();
            total_batches_++;
        }
    }

    std::shared_ptr<DemoModel> model_;
    const int max_batch_;
    const std::chrono::milliseconds max_delay_;
    const size_t cap_;

    std::deque<InferenceTask> queue_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::atomic<bool> stop_;
    std::thread worker_;

    size_t total_processed_ = 0;
    size_t total_batches_ = 0;
};

// ----------------------------------------------------------------
// Helper: generate random preprocessed sample [input_dim]
// ----------------------------------------------------------------
torch::Tensor makeSample(int input_dim) {
    return torch::randn({input_dim});
}

// ----------------------------------------------------------------
// Demo: Submit burst of requests, check latency distribution
// ----------------------------------------------------------------
int main() {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "=== Micro-Batcher Demo ===\n\n";

    int input_dim = 64;
    int output_dim = 10;

    auto model = std::make_shared<DemoModel>(input_dim, output_dim);

    int max_batch = 16;
    int max_delay_ms = 8;
    size_t queue_cap = 32;

    MicroBatcher batcher(model, max_batch, max_delay_ms, queue_cap);

    std::cout << "Configuration:\n";
    std::cout << "  max_batch       = " << max_batch << "\n";
    std::cout << "  max_delay_ms    = " << max_delay_ms << "\n";
    std::cout << "  queue_capacity  = " << queue_cap << "\n\n";

    // Submit a burst of requests
    int num_requests = 100;
    std::vector<std::future<torch::Tensor>> futures;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_requests; i++) {
        try {
            auto f = batcher.submit(makeSample(input_dim));
            futures.push_back(std::move(f));
        } catch (const std::exception &e) {
            std::cerr << "Request " << i << " rejected: " << e.what() << "\n";
        }
    }

    // Collect results
    int success = 0;
    std::vector<double> latencies;
    for (int i = 0; i < (int)futures.size(); i++) {
        try {
            auto t_req = std::chrono::high_resolution_clock::now();
            auto result = futures[i].get();
            auto t_end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t_end - t_req).count();
            latencies.push_back(ms);
            success++;
        } catch (const std::exception &e) {
            std::cerr << "Future " << i << " error: " << e.what() << "\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Sort latencies for percentiles
    std::sort(latencies.begin(), latencies.end());
    double p50 = latencies[latencies.size() / 2];
    double p95 = latencies[static_cast<size_t>(latencies.size() * 0.95)];
    double p99 = latencies[static_cast<size_t>(latencies.size() * 0.99)];

    std::cout << "Results:\n";
    std::cout << "  Requests submitted: " << num_requests << "\n";
    std::cout << "  Successful: " << success << "\n";
    std::cout << "  Total wall time: " << total_ms << " ms\n";
    std::cout << "  Throughput: " << (success * 1000.0 / total_ms) << " req/s\n";
    std::cout << "  p50 latency: " << p50 << " ms\n";
    std::cout << "  p95 latency: " << p95 << " ms\n";
    std::cout << "  p99 latency: " << p99 << " ms\n";

    auto st = batcher.stats();
    std::cout << "  Avg batch size: " << st.avg_batch_size << "\n";
    std::cout << "  Total batches: " << st.total_batches << "\n";

    batcher.shutdown();

    std::cout << "\n--- Key Points ---\n";
    std::cout << "1. Bounded queue prevents memory exhaustion.\n";
    std::cout << "2. max_delay_ms balances throughput vs tail latency.\n";
    std::cout << "3. Futures decouple submission from result collection.\n";
    std::cout << "4. Warm-up + fixed shapes enable CUDA Graphs in production.\n";

    return 0;
}
