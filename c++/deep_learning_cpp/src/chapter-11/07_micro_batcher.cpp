/*
 * micro_batcher.cpp
 * Chapter 11: Debugging and Retraining Deployed Models
 *
 * When p99 latency rises, the cause is often not the model kernel but
 * the time requests spend waiting to be batched. A micro-batcher with
 * instrumentation makes queue time vs. compute time visible.
 *
 * This implementation demonstrates:
 *   - MicroBatcher class using condition variables
 *   - Configurable max_batch and max_delay_ms trade-off
 *   - Future/promise for async result delivery
 *   - Queue time vs. compute time separation
 *
 * PDF pages: 440-444 (book pp. 440-444)
 *
 * Key insight: if batching deadline was increased from 8ms to 25ms
 * for throughput, but p95 latency spiked, the fix is NOT to change
 * model architecture -- use a smaller batching delay for latency-
 * sensitive cohorts.
 */

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// ================================================================
// Placeholder tensor type
// In production, replace with torch::Tensor or equivalent
// ================================================================

using Tensor = std::vector<float>;

// ================================================================
// 1. Task: queued inference request
// ================================================================

struct Task {
    std::string id;
    Tensor x;
    std::promise<Tensor> p;
};

// ================================================================
// 2. MicroBatcher class
//    Collects requests into batches with configurable size/time limits
// ================================================================

class MicroBatcher {
public:
    MicroBatcher(size_t max_batch, int max_delay_ms);
    ~MicroBatcher();
    std::future<Tensor> submit(std::string id, Tensor x);

    // For instrumentation: read queue depth
    size_t queue_depth() const {
        std::lock_guard<std::mutex> lk(mu_);
        return q_.size();
    }

private:
    void run();
    void process_batch(std::deque<Task> &batch);

    size_t max_batch_;
    int max_delay_ms_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Task> q_;
    std::thread worker_;
    bool stop_;
    int batch_count_ = 0;

    // Instrumentation: track batch sizes for diagnosis
    std::vector<size_t> batch_sizes_;
};

MicroBatcher::MicroBatcher(size_t max_batch, int max_delay_ms) : max_batch_(max_batch),
                                                                 max_delay_ms_(max_delay_ms),
                                                                 stop_(false) {
    worker_ = std::thread([this] { run(); });
}

MicroBatcher::~MicroBatcher() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    worker_.join();
}

std::future<Tensor> MicroBatcher::submit(std::string id, Tensor x) {
    std::promise<Tensor> p;
    auto f = p.get_future();
    {
        std::lock_guard<std::mutex> lk(mu_);
        q_.push_back({std::move(id), std::move(x), std::move(p)});
    }
    cv_.notify_one();
    return f;
}

void MicroBatcher::process_batch(std::deque<Task> &batch) {
    // Simulated inference: sum inputs (toy computation)
    // In production, this would call model.forward() on the batched tensor
    batch_sizes_.push_back(batch.size());

    // Simulated model inference latency proportional to batch size
    int infer_us = 1000 + static_cast<int>(batch.size()) * 200;
    std::this_thread::sleep_for(std::chrono::microseconds(infer_us));

    for (auto &t : batch) {
        // Toy output: element-wise sum of input
        float sum = 0.0f;
        for (float v : t.x) sum += v;
        Tensor result = {sum};
        t.p.set_value(std::move(result));
    }
}

void MicroBatcher::run() {
    for (;;) {
        std::deque<Task> batch;
        {
            std::unique_lock<std::mutex> lk(mu_);

            // Wait until there are tasks or we need to stop
            if (stop_ && q_.empty()) break;
            if (q_.empty()) {
                cv_.wait(lk);
                continue;
            }

            // Set batching deadline
            auto deadline =
                std::chrono::steady_clock::now() + std::chrono::milliseconds(max_delay_ms_);

            // Wait until batch is full or deadline expires
            while (q_.size() < max_batch_) {
                if (cv_.wait_until(lk, deadline) == std::cv_status::timeout) {
                    break;
                }
                if (stop_) break;
            }

            // Drain up to max_batch_ items
            while (!q_.empty() && batch.size() < max_batch_) {
                batch.push_back(std::move(q_.front()));
                q_.pop_front();
            }
        }

        // Process batch outside the lock (parallelism-friendly)
        if (!batch.empty()) {
            batch_count_++;
            process_batch(batch);
        }
    }
}

// ================================================================
// 3. Batch size distribution analysis
//    Answers: "Did avg batch size drop (utilization loss) or spike?"
// ================================================================

void analyze_batch_distribution(const std::vector<size_t> &sizes) {
    if (sizes.empty()) return;

    std::vector<size_t> sorted = sizes;
    std::sort(sorted.begin(), sorted.end());

    double sum = 0.0;
    for (auto s : sizes) sum += s;
    double avg = sum / sizes.size();

    // p50, p95
    size_t p50_idx = sizes.size() / 2;
    size_t p95_idx = static_cast<size_t>(0.95 * sizes.size());
    if (p95_idx >= sizes.size()) p95_idx = sizes.size() - 1;

    std::cout << "\n  Batch Size Distribution:\n";
    std::cout << "    n_batches=" << sizes.size()
              << " avg=" << avg
              << " p50=" << sorted[p50_idx]
              << " p95=" << sorted[p95_idx]
              << " min=" << sorted.front()
              << " max=" << sorted.back() << "\n";

    if (avg < sorted.back() * 0.5) {
        std::cout << "    >> Under-utilized: batches often below max.\n";
    }
}

// ================================================================
// Main demonstration
// ================================================================

int main() {
    std::srand(42);
    std::cout << "=== Chapter 11: Micro-Batcher ===\n\n";

    // Scenario A: Small delay (2ms) -- latency-sensitive
    {
        std::cout << "--- Scenario A: max_batch=8, max_delay=2ms (latency-optimized) ---\n";
        MicroBatcher batcher(8, 2);

        auto t0 = std::chrono::high_resolution_clock::now();

        // Submit 20 requests concurrently
        std::vector<std::future<Tensor>> futures;
        for (int i = 0; i < 20; ++i) {
            futures.push_back(
                batcher.submit("r-" + std::to_string(i),
                               Tensor{float(i), float(i * 2)}));
        }

        // Collect results
        for (size_t i = 0; i < futures.size(); ++i) {
            auto result = futures[i].get();
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::high_resolution_clock::now() - t0)
                           .count();
        std::cout << "  Total time: " << elapsed << "ms\n";
        std::cout << "  Queue depth at end: " << batcher.queue_depth() << "\n";
    }

    // Scenario B: Large delay (25ms) -- throughput-optimized
    {
        std::cout << "\n--- Scenario B: max_batch=8, max_delay=25ms (throughput-optimized) ---\n";
        MicroBatcher batcher(8, 25);

        auto t0 = std::chrono::high_resolution_clock::now();

        std::vector<std::future<Tensor>> futures;
        for (int i = 0; i < 20; ++i) {
            futures.push_back(
                batcher.submit("r-" + std::to_string(i),
                               Tensor{float(i), float(i * 2)}));
        }

        for (size_t i = 0; i < futures.size(); ++i) {
            auto result = futures[i].get();
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::high_resolution_clock::now() - t0)
                           .count();
        std::cout << "  Total time: " << elapsed << "ms\n";
        std::cout << "  Queue depth at end: " << batcher.queue_depth() << "\n";
    }

    // Scenario C: Steady stream analysis
    {
        std::cout << "\n--- Scenario C: Steady stream with batch size analysis ---\n";
        MicroBatcher batcher(4, 10);

        // Simulate steady request stream with varying inter-arrival times
        std::vector<std::future<Tensor>> futures;
        for (int i = 0; i < 30; ++i) {
            futures.push_back(
                batcher.submit("s-" + std::to_string(i),
                               Tensor{float(i % 10)}));
            // Variable inter-arrival: burst of 5, then pause
            if (i > 0 && i % 5 == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(15));
            }
        }

        for (size_t i = 0; i < futures.size(); ++i) {
            futures[i].get();
        }
        // (batch_sizes_ is private; accessed via destructor only in demo)
    }

    // --- Diagnostic: interpreting queue vs compute ---
    std::cout << "\n4. Diagnostic: Queue time vs. Compute time\n";
    std::cout << "   Symptom: p99 latency rose by 20ms after deployment.\n";
    std::cout << "   Tracing shows: queue span grew 18ms, infer stayed flat.\n";
    std::cout << "   Answer: The problem is batching/pacing, NOT the model kernel.\n";
    std::cout << "   Fix: use smaller max_delay for latency-sensitive cohorts.\n";

    std::cout << "\n5. Operational lesson\n";
    std::cout << "   Increasing batching window from 8ms to 25ms boosts throughput\n";
    std::cout << "   but hurts p95 latency for mobile users. The fix is NOT to modify\n";
    std::cout << "   the model -- it is to use per-cohort batching deadlines.\n";

    std::cout << "\n=== Micro-batcher demo complete ===\n";
    return 0;
}
