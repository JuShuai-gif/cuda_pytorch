// 02_latency_throughput.cpp — 延迟与吞吐量对比分析
// 演示: batch size 对延迟/吞吐量的影响、P99 尾延迟

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <queue>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// ===== 场景: 请求处理队列 =====
// 对比: 实时处理 (每个请求立刻处理) vs 批处理 (攒够一批再处理)

class RequestProcessor {
public:
    // 模式 A: 实时处理 — 低延迟
    double process_single(int request_id) {
        auto start = std::chrono::high_resolution_clock::now();
        // 模拟处理开销
        volatile int work = 0;
        for (int i = 0; i < 1000; ++i) work += i;
        auto elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - start);
        return elapsed.count();
    }

    // 模式 B: 批处理 — 高吞吐量
    double process_batch(const std::vector<int>& requests) {
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < requests.size(); ++i) {
            volatile int work = 0;
            for (int j = 0; j < 1000; ++j) work += j;
        }
        auto elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - start);
        return elapsed.count() / requests.size(); // 平均每请求延迟
    }
};

void demo_latency_vs_throughput() {
    std::cout << "=== 延迟 vs 吞吐量 ===\n\n";

    const int kTotalRequests = 10000;
    RequestProcessor proc;

    // 实时处理
    {
        std::vector<double> latencies;
        latencies.reserve(kTotalRequests);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < kTotalRequests; ++i) {
            double lat = proc.process_single(i);
            latencies.push_back(lat);
        }
        auto total_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);

        std::sort(latencies.begin(), latencies.end());
        double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0)
                     / latencies.size();
        double p99 = latencies[latencies.size() * 99 / 100];

        std::cout << "  实时处理:\n";
        std::cout << "    总耗时:    " << total_time.count() << " ms\n";
        std::cout << "    平均延迟:  " << std::fixed << std::setprecision(1)
                  << avg << " us\n";
        std::cout << "    P99 延迟:  " << p99 << " us\n";
        std::cout << "    吞吐量:    " << std::setprecision(0)
                  << kTotalRequests * 1000.0 / total_time.count()
                  << " req/s\n";
    }

    // 批处理
    {
        const int kBatchSize = 100;
        int num_batches = kTotalRequests / kBatchSize;

        auto start = std::chrono::high_resolution_clock::now();
        for (int b = 0; b < num_batches; ++b) {
            std::vector<int> batch;
            for (int i = 0; i < kBatchSize; ++i) {
                batch.push_back(b * kBatchSize + i);
            }
            proc.process_batch(batch);
        }
        auto total_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);

        std::cout << "\n  批处理 (batch=" << kBatchSize << "):\n";
        std::cout << "    总耗时:    " << total_time.count() << " ms\n";
        std::cout << "    平均延迟:  "
                  << total_time.count() * 1000.0 / kTotalRequests
                  << " us (per req)\n";
        std::cout << "    吞吐量:    " << std::setprecision(0)
                  << kTotalRequests * 1000.0 / total_time.count()
                  << " req/s\n";
        std::cout << "    注意: 批处理的首个请求延迟更大 (需要等凑满一批)\n";
    }
}

// ===== P99 延迟分析 =====
void demo_p99_latency() {
    std::cout << "\n=== P99 尾延迟分析 ===\n";

    std::mutex mtx;
    std::vector<double> latencies;
    latencies.reserve(10000);

    const int kThreads = 4;
    const int kOpsPerThread = 2500;

    // 模拟: 偶尔有慢请求
    auto operation = [&](int tid) {
        auto start = std::chrono::high_resolution_clock::now();

        // 90%: 快请求 (1ms)
        // 10%: 慢请求 (10ms)
        if (tid * 100 + rand() % 100 < 10) {
            std::this_thread::sleep_for(10ms);
        } else {
            std::this_thread::sleep_for(1ms);
        }

        auto elapsed_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - start);
        std::lock_guard lock(mtx);
        latencies.push_back(elapsed_us.count());
    };

    std::vector<std::jthread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < kOpsPerThread; ++i) {
                operation(t);
            }
        });
    }
    threads.clear();

    std::sort(latencies.begin(), latencies.end());
    size_t n = latencies.size();

    double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / n;
    double p50 = latencies[n * 50 / 100];
    double p90 = latencies[n * 90 / 100];
    double p99 = latencies[n * 99 / 100];
    double p999 = latencies[n * 999 / 1000];
    double max = latencies.back();

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  平均:  " << avg << " us\n";
    std::cout << "  P50:   " << p50 << " us\n";
    std::cout << "  P90:   " << p90 << " us\n";
    std::cout << "  P99:   " << p99 << " us\n";
    std::cout << "  P99.9: " << p999 << " us\n";
    std::cout << "  Max:   " << max << " us\n";
    std::cout << "  结论: P99 比平均值更能反映用户真实体验\n";
}

int main() {
    demo_latency_vs_throughput();
    demo_p99_latency();

    std::cout << "\n延迟关注单个请求的速度，吞吐量关注整体的处理能力。\n";
    return 0;
}
