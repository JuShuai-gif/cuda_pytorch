// Ch6.2 & Ch4.1：经典生产者-消费者示例
// 使用以下机制演示经典的生产者-消费者模式：
//   - 带 condition_variable 的 TaskQueue（Ch4.1 + Ch6.2）
//   - 带多优先级的 PriorityTaskQueue（Ch6.3）
//   - 多生产者和多消费者（Ch8.4：负载均衡）
//   - 用于优雅关闭的停止令牌（Ch9.2）
//
// 场景：数据摄取流水线，生产者读取传感器数据，
// 消费者通过任务队列处理数据。

#include "task_scheduler/task_queue.hpp"
#include "task_scheduler/priority_task_queue.hpp"
#include "task_scheduler/stop_token.hpp"
#include "task_scheduler/logger.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "task_scheduler/format_compat.hpp"
#include <atomic>

using namespace task_scheduler;

// Ch6.2.4：传感器生产的数据项。
// 传感器数据结构
struct SensorData {
    int sensor_id;
    int sequence;
    double value;
    std::chrono::steady_clock::time_point timestamp;
};

int main() {
    Logger::instance().set_level(LogLevel::INFO);
    Logger::instance().info("=== 示例：生产者-消费者模式 ===");

    // Ch6.2：用于传感器数据的线程安全队列。
    TaskQueue<SensorData> data_queue;

    // Ch9.2.1：用于优雅关闭的停止令牌。
    stop_source stop_src;
    auto stop_tok = stop_src.get_token();

    constexpr int num_producers = 4;      // 生产者数量
    constexpr int num_consumers = 3;      // 消费者数量
    constexpr int items_per_producer = 50; // 每个生产者生产的项数

    std::atomic<int> total_produced{0};
    std::atomic<int> total_consumed{0};
    std::atomic<double> sum_values{0.0};

    // Ch8.4.1：多个生产者线程。
    std::vector<std::jthread> producers;
    for (int p = 0; p < num_producers; ++p) {
        producers.emplace_back([&, sensor_id = p, stop_tok]() {
            std::mt19937 rng(sensor_id * 100);
            std::uniform_real_distribution<double> dist(0.0, 100.0);

            for (int seq = 0; seq < items_per_producer; ++seq) {
                // Ch9.2.4：中断点检查。
                stop_tok.interruption_point();

                SensorData data{
                    .sensor_id = sensor_id,
                    .sequence = seq,
                    .value = dist(rng),
                    .timestamp = std::chrono::steady_clock::now()
                };

                // Ch6.2.1：推送到线程安全队列。
                data_queue.push(data);
                total_produced.fetch_add(1);

                // 模拟传感器读取间隔。
                std::this_thread::sleep_for(std::chrono::milliseconds(2 + rand() % 5));
            }

            Logger::instance().debug(TS_FORMAT(
                "Producer {} finished ({} items)", sensor_id, items_per_producer));
        });
    }

    // Ch8.4.2：多个消费者线程。
    std::vector<std::jthread> consumers;
    for (int c = 0; c < num_consumers; ++c) {
        consumers.emplace_back([&, consumer_id = c, stop_tok]() mutable {
            while (!stop_tok.stop_requested()) {
                // Ch6.2.3：带超时等待以定期检查停止标志。
                auto item = data_queue.wait_and_pop_for(std::chrono::milliseconds(50));
                if (item) {
                    sum_values.fetch_add(item->value, std::memory_order_relaxed);
                    total_consumed.fetch_add(1);

                    Logger::instance().debug(TS_FORMAT(
                        "Consumer {} processed sensor={} seq={} value={:.2f}",
                        consumer_id, item->sensor_id, item->sequence, item->value));
                }

                // Ch9.2.2：如果已消费完所有项则退出。
                if (total_consumed.load() >= items_per_producer * num_producers) {
                    break;
                }
            }
        });
    }

    // Ch2.2：等待所有生产者完成。
    for (auto& t : producers) {
        t.join();
    }

    // Ch9.2.1：通知消费者停止。
    Logger::instance().info("All producers finished. Draining queue...");

    // 等待剩余项被消费。
    while (total_consumed.load() < total_produced.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    stop_src.request_stop();
    data_queue.notify_all();

    // Ch2.3：等待所有消费者完成。
    for (auto& t : consumers) {
        t.join();
    }

    // Ch8.4.7：报告统计信息。
    double avg_value = sum_values.load() / total_consumed.load();
    std::cout << TS_FORMAT("\nProducer-Consumer Statistics:\n");
    std::cout << TS_FORMAT("  Total produced: {}\n", total_produced.load());
    std::cout << TS_FORMAT("  Total consumed: {}\n", total_consumed.load());
    std::cout << TS_FORMAT("  Average value:  {:.2f}\n", avg_value);
    std::cout << TS_FORMAT("  Producers: {}, Consumers: {}\n",
                              num_producers, num_consumers);

    Logger::instance().info("=== 生产者-消费者示例完成 ===");
    return 0;
}
