// Chapter 6.2 & 4.1: Classic Producer-Consumer Example
// Demonstrates the classic producer-consumer pattern using:
//   - TaskQueue with condition_variable (Ch4.1 + Ch6.2)
//   - PriorityTaskQueue with multiple priority levels (Ch6.3)
//   - Multiple producers and consumers (Ch8.4: load balancing)
//   - Stop token for graceful shutdown (Ch9.2)
//
// Scenario: Data ingestion pipeline where producers read sensor data
// and consumers process it through a task queue.

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

// Ch6.2.4: Data item produced by sensors.
struct SensorData {
    int sensor_id;
    int sequence;
    double value;
    std::chrono::steady_clock::time_point timestamp;
};

int main() {
    Logger::instance().set_level(LogLevel::INFO);
    Logger::instance().info("=== Example: Producer-Consumer Pattern ===");

    // Ch6.2: Thread-safe queue for sensor data.
    TaskQueue<SensorData> data_queue;

    // Ch9.2.1: Stop token for graceful shutdown.
    stop_source stop_src;
    auto stop_tok = stop_src.get_token();

    constexpr int num_producers = 4;
    constexpr int num_consumers = 3;
    constexpr int items_per_producer = 50;

    std::atomic<int> total_produced{0};
    std::atomic<int> total_consumed{0};
    std::atomic<double> sum_values{0.0};

    // Ch8.4.1: Multiple producer threads.
    std::vector<std::jthread> producers;
    for (int p = 0; p < num_producers; ++p) {
        producers.emplace_back([&, sensor_id = p, stop_tok]() {
            std::mt19937 rng(sensor_id * 100);
            std::uniform_real_distribution<double> dist(0.0, 100.0);

            for (int seq = 0; seq < items_per_producer; ++seq) {
                // Ch9.2.4: Interruption point check.
                stop_tok.interruption_point();

                SensorData data{
                    .sensor_id = sensor_id,
                    .sequence = seq,
                    .value = dist(rng),
                    .timestamp = std::chrono::steady_clock::now()
                };

                // Ch6.2.1: Push to thread-safe queue.
                data_queue.push(data);
                total_produced.fetch_add(1);

                // Simulate sensor read interval.
                std::this_thread::sleep_for(std::chrono::milliseconds(2 + rand() % 5));
            }

            Logger::instance().debug(TS_FORMAT(
                "Producer {} finished ({} items)", sensor_id, items_per_producer));
        });
    }

    // Ch8.4.2: Multiple consumer threads.
    std::vector<std::jthread> consumers;
    for (int c = 0; c < num_consumers; ++c) {
        consumers.emplace_back([&, consumer_id = c, stop_tok]() mutable {
            while (!stop_tok.stop_requested()) {
                // Ch6.2.3: Wait with timeout to periodically check stop flag.
                auto item = data_queue.wait_and_pop_for(std::chrono::milliseconds(50));
                if (item) {
                    sum_values.fetch_add(item->value, std::memory_order_relaxed);
                    total_consumed.fetch_add(1);

                    Logger::instance().debug(TS_FORMAT(
                        "Consumer {} processed sensor={} seq={} value={:.2f}",
                        consumer_id, item->sensor_id, item->sequence, item->value));
                }

                // Ch9.2.2: Exit if all items consumed.
                if (total_consumed.load() >= items_per_producer * num_producers) {
                    break;
                }
            }
        });
    }

    // Ch2.2: Wait for all producers to finish.
    for (auto& t : producers) {
        t.join();
    }

    // Ch9.2.1: Signal consumers to stop.
    Logger::instance().info("All producers finished. Draining queue...");

    // Wait for remaining items to be consumed.
    while (total_consumed.load() < total_produced.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    stop_src.request_stop();
    data_queue.notify_all();

    // Ch2.3: Wait for consumers to finish.
    for (auto& t : consumers) {
        t.join();
    }

    // Ch8.4.7: Report statistics.
    double avg_value = sum_values.load() / total_consumed.load();
    std::cout << TS_FORMAT("\nProducer-Consumer Statistics:\n");
    std::cout << TS_FORMAT("  Total produced: {}\n", total_produced.load());
    std::cout << TS_FORMAT("  Total consumed: {}\n", total_consumed.load());
    std::cout << TS_FORMAT("  Average value:  {:.2f}\n", avg_value);
    std::cout << TS_FORMAT("  Producers: {}, Consumers: {}\n",
                              num_producers, num_consumers);

    Logger::instance().info("=== Producer-Consumer Example Complete ===");
    return 0;
}
