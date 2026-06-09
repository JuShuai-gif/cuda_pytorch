#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <thread>
#include <vector>

#include "lifecycle.h"
#include "qos_demo.h"
#include "rt_executor.h"
#include "spsc_ringbuffer.h"

using namespace std::chrono;

// ============================================================
// 辅助函数: 获取当前时间 (纳秒)
// ============================================================
static int64_t get_time_ns() {
    auto now = high_resolution_clock::now().time_since_epoch();
    return duration_cast<nanoseconds>(now).count();
}

// ============================================================
// Demo 1: SPSC 环形缓冲区压力测试
// ============================================================
struct StressTestResult {
    double million_ops_per_sec;
    uint64_t total_pushed;
    uint64_t total_popped;
    uint64_t overwrites;
    double push_latency_us;
    double pop_latency_us;
};

static StressTestResult run_spsc_stress_test() {
    printf("\n");
    printf("╔══════════════════════════════════════════╗\n");
    printf("║   Demo 1: SPSC 环形缓冲区压力测试        ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    constexpr size_t BUF_SIZE = 1024;
    constexpr uint64_t TOTAL_ITEMS = 1'000'000;

    SPSCRingBuffer<int64_t, BUF_SIZE> buffer;
    StressTestResult result = {};

    std::atomic<bool> producer_done{false};
    std::atomic<uint64_t> consumed{0};
    std::atomic<uint64_t> overwritten{0};

    // 生产者线程
    auto producer = [&]() {
        auto start = high_resolution_clock::now();
        for (uint64_t i = 1; i <= TOTAL_ITEMS; ++i) {
            // 约 10% 概率使用 overwrite 模式
            if (i % 10 == 0) {
                buffer.push_overwrite(static_cast<int64_t>(i));
                overwritten.fetch_add(1, std::memory_order_relaxed);
            } else {
                while (!buffer.try_push(static_cast<int64_t>(i))) {
                    // 自旋等待消费者
                    std::this_thread::yield();
                }
            }
        }
        auto end = high_resolution_clock::now();
        auto elapsed = duration_cast<nanoseconds>(end - start).count();
        result.push_latency_us = static_cast<double>(elapsed) / 1000.0 / TOTAL_ITEMS;
        result.total_pushed = TOTAL_ITEMS;
        producer_done.store(true);
    };

    // 消费者线程
    auto consumer = [&]() {
        auto start = high_resolution_clock::now();
        int64_t item;
        while (consumed.load() < TOTAL_ITEMS) {
            if (buffer.try_pop(item)) {
                consumed.fetch_add(1, std::memory_order_relaxed);
            } else if (producer_done.load()) {
                // 生产者已完成，检查缓冲区是否真的为空
                if (buffer.empty()) break;
            }
        }
        auto end = high_resolution_clock::now();
        auto elapsed = duration_cast<nanoseconds>(end - start).count();
        uint64_t c = consumed.load();
        if (c > 0) {
            result.pop_latency_us = static_cast<double>(elapsed) / 1000.0 / c;
        }
        result.total_popped = c;
    };

    auto t1_start = high_resolution_clock::now();

    std::thread prod_thread(producer);
    std::thread cons_thread(consumer);

    prod_thread.join();
    cons_thread.join();

    auto t1_end = high_resolution_clock::now();
    auto total_elapsed = duration_cast<nanoseconds>(t1_end - t1_start).count();

    result.million_ops_per_sec = static_cast<double>(TOTAL_ITEMS)
                                 / (static_cast<double>(total_elapsed) / 1e9)
                                 / 1e6;
    result.overwrites = overwritten.load();

    printf("  总条目数:     %lu\n", TOTAL_ITEMS);
    printf("  发布条目:     %lu\n", result.total_pushed);
    printf("  消费条目:     %lu\n", result.total_popped);
    printf("  覆盖写入:     %lu\n", result.overwrites);
    printf("  吞吐量:       %.2f 百万次/秒\n", result.million_ops_per_sec);
    printf("  平均 push:    %.3f μs\n", result.push_latency_us);
    printf("  平均 pop:     %.3f μs\n", result.pop_latency_us);
    printf("  缓冲区使用率: %.1f%%\n",
           100.0 * static_cast<double>(result.total_pushed - result.total_popped)
               / BUF_SIZE);

    return result;
}

// ============================================================
// Demo 2: 实时执行器演示
// ============================================================
static std::vector<RTExecutor::JitterStats> run_rt_executor_demo() {
    printf("\n");
    printf("╔══════════════════════════════════════════╗\n");
    printf("║   Demo 2: 实时执行器抖动测量             ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    RTExecutor executor;

    // 1kHz 控制回调 (模拟电机控制)
    executor.register_callback("motor_control_1khz", 1000,
                               [](uint64_t iteration) -> int64_t {
                                   // 模拟 PID 控制计算 (确定性的计算量)
                                   volatile double pid_output = 0.0;
                                   for (int i = 0; i < 100; ++i) {
                                       pid_output += std::sin(static_cast<double>(i + iteration) * 0.01);
                                   }
                                   auto start = get_time_ns();
                                   // 制造约 50μs 的确定性工作
                                   volatile double x = 0.0;
                                   for (int i = 0; i < 500; ++i) {
                                       x += std::sqrt(static_cast<double>(i));
                                   }
                                   (void)pid_output;
                                   (void)x;
                                   return get_time_ns() - start;
                               });

    // 200Hz 传感器读取回调 (模拟 IMU 数据采集)
    executor.register_callback("imu_sensor_200hz", 200,
                               [](uint64_t /*iter*/) -> int64_t {
                                   auto start = get_time_ns();
                                   // 模拟 SPI 总线读取 (约 30μs 工作)
                                   volatile double x = 0.0;
                                   for (int i = 0; i < 300; ++i) {
                                       x += std::sqrt(static_cast<double>(i));
                                   }
                                   (void)x;
                                   return get_time_ns() - start;
                               });

    // 30Hz 感知回调 (模拟目标检测结果的后处理)
    executor.register_callback("perception_30hz", 30,
                               [](uint64_t /*iter*/) -> int64_t {
                                   auto start = get_time_ns();
                                   // 模拟 NMS 后处理 (约 200μs 工作)
                                   volatile double x = 0.0;
                                   for (int i = 0; i < 2000; ++i) {
                                       x += std::sqrt(static_cast<double>(i));
                                   }
                                   (void)x;
                                   return get_time_ns() - start;
                               });

    printf("  已注册 3 个回调:\n");
    printf("    - motor_control_1khz: 1000 Hz, ~50μs  工作负载\n");
    printf("    - imu_sensor_200hz:   200 Hz,  ~30μs  工作负载\n");
    printf("    - perception_30hz:   30 Hz,   ~200μs 工作负载\n");

    printf("\n  运行 5 秒...\n");
    auto records = executor.run(5.0);

    auto stats = executor.get_stats();

    printf("\n");
    printf("  ┌─────────────────────┬────────┬──────────┬──────────┬──────────┬───────────┬──────────┐\n");
    printf("  │ 回调名称            │ 频率   │ 迭代次数 │ 最小抖动 │ 最大抖动 │ 平均抖动  │ 超时次数 │\n");
    printf("  ├─────────────────────┼────────┼──────────┼──────────┼──────────┼───────────┼──────────┤\n");

    for (const auto &s : stats) {
        printf("  │ %-19s │ %4dHz │ %8lu │ %5dμs  │ %5dμs  │ %5dμs   │ %8lu │\n",
               s.name.c_str(), s.target_hz,
               (unsigned long)s.total_iterations,
               (int)(s.min_jitter_ns / 1000),
               (int)(s.max_jitter_ns / 1000),
               (int)(s.avg_jitter_ns / 1000),
               (unsigned long)s.overrun_count);
    }
    printf("  └─────────────────────┴────────┴──────────┴──────────┴──────────┴───────────┴──────────┘\n");

    return stats;
}

// ============================================================
// Demo 3: 生命周期管理演示
// ============================================================
struct LifecycleDemoResult {
    bool full_cycle_ok;
    bool error_injection_ok;
    bool dependency_order_ok;
};

static LifecycleDemoResult run_lifecycle_demo() {
    printf("\n");
    printf("╔══════════════════════════════════════════╗\n");
    printf("║   Demo 3: 生命周期管理                    ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    LifecycleDemoResult result = {};

    // 3.1 正常状态转换
    printf("--- 3.1 正常的状态转换 ---\n");
    {
        TestLifecycleNode node("control_node", false, false);

        // UNCONFIGURED -> INACTIVE
        auto r = node.configure();
        printf("  configure 结果: %s (状态: %s)\n",
               r.success ? "✅" : "❌", state_to_string(node.get_state()));

        // INACTIVE -> ACTIVE
        r = node.activate();
        printf("  activate 结果:  %s (状态: %s)\n",
               r.success ? "✅" : "❌", state_to_string(node.get_state()));

        // ACTIVE -> INACTIVE
        r = node.deactivate();
        printf("  deactivate 结果: %s (状态: %s)\n",
               r.success ? "✅" : "❌", state_to_string(node.get_state()));

        // INACTIVE -> UNCONFIGURED
        r = node.cleanup();
        printf("  cleanup 结果:   %s (状态: %s)\n",
               r.success ? "✅" : "❌", state_to_string(node.get_state()));

        // -> FINALIZED
        r = node.shutdown();
        printf("  shutdown 结果:  %s (状态: %s)\n",
               r.success ? "✅" : "❌", state_to_string(node.get_state()));

        result.full_cycle_ok = node.get_state() == LifecycleState::FINALIZED;
    }

    // 3.2 错误注入：激活失败
    printf("\n--- 3.2 错误注入: 激活失败 ---\n");
    {
        TestLifecycleNode node("faulty_sensor_node", true, false);

        node.configure();
        // 尝试激活 (将会失败)
        auto r = node.activate();
        printf("  activate 结果:          %s\n", r.success ? "✅" : "❌");
        printf("  失败后的状态:           %s\n", state_to_string(node.get_state()));
        printf("  错误信息:               %s\n", r.error_msg.c_str());

        // 验证节点保持在 INACTIVE 而非 ACTIVE
        result.error_injection_ok = (node.get_state() == LifecycleState::INACTIVE);

        node.shutdown();
    }

    // 3.3 非法状态转换检测
    printf("\n--- 3.3 非法状态转换检测 ---\n");
    {
        TestLifecycleNode node("test_node", false, false);

        // 尝试从 UNCONFIGURED 直接激活 (应被拒绝)
        auto r = node.activate();
        printf("  跳过配置直接激活: %s\n", r.success ? "✅ (BUG!)" : "❌ (正确拒绝)");

        // 尝试从 UNCONFIGURED 清理 (应被拒绝)
        r = node.cleanup();
        printf("  跳过配置直接清理: %s\n", r.success ? "✅ (BUG!)" : "❌ (正确拒绝)");

        node.configure();
        node.activate();
        // 尝试在 ACTIVE 状态清理 (应被拒绝)
        r = node.cleanup();
        printf("  在 ACTIVE 状态清理: %s\n", r.success ? "✅ (BUG!)" : "❌ (正确拒绝)");

        node.deactivate();
        node.shutdown();
    }

    // 3.4 依赖顺序管理
    printf("\n--- 3.4 依赖顺序管理 ---\n");
    {
        LifecycleManager manager;

        TestLifecycleNode hardware("hardware_interface", false, false);
        TestLifecycleNode controller("controller", false, false);
        TestLifecycleNode planner("planner", false, false);

        // 控制器依赖硬件接口，规划器依赖控制器
        manager.add_node(&hardware);
        manager.add_node(&controller, {"hardware_interface"});
        manager.add_node(&planner, {"controller"});

        printf("  依赖关系: hardware_interface <- controller <- planner\n");

        bool ok = manager.activate_all();
        result.dependency_order_ok = ok;

        if (ok) {
            printf("  ✅ 所有节点按依赖顺序激活成功\n");
        }

        manager.deactivate_all();
        manager.cleanup_all();
        manager.shutdown_all();
    }

    return result;
}

// ============================================================
// Demo 4: QoS 管线演示
// ============================================================
struct PipelineDemoResult {
    uint64_t sensor_samples;
    uint64_t perception_results;
    uint64_t control_commands;
    uint64_t deadline_misses;
    int64_t avg_perception_latency_us;
    int64_t avg_control_latency_us;
};

static PipelineDemoResult run_qos_pipeline_demo() {
    printf("\n");
    printf("╔══════════════════════════════════════════╗\n");
    printf("║   Demo 4: QoS 多速率管线                 ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    PipelineDemoResult result = {};

    // 通道定义
    BestEffortChannel<SensorSample, 8> sensor_channel;       // 200Hz 传感器 -> 30Hz 感知
    ReliableChannel<PerceptionResult, 4> perception_channel; // 30Hz 感知 -> 1kHz 控制
    BestEffortChannel<ControlCommand, 8> control_channel;    // 1kHz 控制输出 (latest-is-best)

    // 速率匹配器
    RateMatcher sensor_to_perception(200, 30);
    RateMatcher perception_to_control(30, 1000);

    // 截止时间监控
    DeadlineMonitor sensor_deadline(
        10'000'000LL, // 10ms = 100Hz 的最低到达率
        [](int64_t late_ns) {
            printf("    ⚠ 传感器数据超时 %ld μs\n", late_ns / 1000);
        });

    DeadlineMonitor perception_deadline(
        50'000'000LL, // 50ms = 20Hz 的最低到达率
        [](int64_t late_ns) {
            printf("    ⚠ 感知结果超时 %ld μs\n", late_ns / 1000);
        });

    std::atomic<bool> running{true};
    std::atomic<uint64_t> sensor_seq{0};
    std::atomic<uint64_t> perception_seq{0};
    std::atomic<uint64_t> control_seq{0};

    std::vector<int64_t> perception_latencies;
    std::vector<int64_t> control_latencies;

    // 200Hz 传感器模拟线程
    std::thread sensor_thread([&]() {
        int64_t period_ns = 5'000'000LL; // 200Hz = 5ms
        auto next_wake = high_resolution_clock::now();

        while (running.load()) {
            SensorSample sample;
            sample.timestamp_ns = get_time_ns();
            sample.seq = sensor_seq.fetch_add(1);
            sample.value = std::sin(sample.seq * 0.1);

            sensor_channel.publish(sample);
            sensor_to_perception.source_publish(sample.timestamp_ns);
            sensor_deadline.record_arrival(sample.timestamp_ns);

            next_wake += nanoseconds(period_ns);
            std::this_thread::sleep_until(next_wake);
        }
    });

    // 30Hz 感知模拟线程
    std::thread perception_thread([&]() {
        int64_t period_ns = 33'333'333LL; // 30Hz = ~33.3ms
        auto next_wake = high_resolution_clock::now();

        while (running.load()) {
            SensorSample sensor_data;
            bool has_data = sensor_channel.take_latest(sensor_data);

            if (has_data) {
                // 模拟感知计算
                int64_t start = get_time_ns();

                PerceptionResult pr;
                pr.timestamp_ns = sensor_data.timestamp_ns;
                pr.seq = perception_seq.fetch_add(1);
                pr.num_detections = 5 + static_cast<int>(sensor_data.value * 3);
                pr.confidence = 0.7 + 0.3 * std::abs(sensor_data.value);

                perception_channel.publish(pr);
                perception_to_control.source_publish(pr.timestamp_ns);
                perception_deadline.record_arrival(pr.timestamp_ns);

                int64_t elapsed = get_time_ns() - start;
                perception_latencies.push_back(elapsed);
            }

            next_wake += nanoseconds(period_ns);
            std::this_thread::sleep_until(next_wake);
        }
    });

    // 1kHz 控制模拟线程 (实时)
    std::thread control_thread([&]() {
        int64_t period_ns = 1'000'000LL; // 1kHz = 1ms
        auto next_wake = high_resolution_clock::now();

        while (running.load()) {
            int64_t now = get_time_ns();

            // 检查截止时间
            sensor_deadline.check(now);
            perception_deadline.check(now);

            // 获取最新感知结果 (hold-last 语义)
            PerceptionResult pr;
            bool has_perception = perception_channel.take(pr);

            // 模拟控制计算
            int64_t start = get_time_ns();

            ControlCommand cmd;
            cmd.timestamp_ns = now;
            cmd.seq = control_seq.fetch_add(1);
            cmd.target_velocity = has_perception ? pr.confidence * 10.0 : 0.0;
            cmd.target_position += cmd.target_velocity * 0.001; // 1ms 周期

            control_channel.publish(cmd);

            int64_t elapsed = get_time_ns() - start;
            control_latencies.push_back(elapsed);

            next_wake += nanoseconds(period_ns);
            std::this_thread::sleep_until(next_wake);
        }
    });

    // 运行 5 秒
    printf("  多速率管线运行中:\n");
    printf("    200Hz 传感器 ──(latest-is-best)──▶ 30Hz 感知\n");
    printf("    30Hz 感知  ──(hold-last)──────▶ 1kHz 控制\n");
    printf("    运行 5 秒...\n\n");

    std::this_thread::sleep_for(seconds(5));
    running.store(false);

    sensor_thread.join();
    perception_thread.join();
    control_thread.join();

    // 收集结果
    result.sensor_samples = sensor_seq.load();
    result.perception_results = perception_seq.load();
    result.control_commands = control_seq.load();
    result.deadline_misses = sensor_deadline.miss_count()
                             + perception_deadline.miss_count();

    if (!perception_latencies.empty()) {
        result.avg_perception_latency_us = std::accumulate(
                                               perception_latencies.begin(), perception_latencies.end(), 0LL)
                                           / perception_latencies.size() / 1000;
    }
    if (!control_latencies.empty()) {
        result.avg_control_latency_us = std::accumulate(
                                            control_latencies.begin(), control_latencies.end(), 0LL)
                                        / control_latencies.size() / 1000;
    }

    printf("  ┌──────────────────────┬──────────────┐\n");
    printf("  │ 指标                 │ 值           │\n");
    printf("  ├──────────────────────┼──────────────┤\n");
    printf("  │ 传感器样本数         │ %12lu │\n", result.sensor_samples);
    printf("  │ 感知结果数           │ %12lu │\n", result.perception_results);
    printf("  │ 控制指令数           │ %12lu │\n", result.control_commands);
    printf("  │ 截止时间丢失次数     │ %12lu │\n", result.deadline_misses);
    printf("  │ 感知平均延迟         │ %9ld μs │\n", result.avg_perception_latency_us);
    printf("  │ 控制平均延迟         │ %9ld μs │\n", result.avg_control_latency_us);
    printf("  └──────────────────────┴──────────────┘\n");

    return result;
}

// ============================================================
// JSON 输出
// ============================================================
static void write_metrics_json(const StressTestResult &ring,
                               const std::vector<RTExecutor::JitterStats> &jitter,
                               const LifecycleDemoResult &lifecycle,
                               const PipelineDemoResult &pipeline) {
    std::ofstream f("ros2_realtime_metrics.json");
    if (!f.is_open()) {
        printf("⚠ 无法写入 ros2_realtime_metrics.json\n");
        return;
    }

    f << "{\n";
    f << "  \"spsc_ring_buffer\": {\n";
    f << "    \"throughput_million_ops_per_sec\": " << ring.million_ops_per_sec << ",\n";
    f << "    \"total_pushed\": " << ring.total_pushed << ",\n";
    f << "    \"total_popped\": " << ring.total_popped << ",\n";
    f << "    \"overwrites\": " << ring.overwrites << ",\n";
    f << "    \"avg_push_latency_us\": " << ring.push_latency_us << ",\n";
    f << "    \"avg_pop_latency_us\": " << ring.pop_latency_us << "\n";
    f << "  },\n";

    f << "  \"rt_executor\": {\n";
    f << "    \"callbacks\": [\n";
    for (size_t i = 0; i < jitter.size(); ++i) {
        f << "      {\n";
        f << "        \"name\": \"" << jitter[i].name.c_str() << "\",\n";
        f << "        \"target_hz\": " << jitter[i].target_hz << ",\n";
        f << "        \"iterations\": " << jitter[i].total_iterations << ",\n";
        f << "        \"min_jitter_us\": " << (jitter[i].min_jitter_ns / 1000) << ",\n";
        f << "        \"max_jitter_us\": " << (jitter[i].max_jitter_ns / 1000) << ",\n";
        f << "        \"avg_jitter_us\": " << (jitter[i].avg_jitter_ns / 1000) << ",\n";
        f << "        \"p99_jitter_us\": " << (jitter[i].p99_jitter_ns / 1000) << ",\n";
        f << "        \"overruns\": " << jitter[i].overrun_count << "\n";
        f << "      }" << (i < jitter.size() - 1 ? "," : "") << "\n";
    }
    f << "    ]\n";
    f << "  },\n";

    f << "  \"lifecycle\": {\n";
    f << "    \"full_cycle_ok\": " << (lifecycle.full_cycle_ok ? "true" : "false") << ",\n";
    f << "    \"error_injection_ok\": " << (lifecycle.error_injection_ok ? "true" : "false") << ",\n";
    f << "    \"dependency_order_ok\": " << (lifecycle.dependency_order_ok ? "true" : "false") << "\n";
    f << "  },\n";

    f << "  \"qos_pipeline\": {\n";
    f << "    \"sensor_samples\": " << pipeline.sensor_samples << ",\n";
    f << "    \"perception_results\": " << pipeline.perception_results << ",\n";
    f << "    \"control_commands\": " << pipeline.control_commands << ",\n";
    f << "    \"deadline_misses\": " << pipeline.deadline_misses << ",\n";
    f << "    \"avg_perception_latency_us\": " << pipeline.avg_perception_latency_us << ",\n";
    f << "    \"avg_control_latency_us\": " << pipeline.avg_control_latency_us << "\n";
    f << "  }\n";
    f << "}\n";

    f.close();
    printf("\n✅ 指标已写入 ros2_realtime_metrics.json\n");
}

// ============================================================
// 主入口
// ============================================================
int main() {
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  ROS2 实时控制与通信模式演示                      ║\n");
    printf("║  模拟 ros2_control 基础设施模式                   ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");

    printf("\n本演示模拟 ROS2 机器人系统的中间件/基础设施模式:\n");
    printf("  1. 无锁 SPSC 环形缓冲区 (传感器 → RT 控制)\n");
    printf("  2. 实时执行器架构 (SCHED_FIFO + clock_nanosleep)\n");
    printf("  3. 生命周期管理 (状态机 + 依赖顺序)\n");
    printf("  4. QoS 多速率管线 (200Hz → 30Hz → 1kHz)\n");

    // Demo 1: SPSC 环形缓冲区压力测试
    auto ring_result = run_spsc_stress_test();

    // Demo 2: 实时执行器
    auto jitter_stats = run_rt_executor_demo();

    // Demo 3: 生命周期管理
    auto lifecycle_result = run_lifecycle_demo();

    // Demo 4: QoS 管线
    auto pipeline_result = run_qos_pipeline_demo();

    // 写入 JSON
    write_metrics_json(ring_result, jitter_stats, lifecycle_result, pipeline_result);

    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║  所有演示完成                                     ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");

    return 0;
}
