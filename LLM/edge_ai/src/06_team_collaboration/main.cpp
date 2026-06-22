#include "interface.h"
#include "modules.h"
#include "validator.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// ============================================================================
// 将单个模块的验证结果写入 JSON
// ============================================================================
static void write_module_json(std::ofstream &of,
                              const std::string &json_key,
                              const PerformanceContract &contract,
                              const MeasurementBatch &batch,
                              const std::vector<Violation> &violations,
                              bool sla_met,
                              bool is_last) {
    std::vector<int64_t> latencies;
    for (const auto &s : batch.samples) latencies.push_back(s.latency_us);

    int64_t p50, p99, max_val, jitter;
    ContractValidator::compute_stats(latencies, p50, p99, max_val, jitter);

    of << "  \"" << json_key << "\": {\n";
    of << "    \"contract\": {\n";
    of << "      \"latency_p50_us\": " << contract.latency_p50_us << ",\n";
    of << "      \"latency_p99_us\": " << contract.latency_p99_us << ",\n";
    of << "      \"max_latency_us\": " << contract.latency_max_us << ",\n";
    of << "      \"jitter_us\": " << contract.jitter_max_us << ",\n";
    of << "      \"throughput_fps\": " << contract.min_fps << ",\n";
    of << "      \"missed_detections_per_1000\": " << contract.missed_detections_per_1000 << ",\n";
    of << "      \"planning_timeout_count\": " << contract.planning_timeout_count << "\n";
    of << "    },\n";
    of << "    \"measured\": {\n";
    of << "      \"num_frames\": " << batch.samples.size() << ",\n";
    of << "      \"latency_p50_us\": " << p50 << ",\n";
    of << "      \"latency_p99_us\": " << p99 << ",\n";
    of << "      \"max_latency_us\": " << max_val << ",\n";
    of << "      \"jitter_us\": " << jitter << ",\n";
    of << "      \"throughput_fps\": " << std::fixed << std::setprecision(2) << batch.measured_fps << ",\n";
    of << "      \"missed_detections\": " << batch.missed_detections << ",\n";
    of << "      \"planning_timeouts\": " << batch.planning_timeouts << "\n";
    of << "    },\n";
    of << "    \"violations\": [\n";

    bool first_v = true;
    for (const auto &v : violations) {
        if (!v.is_pass) {
            if (!first_v) of << ",\n";
            of << "      {\n";
            of << "        \"metric\": \"" << v.metric_name << "\",\n";
            of << "        \"required\": " << v.required_value << ",\n";
            of << "        \"measured\": " << v.measured_value << ",\n";
            of << "        \"unit\": \"" << v.unit << "\",\n";
            of << "        \"severity\": " << std::fixed << std::setprecision(2) << v.severity() << "\n";
            of << "      }";
            first_v = false;
        }
    }
    of << "\n    ],\n";
    of << "    \"sla_met\": " << (sla_met ? "true" : "false") << "\n";
    of << "  }";
    if (!is_last) of << ",";
    of << "\n";
}

// ============================================================================
// 检查所有违规是否通过
// ============================================================================
static bool all_pass(const std::vector<Violation> &violations) {
    for (const auto &v : violations) {
        if (!v.is_pass) return false;
    }
    return true;
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    std::cout << "============================================================\n";
    std::cout << "  跨团队性能合约验证\n";
    std::cout << "  机器人系统流水线: 感知 -> 规划 -> 控制\n";
    std::cout << "============================================================\n";

    const int num_frames = 100;

    PerceptionModule perception;
    PlanningModule planning;
    ControlModule control;

    MeasurementBatch perception_batch;
    perception_batch.module_name = "PerceptionModule";
    MeasurementBatch planning_batch;
    planning_batch.module_name = "PlanningModule";
    MeasurementBatch control_batch;
    control_batch.module_name = "ControlModule";

    perception_batch.missed_detections = 0;
    perception_batch.planning_timeouts = 0;
    planning_batch.missed_detections = 0;
    planning_batch.planning_timeouts = 0;
    control_batch.missed_detections = 0;
    control_batch.planning_timeouts = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_frames; ++i) {
        // 感知
        PerceptionInput p_in{i, 640, 480};
        PerceptionOutput p_out = perception.process(p_in);
        perception_batch.samples.push_back({p_out.latency_us, i, p_out.timestamp_us});
        if (p_out.num_detections < 10) {
            ++perception_batch.missed_detections;
        }

        // 规划（以检测数为障碍物参考值，上限 800）
        int obstacle_hint = std::min(p_out.num_detections, 80) + 50;
        PlanningInput plan_in{i, obstacle_hint};
        PlanningOutput plan_out = planning.process(plan_in);
        planning_batch.samples.push_back({plan_out.latency_us, i, plan_out.timestamp_us});
        if (plan_out.timed_out) {
            ++planning_batch.planning_timeouts;
        }

        // 控制
        double target_speed = plan_out.trajectory.size() > 5 ? 15.0 : 5.0;
        double target_angle = plan_out.trajectory.size() > 10 ? static_cast<double>(plan_out.trajectory[10].second) / PlanningModule::GRID_SIZE * 2.0 - 1.0 : 0.0;
        ControlInput ctrl_in{i, target_speed, target_angle};
        ControlOutput ctrl_out = control.process(ctrl_in);
        control_batch.samples.push_back({ctrl_out.latency_us, i, ctrl_out.timestamp_us});
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(end_time - start_time).count();
    double fps = num_frames / total_sec;

    perception_batch.measured_fps = fps;
    planning_batch.measured_fps = fps;
    control_batch.measured_fps = fps;

    // 验证每个模块
    auto pc = perception.get_contract();
    auto violations_p = ContractValidator::validate(pc, perception_batch);
    bool perception_sla = all_pass(violations_p);

    auto plc = planning.get_contract();
    auto violations_pl = ContractValidator::validate(plc, planning_batch);
    bool planning_sla = all_pass(violations_pl);

    auto cc = control.get_contract();
    auto violations_c = ContractValidator::validate(cc, control_batch);
    bool control_sla = all_pass(violations_c);

    // 打印摘要
    auto print_module = [](const std::string &name, const MeasurementBatch &batch,
                           const std::vector<Violation> &violations, bool sla) {
        int64_t p50, p99, max_val, jitter;
        std::vector<int64_t> lats;
        for (const auto &s : batch.samples) lats.push_back(s.latency_us);
        ContractValidator::compute_stats(lats, p50, p99, max_val, jitter);

        std::cout << "\n--- " << name << " ---\n";
        std::cout << "  帧数: " << batch.samples.size()
                  << "  FPS: " << std::fixed << std::setprecision(1) << batch.measured_fps << "\n";
        std::cout << "  P50: " << p50 / 1000.0 << " ms  P99: " << p99 / 1000.0
                  << " ms  最大: " << max_val / 1000.0 << " ms  抖动: " << jitter / 1000.0 << " ms\n";
        std::cout << "  漏检: " << batch.missed_detections
                  << "  规划超时: " << batch.planning_timeouts << "\n";
        std::cout << "  违规: ";
        int fail_count = 0;
        for (const auto &v : violations)
            if (!v.is_pass) ++fail_count;
        if (fail_count == 0) {
            std::cout << "无  SLA: " << (sla ? "通过" : "失败") << "\n";
        } else {
            std::cout << fail_count << "  SLA: " << (sla ? "通过" : "失败") << "\n";
            for (const auto &v : violations) {
                if (!v.is_pass) {
                    std::cout << "    - " << v.metric_name << ": "
                              << v.measured_value << " vs " << v.required_value
                              << " " << v.unit << "\n";
                }
            }
        }
    };

    print_module("PerceptionModule", perception_batch, violations_p, perception_sla);
    print_module("PlanningModule", planning_batch, violations_pl, planning_sla);
    print_module("ControlModule", control_batch, violations_c, control_sla);

    // 端到端
    std::vector<int64_t> e2e_latencies;
    size_t min_frames = std::min({perception_batch.samples.size(),
                                  planning_batch.samples.size(),
                                  control_batch.samples.size()});
    for (size_t i = 0; i < min_frames; ++i) {
        e2e_latencies.push_back(
            perception_batch.samples[i].latency_us + planning_batch.samples[i].latency_us + control_batch.samples[i].latency_us);
    }
    int64_t e2e_p50, e2e_p99, e2e_max, e2e_jitter;
    ContractValidator::compute_stats(e2e_latencies, e2e_p50, e2e_p99, e2e_max, e2e_jitter);
    double e2e_mean = std::accumulate(e2e_latencies.begin(), e2e_latencies.end(), 0.0) / e2e_latencies.size();

    std::cout << "\n--- 端到端流水线 ---\n";
    std::cout << "  P50: " << e2e_p50 / 1000.0 << " ms  P99: " << e2e_p99 / 1000.0
              << " ms  最大: " << e2e_max / 1000.0 << " ms\n";
    std::cout << "  均值: " << e2e_mean / 1000.0 << " ms  吞吐量: "
              << std::fixed << std::setprecision(1) << fps << " FPS\n";

    // 确定整体端到端 SLA
    bool e2e_sla_met = perception_sla && planning_sla && control_sla;

    // 写入 JSON 报告
    {
        std::ofstream of("contract_validation_report.json");
        of << "{\n";
        write_module_json(of, "perception", pc, perception_batch, violations_p, perception_sla, false);
        write_module_json(of, "planning", plc, planning_batch, violations_pl, planning_sla, false);
        write_module_json(of, "control", cc, control_batch, violations_c, control_sla, false);
        of << "  \"e2e\": {\n";
        of << "    \"sla_met\": " << (e2e_sla_met ? "true" : "false") << ",\n";
        of << "    \"p50_us\": " << e2e_p50 << ",\n";
        of << "    \"p99_us\": " << e2e_p99 << ",\n";
        of << "    \"max_us\": " << e2e_max << ",\n";
        of << "    \"mean_us\": " << std::fixed << std::setprecision(1) << e2e_mean << ",\n";
        of << "    \"throughput_fps\": " << std::fixed << std::setprecision(2) << fps << "\n";
        of << "  }\n";
        of << "}\n";
        of.close();
        std::cout << "\n报告已写入 contract_validation_report.json\n";
    }

    std::cout << "\n完成。\n";
    return 0;
}
