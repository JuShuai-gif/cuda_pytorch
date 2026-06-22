#include "qos_demo.h"

#include <cstdio>
#include <vector>

// 在头文件中已完整实现 (模板类 + 简单类)
// 此文件保留用于未来扩展 (如集成测试函数)

namespace qos_demo {

// 运行多速率管线演示
struct PipelineMetrics {
    uint64_t sensor_samples;
    uint64_t perception_outputs;
    uint64_t control_commands;
    uint64_t deadline_misses;
    int64_t max_perception_latency_ns;
    int64_t max_control_latency_ns;
};

PipelineMetrics run_pipeline(double /*duration_seconds*/) {
    // 实现放在 main.cpp 中以便直接访问所有类型
    PipelineMetrics m = {};
    return m;
}

} // namespace qos_demo
