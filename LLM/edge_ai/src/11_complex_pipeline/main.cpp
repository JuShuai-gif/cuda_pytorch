#include "pipeline_config.h"
#include "latency_stats.h"
#include "pipeline_executor.h"

#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

static void print_usage(const char *prog) {
    std::cout << "用法：" << prog << " [选项]\n"
              << "自动驾驶流水线模拟器（七阶段）\n\n"
              << "选项：\n"
              << "  --mode <sequential|pipelined>  执行模式（默认：pipelined）\n"
              << "  --depth <N>                     流水线深度（默认：3）\n"
              << "  --frames <N>                    帧数（默认：100）\n"
              << "  --seed <N>                      随机种子（默认：42）\n"
              << "  --verbose                       打印每帧详情\n"
              << "  --help                          显示此帮助信息\n";
}

int main(int argc, char *argv[]) {
    PipelineConfig cfg;
    std::string mode = "pipelined";

    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--depth" && i + 1 < argc) {
            cfg.pipeline_depth = std::stoi(argv[++i]);
        } else if (arg == "--frames" && i + 1 < argc) {
            cfg.num_frames = std::stoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            cfg.seed = std::stoi(argv[++i]);
        } else if (arg == "--verbose") {
            cfg.verbose = true;
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    std::cout << std::string(60, '=') << "\n";
    std::cout << "  自动驾驶流水线模拟器\n";
    std::cout << "  七阶段：传感器 -> 预处理 -> 检测\n";
    std::cout << "         -> 跟踪 -> 预测 -> 规划 -> 控制\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "配置：\n";
    std::cout << "  模式：           " << mode << "\n";
    std::cout << "  流水线深度：     " << cfg.pipeline_depth << "\n";
    std::cout << "  总帧数：         " << cfg.num_frames << "\n";
    std::cout << "  种子：           " << cfg.seed << "\n";

    LatencyStats stats;

    auto wall_start = std::chrono::high_resolution_clock::now();

    if (mode == "sequential") {
        SequentialExecutor executor(cfg, stats);
        executor.run();
    } else {
        PipelinedExecutor executor(cfg, stats);
        executor.run();
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_sec = std::chrono::duration<double>(wall_end - wall_start).count();

    stats.print_summary();

    std::cout << "\n墙上时钟时间：" << std::fixed << std::setprecision(2)
              << wall_sec << " s\n";
    std::cout << "平均吞吐量：" << std::fixed << std::setprecision(1)
              << (cfg.num_frames / wall_sec) << " FPS\n";

    stats.write_json_report("pipeline_telemetry.json",
                            "autonomous_driving_v1", cfg.num_frames);

    return 0;
}
