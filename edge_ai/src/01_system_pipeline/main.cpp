#include "pipeline_config.h"
#include "latency_stats.h"
#include "pipeline_executor.h"

#include <cstdio>
#include <cstring>
#include <string>

static void print_usage(const char *prog) {
    std::fprintf(stderr, "用法: %s [选项]\n", prog);
    std::fprintf(stderr, "选项:\n");
    std::fprintf(stderr, "  --mode <sequential|pipelined>  执行模式 (默认: pipelined)\n");
    std::fprintf(stderr, "  --depth <N>                     流水线深度 (默认: 3)\n");
    std::fprintf(stderr, "  --frames <N>                    帧数 (默认: 100)\n");
    std::fprintf(stderr, "  --verbose                       打印每帧详细信息\n");
    std::fprintf(stderr, "  --help                          显示此帮助信息\n");
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
        } else if (arg == "--verbose") {
            cfg.verbose = true;
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    std::fprintf(stderr, "============================================================\n");
    std::fprintf(stderr, "  机器人感知 -> 规划 -> 控制流水线\n");
    std::fprintf(stderr, "============================================================\n");
    std::fprintf(stderr, "配置:\n");
    std::fprintf(stderr, "  模式:               %s\n", mode.c_str());
    std::fprintf(stderr, "  流水线深度:         %d\n", cfg.pipeline_depth);
    std::fprintf(stderr, "  总帧数:             %d\n", cfg.num_frames);

    LatencyStats stats;

    if (mode == "sequential") {
        run_sequential(cfg, stats);
    } else {
        PipelinedExecutor executor(cfg, stats);
        executor.run();
    }

    stats.write_json("pipeline_metrics.json", mode, cfg.pipeline_depth, cfg.num_frames);

    std::fprintf(stderr, "\n输出已写入 pipeline_metrics.json\n");
    return 0;
}
