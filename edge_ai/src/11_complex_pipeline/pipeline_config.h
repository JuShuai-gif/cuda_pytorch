#pragma once

struct PipelineConfig {
    int pipeline_depth = 3;  // 流水线中并发处理的帧数（1 = 顺序执行）
    int num_frames = 100;    // 要模拟的总帧数
    int stats_interval = 20; // 进度打印间隔
    bool verbose = false;    // 每帧是否打印详细信息
    int seed = 42;           // 随机种子，确保可复现
};
