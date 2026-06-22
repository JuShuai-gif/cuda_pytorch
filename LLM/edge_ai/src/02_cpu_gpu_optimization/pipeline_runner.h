#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>

// ============================================================================
// 流水线配置
// ============================================================================
constexpr int NUM_FRAMES = 100;
constexpr int SRC_W = 1920;
constexpr int SRC_H = 1080;
constexpr int SRC_C = 3;
constexpr int DST_W = 640;
constexpr int DST_H = 480;
constexpr int DST_C = 3;
constexpr int CONV_C_OUT = 16;
constexpr int CONV_KERNEL = 3;
constexpr int POOL_STRIDE = 2;
constexpr int N_DET = 5;
constexpr int NUM_STREAMS = 3;

// ============================================================================
// 单种方案的计时分解
// ============================================================================
struct PipelineTiming {
    double total_ms = 0.0;
    double cpu_preprocess_ms = 0.0;
    double transfer_to_device_ms = 0.0;
    double gpu_compute_ms = 0.0;
    double transfer_to_host_ms = 0.0;
    double cpu_postprocess_ms = 0.0;
    double throughput_fps = 0.0;
};

// ============================================================================
// 方案 1: 原始串行 (默认流，阻塞式拷贝)
// ============================================================================
PipelineTiming run_naive();

// ============================================================================
// 方案 2: 流重叠 (3 条 CUDA 流，异步重叠)
// ============================================================================
PipelineTiming run_stream_overlapped();

// ============================================================================
// 方案 3: 锁页/映射内存 (零拷贝访问)
// ============================================================================
PipelineTiming run_pinned_memory();

// ============================================================================
// 将结果写入 gpu_pipeline_metrics.json
// ============================================================================
void write_metrics_json(const std::string &path,
                        const PipelineTiming &naive,
                        const PipelineTiming &overlapped,
                        const PipelineTiming &pinned);
