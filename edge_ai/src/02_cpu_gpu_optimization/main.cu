#include "pipeline_runner.h"
#include "gpu_inference.cuh"

#include <iostream>
#include <iomanip>

int main() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "未找到支持 CUDA 的设备。\n";
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "============================================\n";
    std::cout << "  CPU/GPU 混合机器人感知流水线\n";
    std::cout << "============================================\n";
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "计算能力: " << prop.major << "." << prop.minor << "\n";
    std::cout << "全局内存: " << prop.totalGlobalMem / (1024.0 * 1024.0) << " MB\n";
    std::cout << "支持并发拷贝+执行: "
              << (prop.asyncEngineCount > 0 ? "是" : "否") << "\n";
    std::cout << "帧数: " << NUM_FRAMES << "\n";
    std::cout << "流水线: 1920x1080x3(uint8) -> CPU 缩放 640x480x3(float32)\n";
    std::cout << "        -> GPU Conv2D(16x3x3) -> ReLU -> MaxPool(2x2)\n";
    std::cout << "        -> 检测头 -> CPU NMS\n\n";

    PipelineTiming naive = run_naive();
    PipelineTiming overlapped = run_stream_overlapped();
    PipelineTiming pinned = run_pinned_memory();

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "  性能对比\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << std::left
              << std::setw(22) << "方案"
              << std::setw(14) << "总耗时(ms)"
              << std::setw(14) << "FPS"
              << std::setw(14) << "计算(ms)"
              << std::setw(14) << "传输(ms)\n";
    std::cout << std::string(78, '-') << "\n";

    auto print_row = [](const std::string& name, const PipelineTiming& t) {
        std::cout << std::left
                  << std::setw(22) << name
                  << std::setw(14) << std::fixed << std::setprecision(1) << t.total_ms
                  << std::setw(14) << std::fixed << std::setprecision(2) << t.throughput_fps
                  << std::setw(14) << std::fixed << std::setprecision(2)
                  << (t.cpu_preprocess_ms + t.gpu_compute_ms + t.cpu_postprocess_ms)
                  << std::setw(14) << std::fixed << std::setprecision(2)
                  << (t.transfer_to_device_ms + t.transfer_to_host_ms)
                  << "\n";
    };

    print_row("原始串行", naive);
    print_row("流重叠", overlapped);
    print_row("锁页映射 (零拷贝)", pinned);

    std::cout << "\n相对原始串行的加速比:\n";
    std::cout << "  流重叠: "
              << std::fixed << std::setprecision(2)
              << (naive.total_ms / overlapped.total_ms) << "x\n";
    std::cout << "  锁页映射: "
              << std::fixed << std::setprecision(2)
              << (naive.total_ms / pinned.total_ms) << "x\n";

    write_metrics_json("gpu_pipeline_metrics.json", naive, overlapped, pinned);

    std::cout << "\n完成。\n";
    return 0;
}
