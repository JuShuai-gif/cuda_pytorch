#include "pipeline_runner.h"
#include "cpu_preprocess.h"
#include "gpu_inference.cuh"
#include "timer.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <vector>

// ============================================================================
// 图像张量尺寸 (通道优先: [C][H][W])
// ============================================================================
static constexpr int SRC_PIXELS = SRC_W * SRC_H * SRC_C; // uint8
static constexpr int DST_PIXELS = DST_W * DST_H * DST_C; // float32
static constexpr int DST_BYTES = DST_PIXELS * sizeof(float);
static constexpr int CONV_OUT_H = DST_H - 2; // 478
static constexpr int CONV_OUT_W = DST_W - 2; // 638
static constexpr int CONV_OUT_PIXELS = CONV_C_OUT * CONV_OUT_H * CONV_OUT_W;
static constexpr int CONV_OUT_BYTES = CONV_OUT_PIXELS * sizeof(float);
static constexpr int POOL_H = CONV_OUT_H / 2; // 239
static constexpr int POOL_W = CONV_OUT_W / 2; // 319
static constexpr int POOL_PIXELS = CONV_C_OUT * POOL_H * POOL_W;
static constexpr int POOL_BYTES = POOL_PIXELS * sizeof(float);
static constexpr int DET_PIXELS = N_DET * POOL_H * POOL_W;
static constexpr int DET_BYTES = DET_PIXELS * sizeof(float);

// ============================================================================
// 辅助函数: 为单帧执行一次完整的流水线处理
// 返回微秒级的分解计时。
// ============================================================================
struct FrameBreakdown {
    int64_t cpu_pre_us = 0;
    int64_t h2d_us = 0;
    int64_t gpu_us = 0;
    int64_t d2h_us = 0;
    int64_t cpu_post_us = 0;
};

static FrameBreakdown run_single_frame(
    int frame_id,
    uint8_t *h_src, float *h_resized, float *h_det_out,
    float *d_resized, float *d_conv_out, float *d_pool_out, float *d_det_out,
    float *d_conv_w, float *d_head_w, float *d_head_b,
    cudaStream_t stream, bool use_async) {
    FrameBreakdown fb;
    CpuTimer cpu_timer;
    GpuTimer gpu_timer;
    int64_t ts;

    // 步骤 1: CPU 预处理 - 生成图像 + 缩放 + 归一化
    cpu_timer.start();
    load_synthetic_image(h_src, SRC_W, SRC_H, SRC_C, frame_id, &ts);
    cpu_resize_normalize(h_src, SRC_W, SRC_H, SRC_C,
                         h_resized, DST_W, DST_H, &ts);
    fb.cpu_pre_us = static_cast<int64_t>(cpu_timer.elapsed_ms() * 1000.0);
    cpu_timer.start();

    // 步骤 2: H2D 传输
    if (use_async) {
        CUDA_CHECK(cudaMemcpyAsync(d_resized, h_resized, DST_BYTES,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        CUDA_CHECK(cudaMemcpy(d_resized, h_resized, DST_BYTES,
                              cudaMemcpyHostToDevice));
    }
    fb.h2d_us = static_cast<int64_t>(cpu_timer.elapsed_ms() * 1000.0);

    // 步骤 3: GPU 流水线 (conv2d -> relu -> maxpool -> 检测头)
    gpu_timer.start(stream);
    gpu_conv2d(d_resized, d_conv_w, d_conv_out,
               DST_H, DST_W, DST_C, CONV_C_OUT, stream);
    gpu_relu(d_conv_out, CONV_OUT_PIXELS, stream);
    gpu_maxpool(d_conv_out, d_pool_out,
                CONV_OUT_H, CONV_OUT_W, CONV_C_OUT, stream);
    gpu_detection_head(d_pool_out, d_head_w, d_head_b, d_det_out,
                       POOL_H, POOL_W, CONV_C_OUT, N_DET, stream);
    if (stream) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    gpu_timer.stop(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream ? stream : 0));
    fb.gpu_us = static_cast<int64_t>(gpu_timer.elapsed_ms() * 1000.0);

    // 步骤 4: D2H 传输
    cpu_timer.start();
    if (use_async) {
        CUDA_CHECK(cudaMemcpyAsync(h_det_out, d_det_out, DET_BYTES,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        CUDA_CHECK(cudaMemcpy(h_det_out, d_det_out, DET_BYTES,
                              cudaMemcpyDeviceToHost));
    }
    fb.d2h_us = static_cast<int64_t>(cpu_timer.elapsed_ms() * 1000.0);

    // 步骤 5: CPU 后处理 - 对解码框进行 NMS
    // 将检测输出解码为框
    cpu_timer.start();
    constexpr int MAX_BOXES = 1024;
    DetectionBox raw_boxes[MAX_BOXES];
    DetectionBox kept_boxes[MAX_BOXES];
    size_t box_count = 0;

    for (int h = 0; h < POOL_H && box_count < MAX_BOXES; ++h) {
        for (int w = 0; w < POOL_W && box_count < MAX_BOXES; ++w) {
            float conf = h_det_out[0 * POOL_H * POOL_W + h * POOL_W + w];
            if (conf > 0.5f && box_count < MAX_BOXES) {
                float cx = h_det_out[1 * POOL_H * POOL_W + h * POOL_W + w];
                float cy = h_det_out[2 * POOL_H * POOL_W + h * POOL_W + w];
                float bw_ = h_det_out[3 * POOL_H * POOL_W + h * POOL_W + w];
                float bh_ = h_det_out[4 * POOL_H * POOL_W + h * POOL_W + w];
                raw_boxes[box_count] = DetectionBox{
                    cx - bw_ * 0.5f, cy - bh_ * 0.5f, bw_, bh_, conf, 0};
                ++box_count;
            }
        }
    }

    cpu_nms(raw_boxes, box_count, 0.5f, kept_boxes, MAX_BOXES, &ts);
    fb.cpu_post_us = static_cast<int64_t>(cpu_timer.elapsed_ms() * 1000.0);

    return fb;
}

// ============================================================================
// 方案 1: 原始串行
// ============================================================================
PipelineTiming run_naive() {
    PipelineTiming result;
    std::cout << "\n--- 方案 1: 原始串行 ---\n";

    // 分配主机内存
    uint8_t *h_src = new uint8_t[SRC_PIXELS];
    float *h_resized = new float[DST_PIXELS];
    float *h_det_out = new float[DET_PIXELS];

    // 分配设备内存
    float *d_resized, *d_conv_out, *d_pool_out, *d_det_out;
    CUDA_CHECK(cudaMalloc(&d_resized, DST_BYTES));
    CUDA_CHECK(cudaMalloc(&d_conv_out, CONV_OUT_BYTES));
    CUDA_CHECK(cudaMalloc(&d_pool_out, POOL_BYTES));
    CUDA_CHECK(cudaMalloc(&d_det_out, DET_BYTES));

    float *d_conv_w = gpu_alloc_init_conv_weights(DST_C, CONV_C_OUT);
    float *d_head_w, *d_head_b;
    gpu_alloc_init_head_weights(&d_head_w, &d_head_b, CONV_C_OUT, N_DET);

    double total_cpu_pre = 0, total_h2d = 0, total_gpu = 0, total_d2h = 0, total_cpu_post = 0;
    auto wall_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_FRAMES; ++i) {
        auto fb = run_single_frame(i, h_src, h_resized, h_det_out,
                                   d_resized, d_conv_out, d_pool_out, d_det_out,
                                   d_conv_w, d_head_w, d_head_b,
                                   0, false);
        total_cpu_pre += fb.cpu_pre_us;
        total_h2d += fb.h2d_us;
        total_gpu += fb.gpu_us;
        total_d2h += fb.d2h_us;
        total_cpu_post += fb.cpu_post_us;
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    result.total_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
    double n = static_cast<double>(NUM_FRAMES);
    result.cpu_preprocess_ms = total_cpu_pre / (n * 1000.0);
    result.transfer_to_device_ms = total_h2d / (n * 1000.0);
    result.gpu_compute_ms = total_gpu / (n * 1000.0);
    result.transfer_to_host_ms = total_d2h / (n * 1000.0);
    result.cpu_postprocess_ms = total_cpu_post / (n * 1000.0);
    result.throughput_fps = NUM_FRAMES / (result.total_ms / 1000.0);

    std::cout << "  总耗时: " << std::fixed << std::setprecision(1) << result.total_ms << " ms\n";
    std::cout << "  吞吐量: " << result.throughput_fps << " fps\n";

    delete[] h_src;
    delete[] h_resized;
    delete[] h_det_out;
    CUDA_CHECK(cudaFree(d_resized));
    CUDA_CHECK(cudaFree(d_conv_out));
    CUDA_CHECK(cudaFree(d_pool_out));
    CUDA_CHECK(cudaFree(d_det_out));
    CUDA_CHECK(cudaFree(d_conv_w));
    CUDA_CHECK(cudaFree(d_head_w));
    CUDA_CHECK(cudaFree(d_head_b));

    return result;
}

// ============================================================================
// 方案 2: 流重叠 (3 条 CUDA 流)
// ============================================================================
PipelineTiming run_stream_overlapped() {
    PipelineTiming result;
    std::cout << "\n--- 方案 2: 流重叠 ---\n";

    // 每条流的资源
    uint8_t *h_src[NUM_STREAMS];
    float *h_resized[NUM_STREAMS];
    float *h_det_out[NUM_STREAMS];
    float *d_resized[NUM_STREAMS], *d_conv_out[NUM_STREAMS];
    float *d_pool_out[NUM_STREAMS], *d_det_out[NUM_STREAMS];
    cudaStream_t streams[NUM_STREAMS];

    for (int s = 0; s < NUM_STREAMS; ++s) {
        h_src[s] = new uint8_t[SRC_PIXELS];
        h_resized[s] = new float[DST_PIXELS];
        h_det_out[s] = new float[DET_PIXELS];
        CUDA_CHECK(cudaMalloc(&d_resized[s], DST_BYTES));
        CUDA_CHECK(cudaMalloc(&d_conv_out[s], CONV_OUT_BYTES));
        CUDA_CHECK(cudaMalloc(&d_pool_out[s], POOL_BYTES));
        CUDA_CHECK(cudaMalloc(&d_det_out[s], DET_BYTES));
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
    }

    // 跨流共享的权重 (只读)
    float *d_conv_w = gpu_alloc_init_conv_weights(DST_C, CONV_C_OUT);
    float *d_head_w, *d_head_b;
    gpu_alloc_init_head_weights(&d_head_w, &d_head_b, CONV_C_OUT, N_DET);

    double total_cpu_pre = 0, total_h2d = 0, total_gpu = 0, total_d2h = 0, total_cpu_post = 0;
    auto wall_start = std::chrono::high_resolution_clock::now();

    // 按 NUM_STREAMS 大小分批处理帧
    for (int base = 0; base < NUM_FRAMES; base += NUM_STREAMS) {
        int batch_size = std::min(NUM_STREAMS, NUM_FRAMES - base);

        // 为此批次中的每条流发出异步操作
        for (int s = 0; s < batch_size; ++s) {
            int frame_id = base + s;
            cudaStream_t st = streams[s];

            // CPU 预处理
            CpuTimer cpu_t;
            cpu_t.start();
            int64_t ts;
            load_synthetic_image(h_src[s], SRC_W, SRC_H, SRC_C, frame_id, &ts);
            cpu_resize_normalize(h_src[s], SRC_W, SRC_H, SRC_C,
                                 h_resized[s], DST_W, DST_H, &ts);
            total_cpu_pre += cpu_t.elapsed_ms() * 1000.0;

            // 异步 H2D
            GpuTimer h2d_t;
            h2d_t.start(st);
            CUDA_CHECK(cudaMemcpyAsync(d_resized[s], h_resized[s], DST_BYTES,
                                       cudaMemcpyHostToDevice, st));
            h2d_t.stop(st);

            // 异步 GPU 流水线
            gpu_conv2d(d_resized[s], d_conv_w, d_conv_out[s],
                       DST_H, DST_W, DST_C, CONV_C_OUT, st);
            gpu_relu(d_conv_out[s], CONV_OUT_PIXELS, st);
            gpu_maxpool(d_conv_out[s], d_pool_out[s],
                        CONV_OUT_H, CONV_OUT_W, CONV_C_OUT, st);
            gpu_detection_head(d_pool_out[s], d_head_w, d_head_b, d_det_out[s],
                               POOL_H, POOL_W, CONV_C_OUT, N_DET, st);

            // 异步 D2H
            CUDA_CHECK(cudaMemcpyAsync(h_det_out[s], d_det_out[s], DET_BYTES,
                                       cudaMemcpyDeviceToHost, st));
        }

        // 同步并后处理
        for (int s = 0; s < batch_size; ++s) {
            cudaStream_t st = streams[s];
            CUDA_CHECK(cudaStreamSynchronize(st));

            CpuTimer cpu_t;
            cpu_t.start();
            constexpr int MAX_BOXES = 1024;
            DetectionBox raw_boxes[MAX_BOXES];
            DetectionBox kept_boxes[MAX_BOXES];
            size_t box_count = 0;
            int64_t ts;
            for (int h = 0; h < POOL_H && box_count < MAX_BOXES; ++h) {
                for (int w = 0; w < POOL_W && box_count < MAX_BOXES; ++w) {
                    float conf = h_det_out[s][0 * POOL_H * POOL_W + h * POOL_W + w];
                    if (conf > 0.5f && box_count < MAX_BOXES) {
                        float cx = h_det_out[s][1 * POOL_H * POOL_W + h * POOL_W + w];
                        float cy = h_det_out[s][2 * POOL_H * POOL_W + h * POOL_W + w];
                        float bw_ = h_det_out[s][3 * POOL_H * POOL_W + h * POOL_W + w];
                        float bh_ = h_det_out[s][4 * POOL_H * POOL_W + h * POOL_W + w];
                        raw_boxes[box_count++] = DetectionBox{
                            cx - bw_ * 0.5f, cy - bh_ * 0.5f, bw_, bh_, conf, 0};
                    }
                }
            }
            cpu_nms(raw_boxes, box_count, 0.5f, kept_boxes, MAX_BOXES, &ts);
            total_cpu_post += cpu_t.elapsed_ms() * 1000.0;
        }
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    result.total_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
    double n = static_cast<double>(NUM_FRAMES);
    result.cpu_preprocess_ms = total_cpu_pre / (n * 1000.0);
    result.transfer_to_device_ms = total_h2d / (n * 1000.0);
    result.gpu_compute_ms = total_gpu / (n * 1000.0);
    result.transfer_to_host_ms = total_d2h / (n * 1000.0);
    result.cpu_postprocess_ms = total_cpu_post / (n * 1000.0);
    result.throughput_fps = NUM_FRAMES / (result.total_ms / 1000.0);

    std::cout << "  总耗时: " << std::fixed << std::setprecision(1) << result.total_ms << " ms\n";
    std::cout << "  吞吐量: " << result.throughput_fps << " fps\n";

    // 清理
    for (int s = 0; s < NUM_STREAMS; ++s) {
        delete[] h_src[s];
        delete[] h_resized[s];
        delete[] h_det_out[s];
        CUDA_CHECK(cudaFree(d_resized[s]));
        CUDA_CHECK(cudaFree(d_conv_out[s]));
        CUDA_CHECK(cudaFree(d_pool_out[s]));
        CUDA_CHECK(cudaFree(d_det_out[s]));
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
    }
    CUDA_CHECK(cudaFree(d_conv_w));
    CUDA_CHECK(cudaFree(d_head_w));
    CUDA_CHECK(cudaFree(d_head_b));

    return result;
}

// ============================================================================
// 方案 3: 锁页/映射内存 (零拷贝)
// ============================================================================
PipelineTiming run_pinned_memory() {
    PipelineTiming result;
    std::cout << "\n--- 方案 3: 锁页映射内存 (零拷贝) ---\n";

    // 分配可供 GPU 访问的映射主机内存
    uint8_t *h_src = nullptr;
    float *h_resized = nullptr;
    float *h_det_out = nullptr;
    // 使用 cudaHostAllocMapped 进行零拷贝访问
    CUDA_CHECK(cudaHostAlloc(&h_src, SRC_PIXELS * sizeof(uint8_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&h_resized, DST_BYTES, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&h_det_out, DET_BYTES, cudaHostAllocMapped));

    // 获取映射内存的设备指针
    float *d_resized_mapped = nullptr;
    float *d_det_out_mapped = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_resized_mapped, h_resized, 0));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_det_out_mapped, h_det_out, 0));

    // 中间张量仍需要设备内存
    float *d_conv_out, *d_pool_out;
    CUDA_CHECK(cudaMalloc(&d_conv_out, CONV_OUT_BYTES));
    CUDA_CHECK(cudaMalloc(&d_pool_out, POOL_BYTES));

    float *d_conv_w = gpu_alloc_init_conv_weights(DST_C, CONV_C_OUT);
    float *d_head_w, *d_head_b;
    gpu_alloc_init_head_weights(&d_head_w, &d_head_b, CONV_C_OUT, N_DET);

    double total_cpu_pre = 0, total_h2d = 0, total_gpu = 0, total_d2h = 0, total_cpu_post = 0;
    auto wall_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_FRAMES; ++i) {
        CpuTimer cpu_t;
        int64_t ts;

        // CPU 预处理写入映射内存 (GPU 可直接看到)
        cpu_t.start();
        load_synthetic_image(h_src, SRC_W, SRC_H, SRC_C, i, &ts);
        cpu_resize_normalize(h_src, SRC_W, SRC_H, SRC_C,
                             h_resized, DST_W, DST_H, &ts);
        total_cpu_pre += cpu_t.elapsed_ms() * 1000.0;

        // 无需显式 H2D——GPU 直接访问映射内存
        // 但为了正确性仍需同步点
        CUDA_CHECK(cudaDeviceSynchronize());

        // GPU 流水线 (从映射输入读取，写入设备中间张量)
        GpuTimer gpu_t;
        gpu_t.start();
        gpu_conv2d(d_resized_mapped, d_conv_w, d_conv_out,
                   DST_H, DST_W, DST_C, CONV_C_OUT);
        gpu_relu(d_conv_out, CONV_OUT_PIXELS);
        gpu_maxpool(d_conv_out, d_pool_out,
                    CONV_OUT_H, CONV_OUT_W, CONV_C_OUT);
        // 将检测输出写入映射内存 (无需显式 D2H)
        gpu_detection_head(d_pool_out, d_head_w, d_head_b, d_det_out_mapped,
                           POOL_H, POOL_W, CONV_C_OUT, N_DET);
        CUDA_CHECK(cudaDeviceSynchronize());
        gpu_t.stop();
        total_gpu += gpu_t.elapsed_ms() * 1000.0;

        // CPU 后处理 (结果已在映射内存中)
        cpu_t.start();
        constexpr int MAX_BOXES = 1024;
        DetectionBox raw_boxes[MAX_BOXES];
        DetectionBox kept_boxes[MAX_BOXES];
        size_t box_count = 0;
        for (int h = 0; h < POOL_H && box_count < MAX_BOXES; ++h) {
            for (int w = 0; w < POOL_W && box_count < MAX_BOXES; ++w) {
                float conf = h_det_out[0 * POOL_H * POOL_W + h * POOL_W + w];
                if (conf > 0.5f && box_count < MAX_BOXES) {
                    float cx = h_det_out[1 * POOL_H * POOL_W + h * POOL_W + w];
                    float cy = h_det_out[2 * POOL_H * POOL_W + h * POOL_W + w];
                    float bw_ = h_det_out[3 * POOL_H * POOL_W + h * POOL_W + w];
                    float bh_ = h_det_out[4 * POOL_H * POOL_W + h * POOL_W + w];
                    raw_boxes[box_count++] = DetectionBox{
                        cx - bw_ * 0.5f, cy - bh_ * 0.5f, bw_, bh_, conf, 0};
                }
            }
        }
        cpu_nms(raw_boxes, box_count, 0.5f, kept_boxes, MAX_BOXES, &ts);
        total_cpu_post += cpu_t.elapsed_ms() * 1000.0;
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    result.total_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
    double n = static_cast<double>(NUM_FRAMES);
    result.cpu_preprocess_ms = total_cpu_pre / (n * 1000.0);
    result.transfer_to_device_ms = total_h2d / std::max(1.0, n);
    result.gpu_compute_ms = total_gpu / (n * 1000.0);
    result.transfer_to_host_ms = total_d2h / std::max(1.0, n);
    result.cpu_postprocess_ms = total_cpu_post / (n * 1000.0);
    result.throughput_fps = NUM_FRAMES / (result.total_ms / 1000.0);

    std::cout << "  总耗时: " << std::fixed << std::setprecision(1) << result.total_ms << " ms\n";
    std::cout << "  吞吐量: " << result.throughput_fps << " fps\n";

    CUDA_CHECK(cudaFreeHost(h_src));
    CUDA_CHECK(cudaFreeHost(h_resized));
    CUDA_CHECK(cudaFreeHost(h_det_out));
    CUDA_CHECK(cudaFree(d_conv_out));
    CUDA_CHECK(cudaFree(d_pool_out));
    CUDA_CHECK(cudaFree(d_conv_w));
    CUDA_CHECK(cudaFree(d_head_w));
    CUDA_CHECK(cudaFree(d_head_b));

    return result;
}

// ============================================================================
// 写入指标 JSON
// ============================================================================
static void write_approach_json(std::ofstream &of, const std::string &name,
                                const PipelineTiming &t) {
    of << "    \"" << name << "\": {\n";
    of << "      \"total_ms\": " << std::fixed << std::setprecision(2) << t.total_ms << ",\n";
    of << "      \"throughput_fps\": " << std::fixed << std::setprecision(2) << t.throughput_fps << ",\n";
    of << "      \"transfer_ms\": " << std::fixed << std::setprecision(3)
       << (t.transfer_to_device_ms + t.transfer_to_host_ms) << ",\n";
    of << "      \"compute_ms\": " << std::fixed << std::setprecision(3)
       << (t.cpu_preprocess_ms + t.gpu_compute_ms + t.cpu_postprocess_ms) << ",\n";
    of << "      \"breakdown\": {\n";
    of << "        \"cpu_preprocess_ms\": " << std::fixed << std::setprecision(3) << t.cpu_preprocess_ms << ",\n";
    of << "        \"h2d_transfer_ms\": " << std::fixed << std::setprecision(3) << t.transfer_to_device_ms << ",\n";
    of << "        \"gpu_compute_ms\": " << std::fixed << std::setprecision(3) << t.gpu_compute_ms << ",\n";
    of << "        \"d2h_transfer_ms\": " << std::fixed << std::setprecision(3) << t.transfer_to_host_ms << ",\n";
    of << "        \"cpu_postprocess_ms\": " << std::fixed << std::setprecision(3) << t.cpu_postprocess_ms << "\n";
    of << "      }\n";
    of << "    }";
}

void write_metrics_json(const std::string &path,
                        const PipelineTiming &naive,
                        const PipelineTiming &overlapped,
                        const PipelineTiming &pinned) {
    std::ofstream of(path);
    of << "{\n";
    of << "  \"num_frames\": " << NUM_FRAMES << ",\n";
    of << "  \"image_size\": \"1920x1080x3\",\n";
    of << "  \"resize_target\": \"640x480x3\",\n";
    of << "  \"gpu_conv_filters\": " << CONV_C_OUT << ",\n";
    of << "  \"approaches\": {\n";
    write_approach_json(of, "naive", naive);
    of << ",\n";
    write_approach_json(of, "stream_overlap", overlapped);
    of << ",\n";
    write_approach_json(of, "pinned_memory", pinned);
    of << "\n  }\n";
    of << "}\n";
    of.close();
    std::cout << "\n指标已写入 " << path << "\n";
}
