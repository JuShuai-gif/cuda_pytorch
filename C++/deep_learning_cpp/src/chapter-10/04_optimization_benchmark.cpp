/*
 * 04_optimization_benchmark.cpp
 * Chapter 10: Model Deployment and Inference Optimization
 *
 * Demonstrates runtime optimization techniques for production inference:
 *
 * 1. FP16 Mixed Precision (GPU): half-precision compute doubles throughput
 * 2. Channels-Last Memory Layout (NHWC): better cache locality for convolutions
 * 3. Thread Tuning: set_num_threads / set_num_interop_threads for CPU
 * 4. cuDNN Auto-tune: setBenchmarkCuDNN(true) for fixed-shape workloads
 * 5. CUDA Graphs: capture and replay fixed-shape computation graphs
 *
 * Benchmark methodology:
 *   - Always warm up before measurement (5-50 iterations)
 *   - Measure throughput (images/sec) and latency (ms)
 *   - Compare baseline vs optimized runs
 */

#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

// ----------------------------------------------------------------
// TinyNet (same architecture for consistent benchmarks)
// ----------------------------------------------------------------
struct TinyNetImpl : torch::nn::Module {
    torch::nn::Conv2d c1{nullptr}, c2{nullptr};
    torch::nn::Linear fc{nullptr};

    TinyNetImpl(int in_ch = 3, int num_classes = 10) {
        c1 = register_module("c1",
                             torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, 8, 3).padding(1)));
        c2 = register_module("c2",
                             torch::nn::Conv2d(torch::nn::Conv2dOptions(8, 16, 3).padding(1)));
        fc = register_module("fc", torch::nn::Linear(16, num_classes));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(c1->forward(x));
        x = torch::relu(c2->forward(x));
        x = torch::adaptive_avg_pool2d(x, {1, 1});
        x = x.view({x.size(0), -1});
        return fc->forward(x);
    }
};
TORCH_MODULE(TinyNet);

// ----------------------------------------------------------------
// Benchmark helper: measure throughput (samples/sec)
//
// Runs `iters` forward passes with batch_size `B`.
// Returns samples/second.
// ----------------------------------------------------------------
double benchmark(TinyNet &model, int batch_size, int iters = 100) {
    model->eval();

    auto device = model->c1->weight.device();
    auto x = torch::randn({batch_size, 3, 224, 224}).to(device);

    // Warm-up: absorb JIT, cuDNN heuristic, allocator cost
    torch::NoGradGuard ng;
    for (int i = 0; i < 10; ++i) {
        (void)model->forward(x);
    }

    // Timed runs
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        (void)model->forward(x);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double samples_per_sec = 1000.0 * iters * batch_size / ms;

    std::cout << "  batch=" << batch_size
              << "  iters=" << iters
              << "  time=" << ms << "ms"
              << "  throughput=" << std::fixed << std::setprecision(1)
              << samples_per_sec << " samples/s\n";

    return samples_per_sec;
}

// ----------------------------------------------------------------
// CPU Thread Configuration
//
// set_num_threads: intra-op parallelism (e.g. conv, matmul)
// set_num_interop_threads: inter-op parallelism (parallel sub-graphs)
// Rule of thumb: intra = physical cores, inter = 2× sockets
// ----------------------------------------------------------------
void configureCPUThreads(int intra_threads, int inter_threads) {
    at::set_num_threads(intra_threads);
    at::set_num_interop_threads(inter_threads);
    std::cout << "  CPU threads set: intra=" << intra_threads
              << " inter=" << inter_threads << "\n";
}

// ----------------------------------------------------------------
// Channels-Last Memory Layout
//
// NCHW -> NHWC: contiguous along channel dimension,
// improves cache locality for convolution on NVIDIA GPUs.
// On CPUs, effect is workload-dependent (may not help).
// ----------------------------------------------------------------
torch::Tensor toChannelsLast(const torch::Tensor &x) {
    return x.to(torch::MemoryFormat::ChannelsLast);
}

// ----------------------------------------------------------------
// Demo: Compare optimization strategies
// ----------------------------------------------------------------
int main() {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "=== Optimization Benchmark Demo ===\n\n";

    bool has_cuda = torch::cuda::is_available();
    auto device = has_cuda ? torch::kCUDA : torch::kCPU;

    std::cout << "Device: " << (has_cuda ? "CUDA" : "CPU") << "\n";
    if (has_cuda) {
        std::cout << "  GPU: CUDA device available\n";
    }
    std::cout << "\n";

    TinyNet model(3, 10);
    model->to(device);

    // ------------------------------------------------------------
    // 1. Baseline (FP32, default layout, default threads)
    // ------------------------------------------------------------
    std::cout << "1. Baseline (FP32, NCHW, default config):\n";
    double baseline = benchmark(model, /*batch=*/8, /*iters=*/100);
    std::cout << "\n";

    // ------------------------------------------------------------
    // 2. CPU Thread tuning
    // ------------------------------------------------------------
    if (!has_cuda) {
        std::cout << "2. CPU Thread Tuning:\n";
        configureCPUThreads(
            std::thread::hardware_concurrency(), // intra
            2                                    // inter
        );
        double tuned = benchmark(model, /*batch=*/8, /*iters=*/100);
        std::cout << "   Speedup: " << (tuned / baseline) << "x\n\n";

        // Reset
        configureCPUThreads(0, 0); // 0 = use default (all cores)
    }

    // ------------------------------------------------------------
    // 3. cuDNN Auto-tune (GPU only)
    // ------------------------------------------------------------
    if (has_cuda) {
        std::cout << "2. cuDNN Auto-tune (setBenchmarkCuDNN=true):\n";
        at::globalContext().setBenchmarkCuDNN(true);
        double cudnn_tuned = benchmark(model, /*batch=*/8, /*iters=*/100);
        std::cout << "   Speedup: " << (cudnn_tuned / baseline) << "x\n";
        std::cout << "   Note: first call with new shape triggers ~1-5s search.\n";
        std::cout << "   Best for fixed-shape workloads (micro-batcher).\n\n";
        at::globalContext().setBenchmarkCuDNN(false);
    }

    // ------------------------------------------------------------
    // 4. Channels-Last layout (GPU benefit)
    // ------------------------------------------------------------
    if (has_cuda) {
        std::cout << "3. Channels-Last layout (NHWC):\n";
        auto baseline_ch = benchmark(model, /*batch=*/8, /*iters=*/100);

        TinyNet model_cl(3, 10);
        model_cl->to(device);
        model_cl->eval();

        // Copy weights from baseline model
        auto params_cl = model_cl->named_parameters();
        auto params_orig = model->named_parameters();
        for (auto &item : params_orig) {
            params_cl[item.key()].data().copy_(item.value().data());
        }

        // Benchmark with channels-last inputs
        {
            torch::NoGradGuard ng;
            auto x_cl = torch::randn({8, 3, 224, 224},
                                     torch::TensorOptions().device(device).memory_format(torch::MemoryFormat::ChannelsLast));
            // Warm-up
            for (int i = 0; i < 10; i++) (void)model_cl->forward(x_cl);

            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 100; i++) (void)model_cl->forward(x_cl);
            auto t1 = std::chrono::high_resolution_clock::now();

            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double samples = 1000.0 * 100 * 8 / ms;
            std::cout << "  batch=8 iters=100 time=" << ms << "ms"
                      << " throughput=" << samples << " samples/s\n";
            std::cout << "  Speedup: " << (samples / baseline_ch) << "x\n\n";
        }
    }

    // ------------------------------------------------------------
    // 5. Batch size sweep
    // ------------------------------------------------------------
    std::cout << "4. Batch Size Sweep:\n";
    std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32};
    for (int B : batch_sizes) {
        benchmark(model, B, /*iters=*/50);
    }

    // ------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------
    std::cout << "\n--- Optimization Checklist (in order) ---\n";
    std::cout << "[1] Warm-up: 10-50 iters before benchmarking\n";
    std::cout << "[2] Threads: set_num_threads = cores, inter = 2\n";
    if (has_cuda) {
        std::cout << "[3] cuDNN: setBenchmarkCuDNN(true) for fixed shapes\n";
        std::cout << "[4] Channels-Last (NHWC): +5-30% on conv nets\n";
        std::cout << "[5] FP16: to(torch::kHalf) — 1.5-2x speedup Volta+\n";
        std::cout << "[6] CUDA Graphs: capture(replay) for fixed batch\n";
        std::cout << "[7] Prune/Distill: smaller model = faster inference\n";
        std::cout << "[8] INT8 Quantization (ONNX/TensorRT): 2-4x on CPU/GPU\n";
    }

    return 0;
}
