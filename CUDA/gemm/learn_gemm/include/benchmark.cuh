#include <cuda_runtime.h>
#include <torch/torch.h>
#include <torch/cuda.h>

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace bench {

// =========================
// Error handling utilities
// =========================

inline void CheckCuda(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        char msg[1024];
        std::snprintf(
            msg,
            sizeof(msg),
            "CUDA error at %s:%d, expression '%s' failed with error: %s",
            file,
            line,
            expr,
            cudaGetErrorString(err)
        );
        throw std::runtime_error(msg);
    }
}

#define CHECK_CUDA(expr) ::bench::CheckCuda((expr), #expr, __FILE__, __LINE__)

inline void CheckTorchCudaAvailable() {
    if (!torch::cuda::is_available()) {
        throw std::runtime_error("Torch CUDA is not available.");
    }
}

// =========================
// Benchmark config/result
// =========================

struct BenchmarkConfig {
    int warmup = 10;
    int repeats = 100;
    bool verbose = true;
    bool synchronize_before_start = true;
};

struct BenchmarkResult {
    std::string name;
    int m = 0;
    int k = 0;
    int n = 0;

    int warmup = 0;
    int repeats = 0;

    double average_time_ms = 0.0;
    double total_time_ms = 0.0;
    double tflops = 0.0;
    std::optional<double> utilization_pct;
};

// GEMM FLOPs = 2 * M * K * N
inline long long ComputeGemmFlops(int m, int k, int n) {
    return 2LL * m * k * n;
}

// 这里你可以接入自己的硬件峰值查询逻辑
inline double GetTheoreticalPeakTflopsFp32() {
    // TODO: replace with your device-specific query logic
    return 0.0;
}

// =========================
// CUDA Event RAII wrapper
// =========================

class CudaEvent {
public:
    CudaEvent() {
        CHECK_CUDA(cudaEventCreate(&event_));
    }

    ~CudaEvent() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
        other.event_ = nullptr;
    }

    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            if (event_ != nullptr) {
                cudaEventDestroy(event_);
            }
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }

    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

// =========================
// Timer backend interface
// =========================

class ITimer {
public:
    virtual ~ITimer() = default;
    virtual void RecordStart(cudaStream_t stream = nullptr) = 0;
    virtual void RecordStop(cudaStream_t stream = nullptr) = 0;
    virtual void SynchronizeStop() = 0;
    virtual float ElapsedMilliseconds() const = 0;
};

class CudaEventTimer final : public ITimer {
public:
    void RecordStart(cudaStream_t stream = nullptr) override {
        CHECK_CUDA(cudaEventRecord(start_.get(), stream));
    }

    void RecordStop(cudaStream_t stream = nullptr) override {
        CHECK_CUDA(cudaEventRecord(stop_.get(), stream));
    }

    void SynchronizeStop() override {
        CHECK_CUDA(cudaEventSynchronize(stop_.get()));
    }

    float ElapsedMilliseconds() const override {
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start_.get(), stop_.get()));
        return ms;
    }

private:
    CudaEvent start_;
    CudaEvent stop_;
};

// =========================
// Sync helpers
// =========================

inline void SynchronizeDevice() {
    CHECK_CUDA(cudaDeviceSynchronize());
}

inline cudaStream_t GetCurrentTorchCudaStream() {
    CheckTorchCudaAvailable();
    return at::cuda::getDefaultCUDAStream().stream();
}

// 如果你的 LibTorch op 明确跑在 current stream，也可以切到 current stream：
// return at::cuda::getDefaultCUDAStream().stream();
// 某些项目会更偏好 getDefaultCUDAStream，便于 benchmark 稳定复现。
// 如果你在项目里显式管理 current stream，可改成 current CUDA stream 接口。

// =========================
// Core benchmark function
// =========================

BenchmarkResult BenchmarkGemm(
    const std::string& name,
    const std::function<void()>& kernel_func,
    int m,
    int k,
    int n,
    const BenchmarkConfig& config,
    cudaStream_t stream = nullptr,
    ITimer* timer = nullptr)
{
    if (kernel_func == nullptr) {
        throw std::invalid_argument("kernel_func must not be empty.");
    }
    if (m <= 0 || k <= 0 || n <= 0) {
        throw std::invalid_argument("m, k, n must be positive.");
    }
    if (config.warmup < 0 || config.repeats <= 0) {
        throw std::invalid_argument("warmup must be >= 0 and repeats must be > 0.");
    }

    CudaEventTimer default_timer;
    ITimer* effective_timer = (timer != nullptr) ? timer : &default_timer;

    BenchmarkResult result;
    result.name = name;
    result.m = m;
    result.k = k;
    result.n = n;
    result.warmup = config.warmup;
    result.repeats = config.repeats;

    if (config.verbose) {
        std::printf("[%-16s] Warmup: %d, Repeats: %d, Shape: (%d, %d, %d)\n",
                    name.c_str(), config.warmup, config.repeats, m, k, n);
    }

    // Warmup
    for (int i = 0; i < config.warmup; ++i) {
        kernel_func();
    }

    SynchronizeDevice();

    if (config.synchronize_before_start) {
        SynchronizeDevice();
    }

    // Timed section
    effective_timer->RecordStart(stream);

    for (int i = 0; i < config.repeats; ++i) {
        kernel_func();
    }

    effective_timer->RecordStop(stream);
    effective_timer->SynchronizeStop();

    const double total_time_ms = static_cast<double>(effective_timer->ElapsedMilliseconds());
    const double avg_time_ms = total_time_ms / static_cast<double>(config.repeats);

    result.total_time_ms = total_time_ms;
    result.average_time_ms = avg_time_ms;

    const long long total_flops_per_iter = ComputeGemmFlops(m, k, n);
    const double total_time_sec = total_time_ms / 1000.0;
    result.tflops =
        (static_cast<double>(config.repeats) * static_cast<double>(total_flops_per_iter)) /
        total_time_sec / 1.0e12;

    const double peak_tflops = GetTheoreticalPeakTflopsFp32();
    if (peak_tflops > 0.0) {
        result.utilization_pct = (result.tflops / peak_tflops) * 100.0;
    }

    if (config.verbose) {
        std::printf("  Average time : %.4f ms\n", result.average_time_ms);
        std::printf("  Total time   : %.4f ms\n", result.total_time_ms);
        std::printf("  Throughput   : %.2f TFLOPS\n", result.tflops);
        if (result.utilization_pct.has_value()) {
            std::printf("  Utilization  : %.1f%% of theoretical peak\n", *result.utilization_pct);
        }
    }

    return result;
}

}  // namespace bench