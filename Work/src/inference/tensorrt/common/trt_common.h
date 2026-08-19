// Common utilities for the TensorRT C++ lab.
//
// Self-contained helpers: an ILogger implementation, host/device timers, file
// read/write for the serialized engine, and a tiny JSON report emitter.  The
// measurement policy mirrors the rest of the repo: device time via CUDA events,
// wall time via steady_clock with explicit synchronization.
#pragma once

#include <NvInfer.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#define TRT_CHECK(cond)                                                        \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "TRT error at %s:%d\n", __FILE__, __LINE__);  \
            std::exit(2);                                                      \
        }                                                                      \
    } while (0)

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,        \
                         __LINE__, cudaGetErrorString(_err));                  \
            std::exit(2);                                                      \
        }                                                                      \
    } while (0)

namespace trt_lab {

// cudaMalloc takes void**; a typed pointer-to-pointer does not implicitly
// convert in C++, so wrap it to keep call sites clean.
template <typename T>
inline void cuda_alloc(T** ptr, size_t bytes) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(ptr), bytes));
}

// Minimal ILogger that reports messages at or above a severity threshold.
class Logger : public nvinfer1::ILogger {
public:
    explicit Logger(nvinfer1::ILogger::Severity sev = nvinfer1::ILogger::Severity::kWARNING)
        : m_severity(sev) {}

    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        if (severity <= m_severity) {
            std::fprintf(stderr, "[TRT] %s\n", msg);
        }
    }

private:
    nvinfer1::ILogger::Severity m_severity;
};

struct WallTimer {
    using clock = std::chrono::steady_clock;
    clock::time_point t0;
    void start() { t0 = clock::now(); }
    double ms() const {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    }
};

struct EventTimer {
    cudaEvent_t start_ev, stop_ev;
    EventTimer() {
        CUDA_CHECK(cudaEventCreate(&start_ev));
        CUDA_CHECK(cudaEventCreate(&stop_ev));
    }
    ~EventTimer() {
        cudaEventDestroy(start_ev);
        cudaEventDestroy(stop_ev);
    }
    void start(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(start_ev, s)); }
    void stop(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(stop_ev, s)); }
    double ms() {
        float v = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&v, start_ev, stop_ev));
        return static_cast<double>(v);
    }
};

inline std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    TRT_CHECK(f.good());
    f.seekg(0, std::ios::end);
    size_t size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    f.read(data.data(), size);
    return data;
}

inline void write_file(const std::string& path, const void* data, size_t size) {
    std::ofstream f(path, std::ios::binary);
    TRT_CHECK(f.good());
    f.write(reinterpret_cast<const char*>(data), size);
}

inline double mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

inline double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

struct JsonReport {
    std::string buf;
    bool first = true;
    void begin() { buf = "{"; first = true; }
    void put(const std::string& k, const std::string& v) {
        if (!first) buf += ", ";
        buf += "\"" + k + "\": \"" + v + "\"";
        first = false;
    }
    void put(const std::string& k, double v) {
        if (!first) buf += ", ";
        char tmp[64];
        std::snprintf(tmp, sizeof(tmp), "%.6f", v);
        buf += "\"" + k + "\": " + tmp;
        first = false;
    }
    void put(const std::string& k, long long v) {
        if (!first) buf += ", ";
        char tmp[64];
        std::snprintf(tmp, sizeof(tmp), "%lld", v);
        buf += "\"" + k + "\": " + tmp;
        first = false;
    }
    std::string end() {
        buf += "}\n";
        return buf;
    }
};

}  // namespace trt_lab
