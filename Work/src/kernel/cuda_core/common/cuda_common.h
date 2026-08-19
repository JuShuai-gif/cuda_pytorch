// Common CUDA utilities for the cuda_core lab.
//
// Every experiment reuses three things: error checking (fail fast), timing
// (CUDA events for device time, steady_clock for wall time) and device
// property reporting.  Keeping them in one header makes the experiments
// readable and keeps the measurement policy explicit.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,         \
                         __LINE__, cudaGetErrorString(_err));                   \
            std::exit(2);                                                       \
        }                                                                       \
    } while (0)

namespace cuda_lab {

// Wall-clock timer in milliseconds (host side).
struct WallTimer {
    using clock = std::chrono::steady_clock;
    clock::time_point t0;

    void start() { t0 = clock::now(); }
    double ms() const {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    }
};

// CUDA event timer in milliseconds (device side).  The device must be a CUDA
// device; events only measure GPU execution time, not host launch overhead.
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

    void start(cudaStream_t stream = 0) { CUDA_CHECK(cudaEventRecord(start_ev, stream)); }
    void stop(cudaStream_t stream = 0) { CUDA_CHECK(cudaEventRecord(stop_ev, stream)); }
    // Returns elapsed device time in milliseconds.
    double ms() {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_ev, stop_ev));
        return static_cast<double>(ms);
    }
};

// Print key device properties so every report is anchored to the hardware.
inline void print_device_info() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    // clockRate / memoryClockRate / memoryBusWidth were removed from
    // cudaDeviceProp in CUDA 13; query them as attributes instead.
    int clock_rate = 0, mem_clock_rate = 0, bus_width = 0;
    cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, 0);
    cudaDeviceGetAttribute(&mem_clock_rate, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&bus_width, cudaDevAttrGlobalMemoryBusWidth, 0);
    std::printf("device: %s\n", prop.name);
    std::printf("sm: %d  max_threads_per_sm: %d  regs_per_block: %d\n",
                prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor,
                prop.regsPerBlock);
    std::printf("shared_mem_per_block: %zu bytes  shared_mem_per_sm: %zu bytes\n",
                prop.sharedMemPerBlock, prop.sharedMemPerMultiprocessor);
    std::printf("clock_rate_khz: %d  memory_clock_rate_khz: %d  bus_width: %d\n",
                clock_rate, mem_clock_rate, bus_width);
    std::printf("l2_cache_size: %d bytes  unified_addressing: %d\n",
                prop.l2CacheSize, prop.unifiedAddressing);
}

// Minimal JSON value emitter (only string and number needed here).
inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else out += c;
    }
    return out;
}

// Simple key-value JSON object builder for experiment results.
struct JsonReport {
    std::string buf;
    bool first = true;

    void begin() { buf = "{"; first = true; }
    void put(const std::string& key, const std::string& value) {
        if (!first) buf += ", ";
        buf += "\"" + json_escape(key) + "\": \"" + json_escape(value) + "\"";
        first = false;
    }
    void put(const std::string& key, double value) {
        if (!first) buf += ", ";
        char tmp[64];
        std::snprintf(tmp, sizeof(tmp), "%.6f", value);
        buf += "\"" + json_escape(key) + "\": " + tmp;
        first = false;
    }
    void put(const std::string& key, long long value) {
        if (!first) buf += ", ";
        char tmp[64];
        std::snprintf(tmp, sizeof(tmp), "%lld", value);
        buf += "\"" + json_escape(key) + "\": " + tmp;
        first = false;
    }
    std::string end() {
        buf += "}\n";
        return buf;
    }
};

// Median of a sample vector (for p50; p90/p95/p99 are less useful for these
// micro-probes but can be added by sorting the same vector).
inline double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2 == 1) return v[n / 2];
    return (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

inline double mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

}  // namespace cuda_lab
