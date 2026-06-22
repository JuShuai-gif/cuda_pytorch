/*
 * gpu_utilization.cpp
 * Chapter 12: Monitoring Deployed Models
 *
 * Resource utilization metrics reveal whether the serving stack is
 * making efficient use of hardware. This file covers:
 *   - Host memory (RSS via /proc/self/status)
 *   - GPU stats via NVML (mock if not available)
 *   - H2D/D2H copy volume tracking
 *   - Utilization interpretation guide
 *
 * PDF pages: 476-478 (book pp. 476-478)
 *
 * Key interpretation rules:
 *   GPU util ~30% + high p95 + avg batch ~1: under-batching
 *   VRAM < 5% free + OOMs: near memory ceiling, quantize/prune
 *   CPU iowait surges: preprocessing/storage bottleneck
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ================================================================
// 1. RSS from /proc/self/status (PDF p. 476)
// ================================================================

size_t read_rss_kb() {
    std::ifstream f("/proc/self/status");
    if (!f.is_open()) return 0;
    std::string key;
    size_t value = 0;
    while (f >> key >> value) {
        if (key == "VmRSS:") return value; // in kB
    }
    return 0;
}

// ================================================================
// 2. CPU usage estimation via /proc/stat (PDF p. 476)
// ================================================================

struct CpuTimes {
    uint64_t user, nice, system, idle, iowait, irq, softirq, steal;
    uint64_t total() const {
        return user + nice + system + idle + iowait + irq + softirq + steal;
    }
};

CpuTimes read_cpu_times() {
    CpuTimes t{};
    std::ifstream f("/proc/stat");
    std::string cpu_label;
    f >> cpu_label; // "cpu"
    f >> t.user >> t.nice >> t.system >> t.idle
        >> t.iowait >> t.irq >> t.softirq >> t.steal;
    return t;
}

double cpu_utilization(const CpuTimes &prev, const CpuTimes &curr) {
    uint64_t total_diff = curr.total() - prev.total();
    uint64_t idle_diff = curr.idle - prev.idle;
    if (total_diff == 0) return 0.0;
    return 100.0 * (1.0 - static_cast<double>(idle_diff) / static_cast<double>(total_diff));
}

double iowait_fraction(const CpuTimes &prev, const CpuTimes &curr) {
    uint64_t total_diff = curr.total() - prev.total();
    uint64_t io_diff = curr.iowait - prev.iowait;
    if (total_diff == 0) return 0.0;
    return 100.0 * static_cast<double>(io_diff) / static_cast<double>(total_diff);
}

// ================================================================
// 3. GPU stats (mock, PDF p. 476)
//    In production, use NVML
// ================================================================

#ifndef USE_NVML
// Mock GPU stats for demo purposes
struct GpuStats {
    unsigned util = 0;      // GPU utilization %
    unsigned mem_used = 0;  // MB
    unsigned mem_total = 0; // MB
};

GpuStats sample_gpu(int idx = 0) {
    // Mock: simulate realistic GPU values
    // In production: nvmlDeviceGetUtilizationRates / nvmlDeviceGetMemoryInfo
    GpuStats s;
    s.mem_total = 8192; // 8 GB
    s.mem_used = 3500 + (std::rand() % 2000);
    s.util = 35 + (std::rand() % 45);
    (void)idx;
    return s;
}

unsigned gpu_mem_free_mb(const GpuStats &s) {
    return s.mem_total - s.mem_used;
}
#else
// NVML implementation (requires nvidia-ml library)
#include <nvml.h>
struct GpuStats {
    unsigned util = 0;
    unsigned mem_used = 0;
    unsigned mem_total = 0;
};

GpuStats sample_gpu(int idx = 0) {
    static bool inited = (nvmlInit() == NVML_SUCCESS);
    (void)inited;
    nvmlDevice_t dev;
    nvmlDeviceGetHandleByIndex(static_cast<unsigned>(idx), &dev);
    nvmlUtilization_t u;
    nvmlDeviceGetUtilizationRates(dev, &u);
    nvmlMemory_t m;
    nvmlDeviceGetMemoryInfo(dev, &m);
    return {static_cast<unsigned>(u.gpu),
            static_cast<unsigned>(m.used / 1024 / 1024),
            static_cast<unsigned>(m.total / 1024 / 1024)};
}
#endif

// ================================================================
// 4. H2D/D2H copy volume tracking (PDF p. 477)
// ================================================================

struct CopyVolumeTracker {
    std::atomic<uint64_t> h2d_bytes{0};
    std::atomic<uint64_t> d2h_bytes{0};

    void record_h2d(uint64_t nbytes) {
        h2d_bytes.fetch_add(nbytes, std::memory_order_relaxed);
    }

    void record_d2h(uint64_t nbytes) {
        d2h_bytes.fetch_add(nbytes, std::memory_order_relaxed);
    }

    void report() const {
        auto h2d = h2d_bytes.load(std::memory_order_relaxed);
        auto d2h = d2h_bytes.load(std::memory_order_relaxed);
        std::cout << "  H2D: " << (h2d / 1024.0 / 1024.0) << " MB | ";
        std::cout << "D2H: " << (d2h / 1024.0 / 1024.0) << " MB | ";
        std::cout << "Ratio (D2H/H2D): "
                  << (h2d > 0 ? static_cast<double>(d2h) / h2d : 0.0) << "\n";
    }
};

// ================================================================
// 5. Utilization diagnostics
// ================================================================

struct UtilizationSnapshot {
    double cpu_pct;
    double iowait_pct;
    size_t rss_kb;
    unsigned gpu_util;
    unsigned gpu_mem_free;
    unsigned gpu_mem_total;
    size_t avg_batch_size;
    double p95_latency_ms;
};

std::string diagnose_utilization(const UtilizationSnapshot &u) {
    double vram_free_pct = u.gpu_mem_total > 0 ? 100.0 * u.gpu_mem_free / u.gpu_mem_total : 0.0;

    if (u.gpu_util < 40 && u.p95_latency_ms > 50 && u.avg_batch_size <= 2) {
        return "UNDER-BATCHING: GPU underused. Enable micro-batching (8ms delay).";
    }
    if (vram_free_pct < 5) {
        return "VRAM CRITICAL: Near OOM. Reduce batch, quantize (FP16/INT8), or use pooling allocator.";
    }
    if (u.iowait_pct > 10) {
        return "IO PRESSURE: High iowait. Cache preprocessed artifacts; split preprocessing into dedicated service.";
    }
    if (u.gpu_util > 85 && vram_free_pct < 10) {
        return "SATURATED: GPU near capacity. Consider scaling horizontally or model sharding.";
    }
    return "HEALTHY: Utilization within expected range.";
}

// ================================================================
// Main demonstration
// ================================================================

int main() {
    std::srand(42);
    std::cout << "=== Chapter 12: Resource Utilization ===\n\n";

    // --- Host memory (RSS) ---
    std::cout << "1. Host Memory (RSS)\n";
    size_t rss = read_rss_kb();
    std::cout << "   VmRSS: " << rss << " kB (" << (rss / 1024.0) << " MB)\n";

    // --- CPU utilization ---
    std::cout << "\n2. CPU Utilization (sampling /proc/stat)\n";
    CpuTimes t0 = read_cpu_times();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    // Simulate CPU work
    volatile double x = 0;
    for (int i = 0; i < 5000000; ++i) x += std::sqrt(static_cast<double>(i));
    (void)x;
    CpuTimes t1 = read_cpu_times();
    double cpu_pct = cpu_utilization(t0, t1);
    double io_pct = iowait_fraction(t0, t1);
    std::cout << "   CPU: " << std::fixed << std::setprecision(1) << cpu_pct << "% | ";
    std::cout << "iowait: " << io_pct << "%\n";

    // --- GPU stats (mock) ---
    std::cout << "\n3. GPU Stats (mock NVML)\n";
    GpuStats gpu = sample_gpu(0);
    std::cout << "   GPU Util: " << gpu.util << "%\n";
    std::cout << "   VRAM Used: " << gpu.mem_used << " MB / " << gpu.mem_total << " MB\n";
    std::cout << "   VRAM Free: " << gpu_mem_free_mb(gpu) << " MB ("
              << std::fixed << std::setprecision(1)
              << (100.0 * gpu_mem_free_mb(gpu) / gpu.mem_total) << "%)\n";
    std::cout << "   Note: Install nvidia-ml-dev and build with -DUSE_NVML=1 for real GPU metrics.\n";

    // --- H2D/D2H volume ---
    std::cout << "\n4. H2D/D2H Copy Volume\n";
    CopyVolumeTracker copies;
    // Simulate a few inference passes with different batch sizes
    for (int batch = 1; batch <= 8; batch <<= 1) {
        uint64_t input_bytes = batch * 224 * 224 * 3 * 4; // FP32 images
        uint64_t output_bytes = batch * 1000 * 4;         // 1000-class logits
        copies.record_h2d(input_bytes);
        copies.record_d2h(output_bytes);
    }
    copies.report();

    // --- Utilization diagnostics ---
    std::cout << "\n5. Utilization Diagnostic Scenarios\n";

    struct Scenario {
        std::string name;
        UtilizationSnapshot snap;
    };

    std::vector<Scenario> scenarios = {
        {"Under-batching",
         {20.0, 2.0, 500000, 30, 6000, 8192, 1, 85.0}},
        {"VRAM critical",
         {60.0, 1.0, 800000, 90, 300, 8192, 4, 35.0}},
        {"IO bottleneck",
         {25.0, 15.0, 1200000, 80, 2000, 8192, 4, 45.0}},
        {"Healthy",
         {55.0, 2.0, 600000, 78, 2500, 8192, 6, 28.0}},
    };

    for (auto &s : scenarios) {
        std::string d = diagnose_utilization(s.snap);
        std::cout << "  " << s.name << ": " << d << "\n";
    }

    // --- Operational tips ---
    std::cout << "\n6. Operational Tips\n";
    std::cout << "  - Track free VRAM as a gauge: if below 5%, reduce batch or quantize.\n";
    std::cout << "  - Watch CPU iowait: high values often mean preprocessing is the bottleneck.\n";
    std::cout << "  - Monitor RSS growth: steady increase suggests memory leak.\n";
    std::cout << "  - H2D/D2H ratio > 10: GPU is input-bound; consider pinned memory.\n";

    std::cout << "\n=== Utilization demo complete ===\n";
    return 0;
}
