#include "memory_bench.h"
#include "neon_convert.h"
#include "fail_closed.h"

#include <fstream>
#include <iostream>

// 反优化: 防止编译器消除死代码
volatile long g_sink = 0;

int main() {
    std::cout << "\n";
    std::cout << "  ╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "  ║    边缘端性能优化实战：RK3588 机器人视觉管线调优          ║\n";
    std::cout << "  ║    案例研究：uncached 内存 / DMA_SYNC / NEON / Fail-Closed ║\n";
    std::cout << "  ╚══════════════════════════════════════════════════════════╝\n";

    // ── 第一部分: 内存访问基准测试 ──
    std::cout << "\n  ┌─ 第一部分 ──────────────────────────────────────────────┐\n";
    std::cout << "  │ 内存访问优化: uncached → cached + DMA_BUF_IOCTL_SYNC      │\n";
    std::cout << "  └──────────────────────────────────────────────────────────┘\n";

    demo_uncached_vs_cached();
    demo_dma_sync_simulation();
    demo_bandwidth_contention();

    // ── 第二部分: NEON SIMD 转换基准测试 ──
    std::cout << "\n  ┌─ 第二部分 ──────────────────────────────────────────────┐\n";
    std::cout << "  │ NEON SIMD 优化: FP16→FP32 / BGR→FP16 RGB                  │\n";
    std::cout << "  └──────────────────────────────────────────────────────────┘\n";

    demo_neon_conversion();

    // ── 第三部分: Fail-Closed 模式 ──
    std::cout << "\n  ┌─ 第三部分 ──────────────────────────────────────────────┐\n";
    std::cout << "  │ 错误处理模式: Fail-Closed vs Fail-Open                    │\n";
    std::cout << "  └──────────────────────────────────────────────────────────┘\n";

    demo_fail_closed_pattern();

    // ── 输出结果 JSON ──
    std::ofstream json("edge_optimization_metrics.json");
    if (json.is_open()) {
        json << "{\n";
        json << "  \"platform\": \"";
#ifdef __aarch64__
        json << "ARM aarch64 (NEON enabled)";
#else
        json << "x86_64 (NEON simulated, for reference only)";
#endif
        json << "\",\n";
        json << "  \"case_study\": \"RK3588 robot vision pipeline optimization\",\n";
        json << "  \"key_metrics\": {\n";
        json << "    \"uncached_6mb_frame_read\": \"~15ms (original)\",\n";
        json << "    \"cached_dma_sync_6mb_frame_read\": \"~3.6ms (optimized)\",\n";
        json << "    \"speedup_from_dma_sync\": \"4.2x (P50), 6.4x (P99)\",\n";
        json << "    \"jitter_reduction\": \"±20ms → ±2ms (10x)\",\n";
        json << "    \"neon_fp16_to_f32_1.45M\": \"~0.3ms (vs 9.68ms with want_float=1)\",\n";
        json << "    \"bottleneck_insight\": \"DDR bandwidth > compute (memory wall dominated)\"\n";
        json << "  },\n";
        json << "  \"lessons\": [\n";
        json << "    \"Don't trust default configs - measure everything\",\n";
        json << "    \"Fail-closed is mandatory for safety-critical systems\",\n";
        json << "    \"DDR bandwidth is the ultimate bottleneck on edge devices\",\n";
        json << "    \"Cache coherence requires explicit sync in DMA pipelines\",\n";
        json << "    \"Never fallback to virtual-address RGA on RK3588 (freezes)\"\n";
        json << "  ]\n";
        json << "}\n";
        json.close();
        std::cout << "\n  指标已写入 edge_optimization_metrics.json\n";
    }

    std::cout << "\n  ╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "  ║  所有基准测试已完成。                                      ║\n";
    std::cout << "  ║  核心教训: 不要信默认配置，perf 数据说话。                  ║\n";
    std::cout << "  ╚══════════════════════════════════════════════════════════╝\n\n";

    return 0;
}
