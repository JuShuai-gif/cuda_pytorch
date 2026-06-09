#include "timer.h"
#include "cache_bench.h"
#include "numa_bench.h"
#include "simd_bench.h"
#include "sys_info.h"

#include <iostream>

// 反优化出口变量的定义
volatile long g_sink = 0;

int main() {
    std::cout << "=====================================================\n";
    std::cout << "  计算机体系结构与内存层次基准测试\n";
    std::cout << "=====================================================\n";

    demo_cpu_info();
    demo_cache_line_detection();
    demo_cache_hit_vs_miss();
    demo_false_sharing();
    demo_numa_awareness();
    demo_row_vs_column();
    demo_simd_optimization();

    std::cout << "\n所有基准测试已完成。\n";
    return 0;
}
