#include "timer.h"
#include "cache_bench.h"
#include "numa_bench.h"
#include "simd_bench.h"
#include "sys_info.h"

#include <iostream>

// 防优化藏身变量：基准测试将计算结果写入此处，防止编译器将整个计算过程
// 当作死代码优化删除。声明为 volatile 使得编译器无法假定该值永远不会被
// 读取，从而必须保留所有产生副作用的计算过程。
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
