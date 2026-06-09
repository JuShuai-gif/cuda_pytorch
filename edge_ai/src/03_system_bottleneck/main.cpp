#include "cache_bench.h"
#include "lock_bench.h"
#include "memory_bench.h"
#include "timer.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <chrono>
#include <ctime>

void print_header(const std::string &title) {
    std::cout << "\n"
              << std::string(72, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(72, '=') << "\n";
}

// ============================================================================
// 将所有基准测试结果写入 bottleneck_metrics.json
// ============================================================================
static void write_metrics_json() {
    std::ofstream of("bottleneck_metrics.json");

    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);

    of << "{\n";
    of << "  \"timestamp\": \"" << std::ctime(&time_t_now);
    // 从 ctime 输出中移除换行符
    of.seekp(-1, std::ios_base::cur);
    of << "\",\n";
    of << "  \"cache_line_size_bytes\": " << estimate_cache_line_size() << ",\n";

    of << "  \"benchmarks\": {\n";
    of << "    \"false_sharing\": {\n";
    of << "      \"description\": \"对 8 个跟踪运行卡尔曼滤波预测，比较未填充与已填充缓存行布局\",\n";
    of << "      \"workload\": \"8 个卡尔曼跟踪, 6 状态滤波器, 500K 次迭代\",\n";
    of << "      \"impact\": \"填充到缓存行边界可以消除伪共享\"\n";
    of << "    },\n";
    of << "    \"cache_thrashing\": {\n";
    of << "      \"description\": \"在 640x480x3 图像上进行 3x3 盒式模糊，行主序 vs 列主序遍历\",\n";
    of << "      \"workload\": \"640x480x3 float 图像, 3x3 卷积核\",\n";
    of << "      \"impact\": \"行主序是缓存友好的; 列主序导致每次访问都缓存未命中\"\n";
    of << "    },\n";
    of << "    \"lock_contention\": {\n";
    of << "      \"description\": \"在 800 个检测框上使用自旋锁 vs 互斥锁 vs 无锁分区方式运行 NMS\",\n";
    of << "      \"workload\": \"800 个框, 500 轮, 4 个线程, IoU 阈值 0.5\",\n";
    of << "      \"impact\": \"无锁分区方案通过处理不相交的锚框集消除竞争\"\n";
    of << "    },\n";
    of << "    \"memory_copy\": {\n";
    of << "      \"description\": \"1920x1080x3 相机帧拷贝 vs 零拷贝指针交换 vs 环形缓冲区\",\n";
    of << "      \"workload\": \"6.2 MB 帧大小, 8 槽位环形缓冲区, 1000 次迭代\",\n";
    of << "      \"impact\": \"零拷贝指针交换是 O(1); 带交换的环形缓冲区分摊了分配开销\"\n";
    of << "    }\n";
    of << "  }\n";
    of << "}\n";
    of.close();
    std::cout << "\n指标已写入 bottleneck_metrics.json\n";
}

int main() {
    std::cout << "============================================================\n";
    std::cout << "  系统瓶颈识别 - 机器人工作负载\n";
    std::cout << "  缓存行大小: " << estimate_cache_line_size() << " 字节\n";
    std::cout << "============================================================\n";

    demo_false_sharing();
    demo_lock_contention();
    demo_cache_thrashing();
    demo_memory_copy();

    write_metrics_json();

    std::cout << "\n所有基准测试已完成。\n";
    return 0;
}
