// 02_concurrent_vs_parallel.cpp
// 知识点: 并发(Concurrency) vs 并行(Parallelism) 的区别
// 演示: CPU密集型任务(受核心数限制) vs IO模拟任务(可大量并发)，观察线程调度行为

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// 获取当前时间的字符串表示
std::string fct_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(
                   now.time_since_epoch()) %
               1000;
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&t), "%H:%M:%S") << '.' << std::setw(3)
        << std::setfill('0') << ms.count();
    return oss.str();
}

// CPU密集型任务: 纯计算，受限于CPU核心数，适合测量真实的并行度
void fct_cpu_bound(int id, int iterations) {
    volatile double result = 0.0;  // volatile 防止编译器优化掉循环
    auto            start  = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        result += std::sin(i) * std::cos(i);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "[" << fct_timestamp() << "] CPU任务-" << id << " 完成"
              << " (耗时: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << "ms, 结果: " << result << ")\n";
}

// IO模拟任务: 大量时间在等待(如网络/磁盘)，可大量并发运行
void fct_io_bound(int id, int wait_ms) {
    std::cout << "[" << fct_timestamp() << "] IO任务-" << id << " 开始\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
    std::cout << "[" << fct_timestamp() << "] IO任务-" << id << " 完成\n";
}

int main() {
    const unsigned int hw_threads = std::thread::hardware_concurrency();
    std::cout << "=== 并发 vs 并行 ===\n";
    std::cout << "硬件线程数: " << hw_threads << "\n\n";

    // --- 场景1: CPU密集型任务 ---
    // 如果任务数 > 硬件线程数，则存在"并发"(任务切换)
    // 如果任务数 <= 硬件线程数，可实现真正的"并行"(同时在不同核心运行)
    std::cout << "--- 场景1: CPU密集型任务 (并行取决于核心数) ---\n";
    {
        const int task_count    = static_cast<int>(hw_threads) + 2;
        const int work_per_task = 10'000'000;

        std::cout << "启动 " << task_count << " 个CPU密集型线程"
                  << " (硬件线程: " << hw_threads << ")\n";

        std::vector<std::thread> threads;
        threads.reserve(task_count);

        auto total_start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < task_count; ++i) {
            threads.emplace_back(fct_cpu_bound, i, work_per_task);
        }

        for (auto& t : threads) {
            t.join();
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        std::cout << "CPU任务总耗时: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         total_end - total_start)
                         .count()
                  << "ms\n";
    }

    std::cout << "\n--- 场景2: IO模拟任务 (大量并发) ---\n";
    {
        // IO任务主要时间在等待，可以远超硬件线程数的并发量
        const int task_count  = hw_threads * 4;
        const int wait_base   = 200;

        std::cout << "启动 " << task_count << " 个IO模拟线程"
                  << " (硬件线程: " << hw_threads << ")\n";

        std::vector<std::thread> threads;
        threads.reserve(task_count);

        auto total_start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < task_count; ++i) {
            threads.emplace_back(fct_io_bound, i, wait_base + i * 5);
        }

        for (auto& t : threads) {
            t.join();
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        std::cout << "IO任务总耗时: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         total_end - total_start)
                         .count()
                  << "ms\n";
        std::cout << "观察: 总耗时远小于各任务等待时间之和，"
                  << "说明任务被并发执行\n";
    }

    std::cout << "\n结论:\n";
    std::cout << "  - 并行(Parallelism): 多个任务在不同CPU核心上同时执行\n";
    std::cout << "  - 并发(Concurrency): 多个任务交替执行(时间片轮转)\n";
    std::cout << "  - CPU密集型受限于核心数，IO密集型可大量并发\n";

    return 0;
}
