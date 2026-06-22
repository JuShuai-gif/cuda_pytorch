// 03_num_threads.cpp
// 知识点: std::thread::hardware_concurrency() 检测硬件线程数
// 演示: 获取硬件并行度，创建最佳数量线程执行并行计算任务

#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

// 线程安全的进度打印器
// 生产环境中应使用完善的日志库(如 spdlog)，此处为演示目的
class ProgressPrinter {
public:
    void print(int thread_id, int progress) {
        // 简单打印，实际应使用互斥量保护 std::cout
        std::cout << "[线程 " << std::setw(2) << thread_id << "] 进度: "
                  << std::setw(3) << progress << "%\n";
    }
};

// 计算指定范围内的质数个数 (CPU密集型任务)
int fct_count_primes(int start, int end) {
    int count = 0;
    for (int n = std::max(2, start); n <= end; ++n) {
        bool is_prime = true;
        for (int d = 2; d * d <= n; ++d) {
            if (n % d == 0) {
                is_prime = false;
                break;
            }
        }
        if (is_prime) {
            ++count;
        }
    }
    return count;
}

int main() {
    // hardware_concurrency() 返回硬件支持的并发线程数
    // 注意: 返回值可能为 0 (无法确定时)
    const unsigned int hw_threads = std::thread::hardware_concurrency();
    const unsigned int num_threads =
        (hw_threads > 0) ? hw_threads : 2;  // 回退到2个线程

    std::cout << "=== 硬件线程数检测 ===\n";
    std::cout << "std::thread::hardware_concurrency() = " << hw_threads << "\n";
    std::cout << "实际使用线程数: " << num_threads << "\n\n";

    // 并行计算: 统计 2 ~ 1'000'000 范围内的质数个数
    const int  total_range = 1'000'000;
    const int  chunk_size  = total_range / num_threads;
    const int  remainder   = total_range % num_threads;

    std::cout << "任务: 统计 2 ~ " << total_range << " 内的质数个数\n";
    std::cout << "每个线程处理约 " << chunk_size << " 个数\n\n";

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // 每个线程的局部结果
    std::vector<int> partial_results(num_threads, 0);

    auto start_time = std::chrono::high_resolution_clock::now();

    // 创建线程: 每个线程处理一个子范围
    int range_start = 2;
    for (unsigned int i = 0; i < num_threads; ++i) {
        int range_end =
            range_start + chunk_size - 1 + (i < remainder ? 1 : 0);

        threads.emplace_back(
            [i, range_start, range_end, &partial_results]() {
                std::cout << "[线程 " << std::setw(2) << i
                          << "] 处理范围: " << range_start << " ~ "
                          << range_end << "\n";
                partial_results[i] = fct_count_primes(range_start, range_end);
                std::cout << "[线程 " << std::setw(2) << i
                          << "] 完成, 找到 " << partial_results[i]
                          << " 个质数\n";
            });

        range_start = range_end + 1;
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    // 汇总结果
    int total_primes =
        std::accumulate(partial_results.begin(), partial_results.end(), 0);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    std::cout << "\n=== 结果汇总 ===\n";
    std::cout << "线程数: " << num_threads << "\n";
    std::cout << "总质数个数: " << total_primes << "\n";
    std::cout << "总耗时: " << duration.count() << "ms\n";
    std::cout << "各线程结果: ";
    for (unsigned int i = 0; i < num_threads; ++i) {
        std::cout << "T" << i << "=" << partial_results[i]
                  << (i + 1 < num_threads ? ", " : "");
    }
    std::cout << "\n";

    // 最佳实践提示
    std::cout << "\n=== 最佳实践 ===\n";
    std::cout << "1. hardware_concurrency() 可能返回0，需要回退值\n";
    std::cout << "2. 线程数不应远超硬件线程数(CPU密集型任务)\n";
    std::cout << "3. IO密集型任务可以创建更多线程\n";
    std::cout << "4. 过度创建线程会导致上下文切换开销\n";

    return 0;
}
