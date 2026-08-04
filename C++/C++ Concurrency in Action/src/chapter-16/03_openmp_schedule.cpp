// 03_openmp_schedule.cpp — OpenMP schedule 策略深度对比
// 编译需要 -fopenmp

#ifdef _OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// ===== 手动实现不同 schedule 策略 (无需 OpenMP 也能编译) =====

// 模拟不均匀负载: 余弦波，有些迭代快、有些慢
int work_amount(int i, int total) {
    // 模拟: 中部迭代耗时多，两端耗时少
    double ratio = static_cast<double>(i) / total;
    double factor = 1.0 + 2.0 * std::sin(ratio * 3.14159);
    return static_cast<int>(1000 * factor);
}

// Static 分配
template <typename F>
double benchmark_static(int total, int chunk, int nthreads, F work) {
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::jthread> threads;
    threads.reserve(nthreads);

    int per_thread = total / nthreads;
    for (int t = 0; t < nthreads; ++t) {
        int begin = t * per_thread;
        int end = (t == nthreads - 1) ? total : begin + per_thread;
        threads.emplace_back([=]() {
            for (int i = begin; i < end; ++i) {
                work(i);
            }
        });
    }
    threads.clear();

    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);
    return elapsed.count();
}

// Dynamic 分配 (使用原子计数器模拟)
template <typename F>
double benchmark_dynamic(int total, int chunk, int nthreads, F work) {
    std::atomic<int> next_item{0};
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::jthread> threads;
    threads.reserve(nthreads);

    for (int t = 0; t < nthreads; ++t) {
        threads.emplace_back([&]() {
            while (true) {
                int begin = next_item.fetch_add(chunk);
                if (begin >= total) break;
                int end = std::min(begin + chunk, total);
                for (int i = begin; i < end; ++i) {
                    work(i);
                }
            }
        });
    }
    threads.clear();

    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);
    return elapsed.count();
}

void demo_schedule_comparison() {
    std::cout << "=== Schedule 策略对比 (不均匀负载) ===\n\n";

    const int kTotal = 10000;
    const int kThreads = 4;

    auto work_func = [&](int i) {
        int amount = work_amount(i, kTotal);
        volatile long long sum = 0;
        for (int j = 0; j < amount; ++j) sum += j;
        (void)sum;
    };

    std::cout << "负载分布: 中部重、两端轻 (cosine wave)\n\n";

    // 串行基准
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < kTotal; ++i) work_func(i);
        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);
        std::cout << "  串行:         " << elapsed.count() << " ms\n";
    }

    // Static 分配
    double t_static = benchmark_static(kTotal, 1, kThreads, work_func);
    std::cout << "  Static:       " << t_static << " ms "
              << "(线程间工作量可能不均)\n";

    // Dynamic chunk=1
    double t_dyn1 = benchmark_dynamic(kTotal, 1, kThreads, work_func);
    std::cout << "  Dynamic(1):   " << t_dyn1 << " ms "
              << "(负载均衡好，调度开销大)\n";

    // Dynamic chunk=100
    double t_dyn100 = benchmark_dynamic(kTotal, 100, kThreads, work_func);
    std::cout << "  Dynamic(100): " << t_dyn100 << " ms "
              << "(负载均衡与调度开销的折中)\n";

    std::cout << "\n结论:\n";
    std::cout << "  - 均匀负载: static 最快 (无调度开销)\n";
    std::cout << "  - 不均匀负载: dynamic 更快 (负载均衡 > 调度开销)\n";
    std::cout << "  - chunk 大小是权衡: 小 chunk=好均衡，大 chunk=低开销\n";

#ifdef _OPENMP
    std::cout << "\nOpenMP 等效用法:\n";
    std::cout << "  #pragma omp parallel for schedule(static)\n";
    std::cout << "  #pragma omp parallel for schedule(dynamic, 10)\n";
    std::cout << "  #pragma omp parallel for schedule(guided, 10)\n";
#endif
}

int main() {
    demo_schedule_comparison();
    return 0;
}
