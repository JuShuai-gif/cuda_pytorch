// 02_openmp_parallel_reduce.cpp — OpenMP 归约与调度策略对比
// 编译需要 -fopenmp

#ifdef _OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using namespace std::chrono_literals;

// ===== 1. 手动并行归约 vs OpenMP reduction =====
void demo_manual_vs_omp_reduce() {
    std::cout << "=== 手动归约 vs OpenMP reduction ===\n";

    const size_t kSize = 10'000'000;
    std::vector<double> data(kSize);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (auto& v : data) v = dist(rng);

    // OpenMP reduction
#ifdef _OPENMP
    {
        double sum = 0.0;
        auto start = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for reduction(+:sum)
        for (size_t i = 0; i < kSize; ++i) {
            sum += data[i];
        }

        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);
        std::cout << "  OpenMP reduction: " << elapsed.count()
                  << " ms | sum=" << std::fixed
                  << std::setprecision(6) << sum << "\n";
    }
#endif

    // 串行
    {
        double sum = 0.0;
        auto start = std::chrono::high_resolution_clock::now();

        sum = std::accumulate(data.begin(), data.end(), 0.0);

        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);
        std::cout << "  串行 accumulate:  " << elapsed.count()
                  << " ms | sum=" << std::fixed
                  << std::setprecision(6) << sum << "\n";
    }
}

// ===== 2. schedule 调度策略对比 =====
void demo_schedule_policies() {
    std::cout << "\n=== schedule 调度策略 ===\n";

    const int kSize = 1000;
    std::vector<int> work(kSize);

    // 模拟不均匀负载: 前面大、后面小
    for (int i = 0; i < kSize; ++i) {
        work[i] = kSize - i; // 从 1000 递减到 1
    }

    std::vector<std::string> policies = {"static", "dynamic", "guided"};
    std::vector<std::string> chunks = {"1", "10"};

#ifdef _OPENMP
    for (const auto& policy : policies) {
        for (const auto& chunk : chunks) {
            long long total = 0;
            auto start = std::chrono::high_resolution_clock::now();

            #pragma omp parallel for \
                schedule(runtime) reduction(+:total)
            for (int i = 0; i < kSize; ++i) {
                // 模拟 work[i] 次操作
                long long local = 0;
                for (int j = 0; j < work[i]; ++j) {
                    local += j;
                }
                total += local;
            }

            auto elapsed =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start);
            std::cout << "  " << policy << "(chunk=" << chunk << "): "
                      << elapsed.count() << " us\n";
        }
    }
#else
    std::cout << "  OpenMP 未启用\n";
#endif

    std::cout << "\n  说明:\n";
    std::cout << "  static:  编译时分配，适合均匀负载\n";
    std::cout << "  dynamic: 运行时动态分配，适合不均匀负载\n";
    std::cout << "  guided:  逐渐减小 chunk，适应力最强\n";
}

// ===== 3. 嵌套并行 =====
void demo_nested_parallel() {
    std::cout << "\n=== 嵌套并行 ===\n";

#ifdef _OPENMP
    // 启用嵌套并行
    omp_set_nested(1);
    omp_set_max_active_levels(2);

    #pragma omp parallel num_threads(2)
    {
        int outer_tid = omp_get_thread_num();

        #pragma omp parallel num_threads(2)
        {
            int inner_tid = omp_get_thread_num();
            #pragma omp critical
            std::cout << "  Outer " << outer_tid
                      << " / Inner " << inner_tid << "\n";
        }
    }

    std::cout << "  注: 嵌套并行需要 omp_set_nested(1)\n";
    std::cout << "  总线程数 = 外层线程数 × 内层线程数\n";
#else
    std::cout << "  OpenMP 未启用\n";
#endif
}

// ===== 4. nowait 子句 =====
void demo_nowait() {
    std::cout << "\n=== nowait 消除隐式屏障 ===\n";

#ifdef _OPENMP
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < 1000000; ++i) {
            volatile int x = i * i;
            (void)x;
        }
        // nowait: 完成后不等待其他线程，直接继续
        #pragma omp single
        std::cout << "  线程 " << omp_get_thread_num()
                  << " 完成 for 循环 (nowait)\n";
    }

    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);
    std::cout << "  nowait 可以让先完成的线程不等待屏障\n";
#endif
}

int main() {
    demo_manual_vs_omp_reduce();
    demo_schedule_policies();
    demo_nested_parallel();
    demo_nowait();

    std::cout << "\n使用 export OMP_SCHEDULE=\"dynamic,10\" 设置默认调度策略\n";
    return 0;
}
