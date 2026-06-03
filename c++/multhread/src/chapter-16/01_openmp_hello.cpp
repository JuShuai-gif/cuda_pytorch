// 01_openmp_hello.cpp — OpenMP 入门
// 演示: parallel, for, critical, atomic, barrier
// 编译需要 -fopenmp

#ifdef _OPENMP
#include <omp.h>
#endif
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// ===== 1. Hello OpenMP =====
void demo_hello_openmp() {
    std::cout << "=== 1. Hello OpenMP ===\n";

#ifdef _OPENMP
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        #pragma omp critical
        std::cout << "  线程 " << tid << "/" << nthreads << "\n";
    }
    std::cout << "  最大线程数: " << omp_get_max_threads() << "\n";
#else
    std::cout << "  OpenMP 未启用 (编译时请加 -fopenmp)\n";
#endif
}

// ===== 2. parallel for — 最常用的并行模式 =====
void demo_parallel_for() {
    std::cout << "\n=== 2. parallel for ===\n";

    const int kSize = 20;
    std::vector<int> a(kSize), b(kSize), c(kSize);

    for (int i = 0; i < kSize; ++i) {
        a[i] = i;
        b[i] = i * 2;
    }

#ifdef _OPENMP
    #pragma omp parallel for
    for (int i = 0; i < kSize; ++i) {
        c[i] = a[i] + b[i];
    }
#else
    for (int i = 0; i < kSize; ++i) {
        c[i] = a[i] + b[i];
    }
#endif

    std::cout << "  结果验证: ";
    bool ok = true;
    for (int i = 0; i < kSize; ++i) {
        if (c[i] != a[i] + b[i]) { ok = false; break; }
    }
    std::cout << (ok ? "OK" : "FAIL") << "\n";
}

// ===== 3. reduction 归约 =====
void demo_reduction() {
    std::cout << "\n=== 3. reduction 归约 ===\n";

    const long long kSize = 10'000'000;
    std::vector<int> data(kSize, 1);

    // OpenMP 版本
#ifdef _OPENMP
    {
        long long sum = 0;
        auto start = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for reduction(+:sum)
        for (long long i = 0; i < kSize; ++i) {
            sum += data[i];
        }

        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);
        std::cout << "  OpenMP reduction: " << elapsed.count()
                  << " ms | sum=" << sum
                  << " (期望 " << kSize << ")\n";
    }
#endif

    // 串行版本
    {
        long long sum = 0;
        auto start = std::chrono::high_resolution_clock::now();

        for (long long i = 0; i < kSize; ++i) {
            sum += data[i];
        }

        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);
        std::cout << "  串行版本:         " << elapsed.count()
                  << " ms | sum=" << sum << "\n";
    }
}

// ===== 4. critical vs atomic =====
void demo_critical_atomic() {
    std::cout << "\n=== 4. critical vs atomic ===\n";

    int counter = 0;

#ifdef _OPENMP
    #pragma omp parallel for
    for (int i = 0; i < 1000; ++i) {
        #pragma omp atomic
        counter++;
    }

    std::cout << "  atomic counter: " << counter
              << " (期望 1000)\n";
#endif

    std::cout << "  atomic: 硬件原子指令，仅支持简单操作\n";
    std::cout << "  critical: 互斥临界区，支持任意代码块\n";
}

// ===== 5. parallel sections (任务并行) =====
void demo_parallel_sections() {
    std::cout << "\n=== 5. parallel sections (任务并行) ===\n";

#ifdef _OPENMP
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            std::cout << "    Section A: 线程 "
                      << omp_get_thread_num() << "\n";
        }
        #pragma omp section
        {
            std::cout << "    Section B: 线程 "
                      << omp_get_thread_num() << "\n";
        }
        #pragma omp section
        {
            std::cout << "    Section C: 线程 "
                      << omp_get_thread_num() << "\n";
        }
    }
#endif
}

int main() {
#ifdef _OPENMP
    std::cout << "OpenMP 版本: " << _OPENMP << " ("
              << (_OPENMP >= 201511 ? "4.5+" :
                  _OPENMP >= 201307 ? "4.0" :
                  _OPENMP >= 201107 ? "3.1" : "旧版")
              << ")\n\n";
#else
    std::cout << "OpenMP 未启用。使用 -fopenmp 重新编译。\n\n";
#endif

    demo_hello_openmp();
    demo_parallel_for();
    demo_reduction();
    demo_critical_atomic();
    demo_parallel_sections();

    std::cout << "\n小提示:\n";
    std::cout << "  export OMP_NUM_THREADS=4  # 设置 OpenMP 线程数\n";
    std::cout << "  export OMP_SCHEDULE=dynamic # 设置调度策略\n";
    return 0;
}
