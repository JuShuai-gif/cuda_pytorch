#include "simd_bench.h"
#include "timer.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#ifdef __AVX2__
#include <immintrin.h>
#endif

void demo_simd_optimization() {
    print_header("演示 6: SIMD 优化 (AVX2)");

#ifdef __AVX2__
    constexpr size_t N = 1'000'000;
    // 堆分配以避免栈溢出 (4 x 4MB = 16MB > 默认 8MB 栈)
    float *a = static_cast<float *>(std::aligned_alloc(32, N * sizeof(float)));
    float *b = static_cast<float *>(std::aligned_alloc(32, N * sizeof(float)));
    float *c_scalar = static_cast<float *>(std::aligned_alloc(32, N * sizeof(float)));
    float *c_simd = static_cast<float *>(std::aligned_alloc(32, N * sizeof(float)));

    if (!a || !b || !c_scalar || !c_simd) {
        std::cout << "  无法分配对齐内存。\n";
        std::free(a);
        std::free(b);
        std::free(c_scalar);
        std::free(c_simd);
        return;
    }

    // 初始化
    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i) * 0.001f;
        b[i] = static_cast<float>(i) * 0.002f;
    }

    // 标量: c = a * b + a
    {
        Timer t;
        t.start();
        for (size_t i = 0; i < N; ++i) {
            c_scalar[i] = a[i] * b[i] + a[i];
        }
        double ms = t.elapsed_ms();
        std::cout << "  标量（循环）: c = a*b + a\n";
        std::cout << "    耗时: " << std::fixed << std::setprecision(3)
                  << ms << " ms\n";
        std::cout << "    吞吐量: " << std::fixed << std::setprecision(1)
                  << (N / ms * 1000.0) << " 元素/秒\n\n";
    }

    // SIMD: 使用 AVX2 FMA 计算 c = a * b + a
    {
        Timer t;
        t.start();
        for (size_t i = 0; i < N; i += 8) {
            __m256 va = _mm256_load_ps(&a[i]);
            __m256 vb = _mm256_load_ps(&b[i]);
            __m256 vc = _mm256_fmadd_ps(va, vb, va); // vc = va*vb + va
            _mm256_store_ps(&c_simd[i], vc);
        }
        double ms = t.elapsed_ms();
        std::cout << "  AVX2 (FMA): c = a*b + a，每次处理 8 个元素\n";
        std::cout << "    耗时: " << std::fixed << std::setprecision(3)
                  << ms << " ms\n";
        std::cout << "    吞吐量: " << std::fixed << std::setprecision(1)
                  << (N / ms * 1000.0) << " 元素/秒\n\n";
    }

    // SIMD 归约（点积）
    {
        Timer t;
        t.start();
        __m256 sum_vec = _mm256_setzero_ps();
        for (size_t i = 0; i < N; i += 8) {
            __m256 va = _mm256_load_ps(&a[i]);
            __m256 vb = _mm256_load_ps(&b[i]);
            sum_vec = _mm256_fmadd_ps(va, vb, sum_vec);
        }
        // 水平求和
        alignas(32) float sum_arr[8];
        _mm256_store_ps(sum_arr, sum_vec);
        float dot = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3]
                    + sum_arr[4] + sum_arr[5] + sum_arr[6] + sum_arr[7];
        double ms = t.elapsed_ms();
        volatile float sink = dot;
        (void)sink;

        std::cout << "  AVX2 点积:\n";
        std::cout << "    耗时: " << std::fixed << std::setprecision(3)
                  << ms << " ms\n";
        std::cout << "    结果: " << dot << "\n";
    }

    // 验证正确性
    double max_err = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double err = std::abs(c_scalar[i] - c_simd[i]);
        if (err > max_err) max_err = err;
    }
    std::cout << "\n    最大误差（标量 vs SIMD）: " << std::scientific
              << max_err << "\n";

    std::cout << "\n  => SIMD 每条指令处理 8 个浮点数（256 位 AVX2）。\n"
              << "  编译器在 -O2/-O3 下可能会对标量循环进行自动向量化。\n";

    std::free(a);
    std::free(b);
    std::free(c_scalar);
    std::free(c_simd);

#else
    std::cout << "  此 CPU 上 AVX2 不可用，或缺少编译选项。\n";
    std::cout << "  启用方法: 在编译选项中添加 -mavx2 -mfma。\n";
#endif
}
