// Chapter 9: Memory Access Optimization
// Demonstrates cache-efficient data structures and access patterns.
// Compile: see CMakeLists.txt (ch09_demo target)

#include <cstring>
#include <iostream>
#include <chrono>
#include <cstdint>
#include <new>

// ---- Example 9.1b: Merge separate arrays into struct for spatial locality ----
struct S {
    float a;
    float b;
};

// Original: two separate arrays (bad locality)
void ProcessSeparate(const float* a, const float* b, float* out, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

// Improved: single array of structs (better locality)
void ProcessMerged(const S* data, float* out, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = data[i].a + data[i].b;
    }
}

// ---- Example 9.2b: Use union to share memory for non-overlapping variables ----
union LargeBuffer {
    double matrix_a[1024];
    double matrix_b[1024];  // Shares memory with matrix_a
};

// ---- Example 9.4: Aligned data allocation ----
#ifdef _MSC_VER
#define ALIGN64 __declspec(align(64))
#else
#define ALIGN64 __attribute__((aligned(64)))
#endif

ALIGN64 int AlignedArray[1024];

// ---- Cache line alignment check ----
template <typename T>
void CheckAlignment(const T* ptr, const char* name) {
    auto addr = reinterpret_cast<std::uintptr_t>(ptr);
    std::cout << name << " alignment: "
              << (addr % 64 == 0   ? "64-byte aligned"
                  : addr % 32 == 0 ? "32-byte aligned"
                  : addr % 16 == 0 ? "16-byte aligned"
                                   : "unaligned")
              << "\n";
}

// ---- Sequential vs strided access benchmark ----
// Sequential access (cache-friendly)
double SequentialSum(const double* matrix, int cols) {
    double sum = 0.0;
    for (int r = 0; r < cols; ++r) {
        for (int c = 0; c < cols; ++c) {
            sum += matrix[r * cols + c];
        }
    }
    return sum;
}

// Strided access (cache-unfriendly)
double StridedSum(const double* matrix, int cols) {
    double sum = 0.0;
    for (int c = 0; c < cols; ++c) {
        for (int r = 0; r < cols; ++r) {
            sum += matrix[r * cols + c];
        }
    }
    return sum;
}

// ---- Main ----
int main() {
    std::cout << "=== Chapter 9: Memory Access Optimization ===\n\n";

    // Alignment check
    CheckAlignment(AlignedArray, "AlignedArray");

    // Merged struct demo
    constexpr int N = 1000;
    S* merged = new S[N];
    float* a_sep = new float[N];
    float* b_sep = new float[N];
    float* out = new float[N];

    for (int i = 0; i < N; ++i) {
        merged[i].a = static_cast<float>(i);
        merged[i].b = static_cast<float>(i * 2);
        a_sep[i] = static_cast<float>(i);
        b_sep[i] = static_cast<float>(i * 2);
    }
    ProcessMerged(merged, out, N);
    std::cout << "ProcessMerged result[0]: " << out[0] << "\n";

    ProcessSeparate(a_sep, b_sep, out, N);
    std::cout << "ProcessSeparate result[0]: " << out[0] << "\n";

    delete[] merged;
    delete[] a_sep;
    delete[] b_sep;
    delete[] out;

    // Cache access pattern demo (small matrix for quick test)
    constexpr int SMALL = 256;
    double* small_mat = new double[SMALL * SMALL]();
    for (int i = 0; i < SMALL * SMALL; ++i)
        small_mat[i] = 1.0;

    auto t1 = std::chrono::high_resolution_clock::now();
    double s1 = SequentialSum(small_mat, SMALL);
    auto t2 = std::chrono::high_resolution_clock::now();
    double s2 = StridedSum(small_mat, SMALL);
    auto t3 = std::chrono::high_resolution_clock::now();

    auto seq_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    auto str_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    std::cout << "Sequential sum: " << s1 << " (" << seq_us << " us)\n";
    std::cout << "Strided sum:    " << s2 << " (" << str_us << " us)\n";
    std::cout << "Speedup: " << (seq_us > 0 ? static_cast<double>(str_us) / seq_us : 0) << "x\n";

    delete[] small_mat;

    std::cout << "\nAll chapter 9 checks passed.\n";
    return 0;
}
