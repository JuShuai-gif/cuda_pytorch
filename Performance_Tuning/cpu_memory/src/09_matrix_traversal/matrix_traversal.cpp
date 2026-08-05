// Experiment 09: Matrix traversal.
//
// Compares row-major traversal, column-major traversal, and a transposed
// (row-major on transposed copy) traversal of a large matrix.
// Row-major is expected to be much faster due to spatial locality.
//
// Reference: PDF 6.2.1 (matrix multiplication, Figure 6.1).

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#include "benchmark.h"

static constexpr int N = 4096;   // 4096x4096 doubles = 128 MB
static constexpr int kRounds = 5;

int main() {
    std::printf("Experiment 09: matrix traversal (%dx%d doubles)\n", N, N);
    std::vector<double> A((size_t)N * N, 1.0);

    auto row_major = [&] {
        double s = 0;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) s += A[(size_t)i * N + j];
        bm::do_not_optimize(s);
    };
    auto col_major = [&] {
        double s = 0;
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) s += A[(size_t)i * N + j];
        bm::do_not_optimize(s);
    };
    auto transpose = [&] {
        std::vector<double> T((size_t)N * N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) T[(size_t)j * N + i] = A[(size_t)i * N + j];
        double s = 0;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) s += T[(size_t)i * N + j];
        bm::do_not_optimize(s);
    };

    struct Mode { const char* name; std::function<void()> fn; };
    Mode modes[] = {{"row_major", row_major}, {"col_major", col_major}, {"transposed", transpose}};

    std::printf("%-14s %-12s %-14s\n", "mode", "time_ms", "GB/s");
    for (auto& m : modes) {
        m.fn();
        auto res = bm::time_rounds(kRounds, m.fn);
        double bytes = (double)N * N * sizeof(double);
        double gbps = bytes / (res.median_ms * 1e6 / 1e9) / 1e9;
        std::printf("%-14s %-12.3f %-14.3f\n", m.name, res.median_ms, gbps);
    }
    return 0;
}
