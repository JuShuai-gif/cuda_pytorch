// Cache thrashing: the same nested loop with a different access order.
//
// The book (PDF p.106-107) shows that writing matrix[i][j] in a row-major
// layout is cache friendly (~40 ms on the author's machine), while
// matrix[j][i] forces every access to miss the L1 cache (~800 ms).
//
// The matrix is allocated on the heap (256 MB does not fit on the stack).
// A row-major readback makes the writes observable so the compiler cannot
// eliminate the fill loops.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "benchmark.hpp"

namespace {

constexpr std::size_t kL1CacheCapacity = 32768;
constexpr std::size_t kSize = kL1CacheCapacity / sizeof(int);

// Flat row-major buffer of kSize x kSize ints.
using Matrix = std::vector<int>;

constexpr std::size_t at(std::size_t row, std::size_t col) {
    return row * kSize + col;
}

// Row-major access: matrix[i][j] -> consecutive addresses.
std::uint64_t fill_row_major(Matrix& matrix) {
    int counter = 0;
    for (std::size_t i = 0; i < kSize; ++i) {
        for (std::size_t j = 0; j < kSize; ++j) {
            matrix[at(i, j)] = counter++;
        }
    }
    // Readback makes the writes observable (prevents dead-store elimination).
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < kSize; ++i) {
        for (std::size_t j = 0; j < kSize; ++j) {
            sum += static_cast<std::uint64_t>(matrix[at(i, j)]);
        }
    }
    return sum;
}

// Column-major access: matrix[j][i] -> jumps by kSize elements.
std::uint64_t fill_column_major(Matrix& matrix) {
    int counter = 0;
    for (std::size_t i = 0; i < kSize; ++i) {
        for (std::size_t j = 0; j < kSize; ++j) {
            matrix[at(j, i)] = counter++;
        }
    }
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < kSize; ++i) {
        for (std::size_t j = 0; j < kSize; ++j) {
            sum += static_cast<std::uint64_t>(matrix[at(i, j)]);
        }
    }
    return sum;
}

}  // namespace

int main() {
    std::printf("== cache_thrashing ==\n");
    std::printf("Matrix %zux%zu int (%zu MiB), L1 data cache %zu bytes\n\n",
                kSize, kSize,
                kSize * kSize * sizeof(int) / (1024 * 1024), kL1CacheCapacity);

    Matrix matrix(kSize * kSize, 0);

    const auto r_row = chp::benchmark(1, 3, 1,
        [&](std::uint64_t& acc) { acc += fill_row_major(matrix); });
    const auto r_col = chp::benchmark(1, 3, 1,
        [&](std::uint64_t& acc) { acc += fill_column_major(matrix); });

    chp::print_result("row-major  matrix[i][j]", r_row);
    chp::print_result("column-major matrix[j][i]", r_col);

    const double ratio = r_col.mean_ns / r_row.mean_ns;
    std::printf("column/row time ratio: %.2fx\n", ratio);
    return 0;
}
