#include <array>
#include <cstddef>
#include <cstdio>

#include "test_utils.hpp"

namespace {

constexpr std::size_t kSize = 8;
using MatrixType = std::array<std::array<int, kSize>, kSize>;

}  // namespace

int main() {
    // Row-major fill: each cell gets a unique, increasing value 0..63.
    MatrixType a{};
    int counter = 0;
    for (std::size_t i = 0; i < kSize; ++i) {
        for (std::size_t j = 0; j < kSize; ++j) {
            a[i][j] = counter++;
        }
    }
    for (std::size_t i = 0; i < kSize; ++i) {
        for (std::size_t j = 0; j < kSize; ++j) {
            CHP_CHECK(a[i][j] == static_cast<int>(i * kSize + j));
        }
    }

    // Column-major fill (the same nested loops, indices swapped) produces
    // the transpose, but every cell must still hold a unique value 0..63.
    MatrixType b{};
    counter = 0;
    for (std::size_t i = 0; i < kSize; ++i) {
        for (std::size_t j = 0; j < kSize; ++j) {
            b[j][i] = counter++;
        }
    }
    for (std::size_t i = 0; i < kSize; ++i) {
        for (std::size_t j = 0; j < kSize; ++j) {
            CHP_CHECK(b[i][j] == static_cast<int>(j * kSize + i));
        }
    }

    // All cells in both matrices are within [0, 64).
    for (std::size_t i = 0; i < kSize; ++i) {
        for (std::size_t j = 0; j < kSize; ++j) {
            CHP_CHECK(a[i][j] >= 0 && a[i][j] < 64);
            CHP_CHECK(b[i][j] >= 0 && b[i][j] < 64);
        }
    }

    return chp::test_summary("cache_thrashing");
}
