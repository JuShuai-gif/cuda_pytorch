// Parallel std::transform: naive chunking vs divide-and-conquer (PDF p.319-326).

#include <cstddef>
#include <cstdio>
#include <vector>

#include "parallel.hpp"

namespace {

// Cost depends on the value: larger inputs take much longer (PDF p.307).
float cost_varying(float v) {
    float sum = v;
    const auto i_max = static_cast<std::size_t>(v / 100.0F);
    for (std::size_t i = 0; i < i_max; ++i) {
        sum += static_cast<float>(i * i * i) * sum;
    }
    return sum;
}

}  // namespace

int main() {
    std::printf("== par_transform ==\n");

    constexpr std::size_t n = 20'000;
    std::vector<float> src(n);
    for (std::size_t i = 0; i < n; ++i) {
        src[i] = static_cast<float>(i);
    }

    // Serial reference.
    std::vector<float> ref(n);
    std::transform(src.begin(), src.end(), ref.begin(), cost_varying);

    // Naive: fixed number of chunks.
    {
        std::vector<float> dst(n);
        chp11::par_transform_naive(src.begin(), src.end(), dst.begin(),
                                   cost_varying);
        std::printf("naive  matches serial: %d\n", dst == ref);
    }

    // Divide and conquer with different chunk sizes.
    for (const std::size_t chunk : {5000, 500, 50}) {
        std::vector<float> dst(n);
        chp11::par_transform(src.begin(), src.end(), dst.begin(), cost_varying,
                             chunk);
        std::printf("chunk=%zu matches serial: %d\n", chunk, dst == ref);
    }

    return 0;
}
