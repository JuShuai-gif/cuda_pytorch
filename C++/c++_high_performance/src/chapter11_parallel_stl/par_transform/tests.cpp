// Correctness checks: parallel transform/count_if match the serial versions.

#include <cstddef>
#include <cstdio>
#include <vector>

#include "test_utils.hpp"

#include "parallel.hpp"

namespace {

float square(float v) { return v * v; }

bool is_odd(int v) { return (v % 2) != 0; }

}  // namespace

int main() {
    constexpr std::size_t n = 100'000;
    std::vector<float> src(n);
    for (std::size_t i = 0; i < n; ++i) {
        src[i] = static_cast<float>(i);
    }

    // par_transform (divide and conquer) vs serial.
    {
        std::vector<float> serial(n), dac(n);
        std::transform(src.begin(), src.end(), serial.begin(), square);
        chp11::par_transform(src.begin(), src.end(), dac.begin(), square, 100);
        CHP_CHECK(serial == dac);
    }

    // par_transform_naive vs serial.
    {
        std::vector<float> serial(n), naive(n);
        std::transform(src.begin(), src.end(), serial.begin(), square);
        chp11::par_transform_naive(src.begin(), src.end(), naive.begin(),
                                   square);
        CHP_CHECK(serial == naive);
    }

    // par_count_if vs serial.
    {
        std::vector<int> vals(n);
        for (std::size_t i = 0; i < n; ++i) {
            vals[i] = static_cast<int>(i);
        }
        const auto serial_count = std::count_if(vals.begin(), vals.end(), is_odd);
        const auto par_count =
            chp11::par_count_if(vals.begin(), vals.end(), is_odd, 100);
        CHP_CHECK(par_count == static_cast<std::size_t>(serial_count));
        CHP_CHECK(par_count == n / 2);
    }

    // Empty and tiny ranges must be handled.
    {
        std::vector<float> empty;
        chp11::par_transform(empty.begin(), empty.end(), empty.begin(), square, 1);
        chp11::par_transform_naive(empty.begin(), empty.end(), empty.begin(),
                                   square);
        const auto cnt = chp11::par_count_if(empty.begin(), empty.end(),
                                             is_odd, 1);
        CHP_CHECK(cnt == 0);
    }

    return chp::test_summary("par_transform");
}
