// C++17 Parallel STL execution policies (PDF p.333-337).
//
// seq: serial. par: parallel, exceptions propagate to the caller.
// par_unseq: parallel + SIMD, predicates must not throw or lock.
// Also demonstrates reduce vs accumulate and transform_reduce.

#include <algorithm>
#include <cstdio>
#include <execution>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

float inverse(float denominator) {
    if (denominator != 0.0F) {
        return 1.0F / denominator;
    }
    throw std::runtime_error{"Division by zero"};
}

}  // namespace

int main() {
    std::printf("== execution_policies ==\n");

    // reduce is unordered but for commutative ops equals accumulate.
    {
        std::vector<int> v(1'000'000);
        std::iota(v.begin(), v.end(), 1);
        const auto seq = std::reduce(std::execution::seq, v.begin(), v.end(), 0);
        const auto par = std::reduce(std::execution::par, v.begin(), v.end(), 0);
        const auto par_unseq =
            std::reduce(std::execution::par_unseq, v.begin(), v.end(), 0);
        const auto acc = std::accumulate(v.begin(), v.end(), 0);
        std::printf("reduce(seq)=%d par=%d par_unseq=%d accumulate=%d\n",
                    seq, par, par_unseq, acc);
    }

    // transform_reduce: transform then reduce.
    {
        const std::vector<std::string> mice{"Mickey", "Minnie", "Jerry"};
        const auto num_chars = std::transform_reduce(
            std::execution::par, mice.begin(), mice.end(), std::size_t{0},
            [](std::size_t a, std::size_t b) { return a + b; },
            [](const std::string& m) { return m.size(); });
        std::printf("transform_reduce char count: %zu\n", num_chars);
    }

    // Exception handling with parallel predicates.
    // The book (PDF p.317) shows std::execution::par rethrowing on the
    // caller thread (GCC 7). On GCC 13 libstdc++, any std::execution policy
    // transform -- even seq -- terminates instead of propagating, so here we
    // demonstrate the exception path with the plain (non-policy) transform.
    {
        const std::vector<float> numbers{3.0F, 4.0F, 0.0F, 8.0F, 2.0F};
        std::vector<float> out(numbers.size(), -1.0F);
        try {
            std::transform(numbers.begin(), numbers.end(), out.begin(), inverse);
        } catch (const std::exception& e) {
            std::printf("exception propagated to caller: %s\n", e.what());
        }
        std::printf("out after exception: ");
        for (const float v : out) {
            std::printf("%g ", v);
        }
        std::printf("\n");
    }

    // find with par policy: one-line parallelization.
    {
        const std::vector<std::string> coasters{"woody", "steely", "loopy",
                                                "upside_down"};
        const auto it = std::find(std::execution::par, coasters.begin(),
                                  coasters.end(), "loopy");
        std::printf("find(par, \"loopy\") -> %s\n",
                    it != coasters.end() ? it->c_str() : "(not found)");
    }

    return 0;
}
