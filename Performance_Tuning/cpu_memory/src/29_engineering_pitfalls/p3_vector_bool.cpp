// Pitfall P3: std::vector<bool> is NOT a normal vector.
//
// std::vector<bool> is a specialized bit-packed container. Element access
// goes through proxy objects, is much slower, and no element is a real bool&.
// This surprises engineers: a "vector of bool" benchmark is not comparable
// to vector<char>.
//
// Related PDF: 6.2.1 (data layout / cache line efficiency), 7.3 (allocation).

#include <cstdint>
#include <cstdio>
#include <vector>

#include "benchmark.h"

static constexpr size_t N = 1 << 26;
static constexpr int kRounds = 5;

int main() {
    std::printf("Pitfall P3: std::vector<bool> vs std::vector<char>\n");

    std::vector<bool> vb(N, false);
    std::vector<char> vc(N, 0);

    // Ensure all elements are touched (avoid page-fault-dominated first run).
    for (size_t i = 0; i < N; ++i) { vb[i] = (i & 1) != 0; vc[i] = char(i & 1); }

    auto run_bool = [&] {
        uint64_t s = 0;
        for (size_t i = 0; i < N; ++i) s += (uint64_t)vb[i];
        bm::do_not_optimize(s);
    };
    auto run_char = [&] {
        uint64_t s = 0;
        for (size_t i = 0; i < N; ++i) s += (uint64_t)vc[i];
        bm::do_not_optimize(s);
    };

    run_bool();
    run_char();

    auto r_bool = bm::time_rounds(kRounds, run_bool);
    auto r_char = bm::time_rounds(kRounds, run_char);

    std::printf("vector<bool> write+read: mean=%.2f ms\n", r_bool.mean_ms);
    std::printf("vector<char> write+read: mean=%.2f ms\n", r_char.mean_ms);
    std::printf("vector<bool> is %.1fx slower here (bit-packed proxy access).\n",
                r_bool.mean_ms / r_char.mean_ms);
    std::printf("Also: you cannot take a bool& into a vector<bool> element.\n");
    return 0;
}
