// 03_branch: compare sorted/random data with branch vs branchless.
//
// PDF p43-45: sorted data => branch almost always takes the same path =>
// well predicted. Random data => 50/50 => mispredicted half the time.
// Run under `perf stat -e branch-misses` to see the difference.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common/benchmark.h"

int branch_sum(const std::vector<int>& v) {
    int sum = 0;
    for (int x : v) if (x >= 128) sum += x;
    return sum;
}

int branchless_sum(const std::vector<int>& v) {
    int sum = 0;
    for (int x : v) sum += x & ~((x - 128) >> 31);
    return sum;
}

int main() {
    const size_t n = 8'000'000;
    std::vector<int> rand_v(n), sorted_v(n);
    for (size_t i = 0; i < n; ++i) rand_v[i] = std::rand() % 256;
    sorted_v = rand_v;
    std::sort(sorted_v.begin(), sorted_v.end());

    std::printf("== random data (poor prediction) ==\n");
    bench("branch",       [&] { return branch_sum(rand_v); });
    bench("branchless",   [&] { return branchless_sum(rand_v); });

    std::printf("\n== sorted data (good prediction) ==\n");
    bench("branch",       [&] { return branch_sum(sorted_v); });
    bench("branchless",   [&] { return branchless_sum(sorted_v); });

    // Correctness: both must produce the same result.
    std::printf("\nrandom : branch=%d branchless=%d\n",
                branch_sum(rand_v), branchless_sum(rand_v));
    std::printf("sorted : branch=%d branchless=%d\n",
                branch_sum(sorted_v), branchless_sum(sorted_v));
    return 0;
}
