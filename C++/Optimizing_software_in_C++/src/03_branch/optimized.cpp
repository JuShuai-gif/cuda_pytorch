// 03_branch: optimized -- branchless (predicate arithmetic).
//
// PDF 7.12 (p43-45) and 14.3 (p148): replacing a poorly-predicted branch with
// arithmetic removes the misprediction penalty entirely.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

int branchless_sum(const std::vector<int>& v) {
    int sum = 0;
    for (int x : v) {
        sum += x & ~((x - 128) >> 31);  // x if x>=128 else 0, no branch
    }
    return sum;
}

int main(int argc, char** argv) {
    const size_t n = 8'000'000;
    std::vector<int> v(n);
    bool sorted = (argc > 1 && argv[1][0] == 's');
    for (size_t i = 0; i < n; ++i) v[i] = std::rand() % 256;
    if (sorted) std::sort(v.begin(), v.end());

    volatile int r = branchless_sum(v);
    std::printf("sorted=%d sum=%d\n", sorted ? 1 : 0, r);
    return 0;
}
