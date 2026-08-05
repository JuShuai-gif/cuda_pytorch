// 03_branch: baseline -- data-dependent branch.
//
// PDF 7.12 (p43-45): a branch that goes 50/50 randomly is mispredicted 50%
// of the time; each misprediction costs ~15-25 cycles.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

int branch_sum(const std::vector<int>& v) {
    int sum = 0;
    for (int x : v) {
        if (x >= 128) sum += x;   // data-dependent branch
    }
    return sum;
}

int main(int argc, char** argv) {
    const size_t n = 8'000'000;
    std::vector<int> v(n);
    bool sorted = (argc > 1 && argv[1][0] == 's');
    for (size_t i = 0; i < n; ++i) v[i] = std::rand() % 256;
    if (sorted) std::sort(v.begin(), v.end());

    volatile int r = branch_sum(v);
    std::printf("sorted=%d sum=%d\n", sorted ? 1 : 0, r);
    return 0;
}
