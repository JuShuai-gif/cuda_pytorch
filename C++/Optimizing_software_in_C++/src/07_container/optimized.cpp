// 07_container: optimized -- reserve() upfront + contiguous storage.
//
// PDF 9.7 (p98): calling std::vector::reserve with the final size avoids the
// repeated reallocations. Contiguous memory is cache friendly.
#include <cstdio>
#include <vector>

int main() {
    const int n = 4'000'000;

    // reserve the final size before inserting (PDF p98).
    std::vector<int> v;
    v.reserve(n);
    for (int i = 0; i < n; ++i) v.push_back(i);

    long long sum = 0;
    for (int x : v) sum += x;

    // Reuse capacity instead of reallocating.
    v.clear();
    for (int i = 0; i < n / 2; ++i) v.push_back(-i);

    std::printf("reserved vector sum=%lld size=%zu\n", sum, v.size());
    return 0;
}
