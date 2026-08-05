// 10_multithreading: optimized -- data decomposition, per-thread partial
// sums, single merge. Each thread accumulates into its OWN slot.
//
// PDF 10 (p111-112): data decomposition + per-thread local results +
// merge at the end keeps synchronization minimal.
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

double parallel_sum(const std::vector<double>& v, int nthreads) {
    size_t chunk = v.size() / (size_t)nthreads;
    std::vector<double> partial((size_t)nthreads, 0.0);

    std::vector<std::thread> threads;
    threads.reserve((size_t)nthreads);
    for (int t = 0; t < nthreads; ++t) {
        size_t b = (size_t)t * chunk;
        size_t e = (t == nthreads - 1) ? v.size() : (size_t)(t + 1) * chunk;
        threads.emplace_back([&, t, b, e] {
            double s = 0.0;
            for (size_t i = b; i < e; ++i) s += v[i];
            partial[(size_t)t] = s;   // own slot: no false sharing within vector? see note
        });
    }
    for (auto& th : threads) th.join();

    double sum = 0.0;
    for (double s : partial) sum += s;
    return sum;
}

int main(int argc, char** argv) {
    int nthreads = 4;
    if (argc > 1) nthreads = std::atoi(argv[1]);
    if (nthreads < 1) nthreads = 1;

    std::vector<double> v(64'000'000, 1.0);
    volatile double r = parallel_sum(v, nthreads);
    std::printf("threads=%d checksum=%.0f\n", nthreads, r);
    return 0;
}
