// 10_multithreading: scaling test -- serial vs mutex vs local-reduction.
//
// PDF 10 (p111-112). Shows: (a) how speed scales with thread count for a
// memory-bound reduction, (b) why shared state / a mutex serializes work.
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

#include "common/benchmark.h"

double serial_sum(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return s;
}

// Every thread adds into one shared variable guarded by a mutex: correct but
// serialized, demonstrating why shared state is a bottleneck (PDF p112).
double atomic_sum(const std::vector<double>& v, int nthreads) {
    std::mutex m;
    double total = 0.0;
    size_t chunk = v.size() / (size_t)nthreads;
    std::vector<std::thread> ts;
    for (int t = 0; t < nthreads; ++t) {
        size_t b = (size_t)t * chunk;
        size_t e = (t == nthreads - 1) ? v.size() : (size_t)(t + 1) * chunk;
        ts.emplace_back([&, b, e] {
            double s = 0.0;
            for (size_t i = b; i < e; ++i) s += v[i];
            std::lock_guard<std::mutex> lk(m);
            total += s;
        });
    }
    for (auto& t : ts) t.join();
    return total;
}

// Local reduction, merged once.
double local_sum(const std::vector<double>& v, int nthreads) {
    std::vector<double> partial((size_t)nthreads, 0.0);
    size_t chunk = v.size() / (size_t)nthreads;
    std::vector<std::thread> ts;
    for (int t = 0; t < nthreads; ++t) {
        size_t b = (size_t)t * chunk;
        size_t e = (t == nthreads - 1) ? v.size() : (size_t)(t + 1) * chunk;
        ts.emplace_back([&, t, b, e] {
            double s = 0.0;
            for (size_t i = b; i < e; ++i) s += v[i];
            partial[(size_t)t] = s;
        });
    }
    for (auto& t : ts) t.join();
    double s = 0.0;
    for (double p : partial) s += p;
    return s;
}

int main() {
    std::vector<double> v(64'000'000, 1.0);

    bench("serial", [&] { return serial_sum(v); });
    bench("local x1", [&] { return local_sum(v, 1); });
    bench("local x2", [&] { return local_sum(v, 2); });
    bench("local x4", [&] { return local_sum(v, 4); });
    bench("local x8", [&] { return local_sum(v, 8); });
    bench("local x16", [&] { return local_sum(v, 16); });
    bench("atomic x4", [&] { return atomic_sum(v, 4); });

    std::printf("\nresults: %.0f %.0f %.0f %.0f %.0f %.0f\n",
                serial_sum(v), local_sum(v, 2), local_sum(v, 4),
                local_sum(v, 8), local_sum(v, 16), atomic_sum(v, 4));
    return 0;
}
