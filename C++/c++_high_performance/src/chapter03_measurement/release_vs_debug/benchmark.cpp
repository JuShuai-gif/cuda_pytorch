// Release vs Debug performance.
//
// Compile the SAME source twice: once with -O0 -g (Debug), once with -O3
// (Release), and compare. Debug builds do not just lack optimizations - they
// also keep every temporary and skip inlining, which can change results by
// orders of magnitude.
//
// Build both via CMake:
//   cmake -S src -B build       -DCMAKE_BUILD_TYPE=Release
//   cmake -S src -B build-debug -DCMAKE_BUILD_TYPE=Debug
//   cmake --build build -j && cmake --build build-debug -j
//   ./build/chapter03_measurement/ch03_release_debug_benchmark
//   ./build-debug/chapter03_measurement/ch03_release_debug_benchmark

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

#ifdef NDEBUG
const char* kBuildType = "Release (NDEBUG defined)";
#else
const char* kBuildType = "Debug (NDEBUG not defined)";
#endif

// A loop that benefits heavily from optimization: std::count_if + sort.
std::uint64_t workload(std::vector<int>& v) {
    std::sort(v.begin(), v.end());
    std::uint64_t count = 0;
    for (int x : v) {
        if (x > 0) {
            count += static_cast<std::uint64_t>(x);
        }
    }
    return count;
}

}  // namespace

int main() {
    std::printf("== release_vs_debug ==\n");
    std::printf("Build type: %s\n\n", kBuildType);

    std::vector<int> data(200'000);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<int>(i) % 1000;
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::uint64_t sum = 0;
    for (int r = 0; r < 5; ++r) {
        sum += workload(data);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("5x workload (sort 200k + count): %.2f ms\n", ms);
    std::printf("checksum: %llu\n", static_cast<unsigned long long>(sum));
    return 0;
}
