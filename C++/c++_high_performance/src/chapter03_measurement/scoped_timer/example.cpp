// ScopedTimer: a primitive instrumentation profiler.
//
// The book (PDF p.99-100) shows a ScopedTimer class that logs how long a
// scope lives. Insert `ScopedTimer t{__func__}` at the top of a function to
// measure it. This is manual instrumentation: it records each entry/exit, but
// the book warns that the added code itself changes the profile.

#include "scoped_timer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

namespace {

void work_a() {
    chp::measure::ScopedTimer timer{"work_a"};
    std::vector<int> v(100'000);
    for (int& x : v) {
        x = 1;
    }
}

void work_b() {
    chp::measure::ScopedTimer timer{"work_b"};
    std::vector<int> v(1'000'000);
    for (int& x : v) {
        x = 2;
    }
}

void work_c() {
    chp::measure::ScopedTimer timer{"work_c"};
    std::vector<int> v(10'000);
    std::sort(v.begin(), v.end());
}

}  // namespace

int main() {
    std::printf("== scoped_timer ==\n");
    work_a();
    work_b();
    work_c();
    return 0;
}
