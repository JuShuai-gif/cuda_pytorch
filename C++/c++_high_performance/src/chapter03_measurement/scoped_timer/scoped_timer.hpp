#pragma once

#include <chrono>
#include <cstdio>
#include <string>

namespace chp {
namespace measure {

// A scoped timer: measures the wall time between construction and
// destruction. This is the instrumentation-profiler style timer described in
// the book (PDF p.99-100), using the monotonic steady_clock.
class ScopedTimer {
public:
    using ClockType = std::chrono::steady_clock;

    explicit ScopedTimer(const char* func)
        : function_(func), start_(ClockType::now()) {}

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer(ScopedTimer&&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    ScopedTimer& operator=(ScopedTimer&&) = delete;

    ~ScopedTimer() {
        const auto stop = ClockType::now();
        const auto duration = stop - start_;
        const auto ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
        std::printf("%lld ms %s\n", static_cast<long long>(ms), function_);
    }

private:
    const char* function_;
    const ClockType::time_point start_;
};

}  // namespace measure
}  // namespace chp
