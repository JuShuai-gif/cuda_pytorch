// Task-based concurrency: packaged_task and std::async (PDF p.299-300).
//
// std::async is the recommended way to run a function asynchronously: it
// wires up the promise/future machinery and decides whether to spawn a
// thread. The function keeps a plain signature.

#include <cstdio>
#include <exception>
#include <future>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

int divide(int a, int b) {
    if (b == 0) {
        throw std::runtime_error{"Divide by zero exception"};
    }
    return a / b;
}

long sum_range(int lo, int hi) {
    long s = 0;
    for (int i = lo; i < hi; ++i) {
        s += i;
    }
    return s;
}

}  // namespace

int main() {
    std::printf("== async_tasks ==\n");

    // std::async: minimal code, result arrives through the future.
    {
        auto f = std::async(divide, 45, 5);
        std::printf("45 / 5 = %d\n", f.get());
    }
    {
        auto f = std::async(divide, 45, 0);
        try {
            (void)f.get();
        } catch (const std::exception& e) {
            std::printf("caught exception: %s\n", e.what());
        }
    }

    // std::packaged_task: explicit control over where the task runs.
    {
        std::packaged_task<int(int, int)> task{divide};
        auto f = task.get_future();
        std::thread{std::move(task), 10, 2}.detach();
        std::printf("10 / 2 = %d\n", f.get());
    }

    // Two async tasks run in parallel and the results combine.
    {
        auto fa = std::async(sum_range, 0, 5'000'000);
        auto fb = std::async(sum_range, 5'000'000, 10'000'000);
        const long total = fa.get() + fb.get();
        std::printf("sum 0..9999999 = %ld\n", total);
    }

    return 0;
}
