// Returning data and errors through std::promise/std::future (PDF p.298-299).
//
// The worker runs on a separate thread and reports its result -- or an
// exception -- through a promise. The caller blocks on future.get() and
// catches errors as normal exceptions, without shared globals or locks.

#include <cstdio>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>
#include <thread>

namespace {

void divide(int a, int b, std::promise<int>& p) {
    if (b == 0) {
        p.set_exception(std::make_exception_ptr(
            std::runtime_error{"Divide by zero exception"}));
    } else {
        p.set_value(a / b);
    }
}

}  // namespace

int main() {
    std::printf("== promise_future ==\n");

    // Success path.
    {
        std::promise<int> p;
        std::thread worker{divide, 45, 5, std::ref(p)};
        worker.detach();
        std::future<int> f = p.get_future();  // get() is non-const
        std::printf("45 / 5 = %d\n", f.get());  // blocks until ready
    }

    // Error path: the exception is transported to the calling thread.
    {
        std::promise<int> p;
        std::thread worker{divide, 45, 0, std::ref(p)};
        worker.detach();
        std::future<int> f = p.get_future();
        try {
            (void)f.get();
        } catch (const std::exception& e) {
            std::printf("caught exception: %s\n", e.what());
        }
    }

    return 0;
}
