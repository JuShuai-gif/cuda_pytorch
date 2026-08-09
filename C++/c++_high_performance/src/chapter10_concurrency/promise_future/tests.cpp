// Correctness checks for promise/future value and exception transport.

#include <cstdio>
#include <exception>
#include <future>
#include <stdexcept>
#include <thread>

#include "test_utils.hpp"

namespace {

int add_and_multiply(int a, int b, int c) {
    return (a + b) * c;
}

}  // namespace

int main() {
    // Value transport.
    {
        std::packaged_task<int(int, int, int)> task{add_and_multiply};
        std::future<int> f = task.get_future();
        std::thread{std::move(task), 2, 3, 4}.detach();
        CHP_CHECK(f.get() == 20);
    }

    // Exception transport.
    {
        std::promise<int> p;
        std::future<int> f = p.get_future();
        std::thread{[&p] {
            p.set_exception(std::make_exception_ptr(
                std::runtime_error{"boom"}));
        }}.detach();
        bool caught = false;
        try {
            (void)f.get();
        } catch (const std::runtime_error&) {
            caught = true;
        }
        CHP_CHECK(caught);
    }

    // get() consumes the shared state: calling it twice on a non-shared
    // future throws std::future_error.
    {
        std::packaged_task<int()> task{[]{ return 42; }};
        std::future<int> f = task.get_future();
        std::thread{std::move(task)}.detach();
        CHP_CHECK(f.get() == 42);
        bool threw = false;
        try {
            (void)f.get();
        } catch (const std::future_error&) {
            threw = true;
        }
        CHP_CHECK(threw);
    }

    return chp::test_summary("promise_future");
}
