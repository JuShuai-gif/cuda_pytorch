#include <cstdio>
#include <stdexcept>
#include <type_traits>

#include "baseline.hpp"
#include "optimized.hpp"
#include "test_utils.hpp"

namespace {

void reset_counters() {
    chp::raii::Resource::constructed = 0;
    chp::raii::Resource::destroyed = 0;
}

}  // namespace

int main() {
    // --- Manual, success path: caller releases the resource ---
    reset_counters();
    chp::raii::Resource* out = nullptr;
    const int r1 = chp::raii::use_manual(out, 3);
    CHP_CHECK(r1 == 6);
    delete out;
    CHP_CHECK(chp::raii::Resource::constructed == 1);
    CHP_CHECK(chp::raii::Resource::destroyed == 1);

    // --- Manual, throwing path: the resource LEAKS ---
    reset_counters();
    chp::raii::Resource* leaked = nullptr;
    bool threw = false;
    try {
        (void)chp::raii::use_manual(leaked, 0);
    } catch (const std::exception&) {
        threw = true;
    }
    CHP_CHECK(threw);
    // Resource was constructed but never released on this path.
    CHP_CHECK(chp::raii::Resource::constructed == 1);
    CHP_CHECK(chp::raii::Resource::destroyed == 0);
    delete leaked;  // Manual cleanup so this test stays leak-free.

    // --- RAII, throwing path: the guard releases the resource ---
    reset_counters();
    bool threw_raii = false;
    {
        chp::raii::ResourceGuard guard;
        try {
            (void)chp::raii::use_raii(guard, 0);
        } catch (const std::exception&) {
            threw_raii = true;
        }
    }  // guard destroyed here
    CHP_CHECK(threw_raii);
    CHP_CHECK(chp::raii::Resource::constructed == 1);
    CHP_CHECK(chp::raii::Resource::destroyed == 1);

    // --- RAII, success path ---
    reset_counters();
    {
        chp::raii::ResourceGuard guard;
        CHP_CHECK(chp::raii::use_raii(guard, 4) == 8);
    }
    CHP_CHECK(chp::raii::Resource::constructed == 1);
    CHP_CHECK(chp::raii::Resource::destroyed == 1);

    // --- Copying the guard must not be possible ---
    static_assert(!std::is_copy_constructible<chp::raii::ResourceGuard>::value,
                  "ResourceGuard must be non-copyable");

    return chp::test_summary("raii_resource");
}
