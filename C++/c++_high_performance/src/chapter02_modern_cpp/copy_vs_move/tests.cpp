#include <cstdio>
#include <utility>

#include "baseline.hpp"
#include "test_utils.hpp"

namespace {

using chp::cvm::Buffer;

Buffer make_buffer(std::size_t n) {
    Buffer local(n);
    return local;
}

}  // namespace

int main() {
    // Copy happens when the source is a named l-value.
    {
        Buffer::reset_counters();
        Buffer a(8);
        Buffer b = a;
        CHP_CHECK(Buffer::copy_count() >= 1);
        CHP_CHECK(b.size() == a.size());
    }

    // Move happens when the source is wrapped in std::move.
    {
        Buffer::reset_counters();
        Buffer a(8);
        Buffer b = std::move(a);
        CHP_CHECK(Buffer::move_count() >= 1);
        CHP_CHECK(b.size() == 8);
        CHP_CHECK(a.size() == 0);  // moved-from state (our implementation)
        (void)a;
    }

    // Returning a local does not copy (guaranteed elision or move).
    {
        Buffer::reset_counters();
        Buffer d = make_buffer(16);
        CHP_CHECK(d.size() == 16);
        // No copy should have happened for the return; at most a move.
        CHP_CHECK(Buffer::copy_count() == 0);
    }

    // Copy preserves the source value.
    {
        Buffer::reset_counters();
        Buffer a(4);
        Buffer b = a;
        CHP_CHECK(b.sum() == a.sum());
    }

    return chp::test_summary("copy_vs_move");
}
