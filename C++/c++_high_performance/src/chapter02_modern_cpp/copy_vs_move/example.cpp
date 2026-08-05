// copy_vs_move: demonstrate when the compiler moves instead of copies.
//
// The book (PDF p.71-75) explains that the compiler moves an object when it
// is an r-value: an unnamed temporary coming out of a function, or a named
// variable wrapped in std::move. Copying happens when the source is a named,
// non-const-qualified l-value.

#include "baseline.hpp"

#include <cstdio>
#include <utility>

namespace {

using chp::cvm::Buffer;

// Returns a buffer: with move semantics the local buffer is moved into the
// caller instead of being copied (book PDF p.70).
Buffer make_buffer(std::size_t n) {
    Buffer local(n);
    return local;  // guaranteed move/copy elision or move
}

}  // namespace

int main() {
    Buffer::reset_counters();

    // 1) Copy construction from a named l-value.
    Buffer a(16);
    const int copies_after_a = Buffer::copy_count();
    std::printf("construct a: copies=%d moves=%d\n", copies_after_a,
                Buffer::move_count());

    // 2) Copy construction: named variable b on the right.
    Buffer b = a;  // a is a named l-value -> copy
    std::printf("Buffer b = a;        copies=%d moves=%d\n",
                Buffer::copy_count(), Buffer::move_count());

    // 3) Move construction via std::move.
    Buffer c = std::move(b);  // std::move makes b an r-value -> move
    std::printf("Buffer c = move(b);  copies=%d moves=%d\n",
                Buffer::copy_count(), Buffer::move_count());

    // 4) Return from a function: no copy (elision/move).
    Buffer d = make_buffer(8);
    std::printf("Buffer d = make();   copies=%d moves=%d\n",
                Buffer::copy_count(), Buffer::move_count());

    // 5) Copy assignment from a named l-value.
    Buffer e(4);
    e = a;  // a is a named l-value -> copy-assignment
    std::printf("e = a;               copies=%d moves=%d\n",
                Buffer::copy_count(), Buffer::move_count());

    // 6) Move assignment via std::move.
    e = std::move(d);  // d is an r-value -> move-assignment
    std::printf("e = move(d);         copies=%d moves=%d\n",
                Buffer::copy_count(), Buffer::move_count());

    // Sanity: moved-from objects are empty (valid but unspecified state).
    std::printf("after move: b.size=%zu c.size=%zu d.size=%zu\n", b.size(),
                c.size(), d.size());
    return 0;
}
