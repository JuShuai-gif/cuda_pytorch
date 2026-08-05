// Memory alignment and padding.
//
// The book (PDF p.186-189): every type has an alignment requirement
// (alignof). The compiler inserts padding between members to satisfy it.
// Reordering members largest-first can shrink the struct. This is both a
// memory-size and a cache-locality concern.

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace {

// Version 1 (book PDF p.188): bool first, then double, then int -> 24 bytes.
struct DocumentV1 {
    bool is_cached = false;
    double rank = 0.0;
    int id = 0;
};

// Version 2 (book PDF p.189): largest members first -> 16 bytes.
struct DocumentV2 {
    double rank = 0.0;
    int id = 0;
    bool is_cached = false;
};

// Alignment observation: objects placed at addresses that are multiples of
// their alignment.
struct alignas(64) CacheAligned {
    int value = 0;
};

}  // namespace

int main() {
    std::printf("== alignment_padding ==\n");

    std::printf("alignof(char)    = %zu\n", alignof(char));
    std::printf("alignof(int)     = %zu\n", alignof(int));
    std::printf("alignof(double)  = %zu\n", alignof(double));
    std::printf("alignof(max_align_t) = %zu\n", alignof(std::max_align_t));

    std::printf("sizeof(DocumentV1) = %zu (bool first, 24 on x86-64)\n",
                sizeof(DocumentV1));
    std::printf("sizeof(DocumentV2) = %zu (double first, 16)\n",
                sizeof(DocumentV2));

    // new/malloc always return max-aligned memory (PDF p.187).
    auto* p = new char{};
    const auto addr =
        reinterpret_cast<std::uintptr_t>(static_cast<void*>(p));
    std::printf("new char address %% max_align = %zu (0 means aligned)\n",
                addr % alignof(std::max_align_t));
    delete p;

    // alignas overrides alignment to a larger value.
    std::printf("alignof(CacheAligned) = %zu\n", alignof(CacheAligned));
    std::printf("sizeof(CacheAligned)  = %zu\n", sizeof(CacheAligned));

    return 0;
}
