#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "test_utils.hpp"

struct DocumentV1 {
    bool is_cached = false;
    double rank = 0.0;
    int id = 0;
};

struct DocumentV2 {
    double rank = 0.0;
    int id = 0;
    bool is_cached = false;
};

int main() {
    // Member order affects sizeof through padding (PDF p.188-189).
    CHP_CHECK(sizeof(DocumentV2) <= sizeof(DocumentV1));

    // Alignment is a power of two and divides the size.
    CHP_CHECK((sizeof(DocumentV1) % alignof(DocumentV1)) == 0);
    CHP_CHECK((sizeof(DocumentV2) % alignof(DocumentV2)) == 0);

    // Data members must be correctly aligned within the struct.
    CHP_CHECK(offsetof(DocumentV2, rank) % alignof(double) == 0);
    CHP_CHECK(offsetof(DocumentV2, id) % alignof(int) == 0);

    // Object addresses must satisfy the type alignment.
    DocumentV2 d;
    CHP_CHECK(reinterpret_cast<std::uintptr_t>(&d) % alignof(DocumentV2) == 0);

    return chp::test_summary("alignment_padding");
}
