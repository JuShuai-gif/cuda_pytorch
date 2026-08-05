#pragma once

#include <cstddef>

namespace chp {
namespace lls {

// A hand-rolled C-style singly linked list node.
struct CNode {
    const char* title;
    const CNode* next;
};

// Counts occurrences of `needle` by walking a C-style linked list with a
// manual for-loop and strcmp(). This mirrors the C version in the book.
std::size_t count_title_c_style(const CNode* head, const char* needle);

}  // namespace lls
}  // namespace chp
