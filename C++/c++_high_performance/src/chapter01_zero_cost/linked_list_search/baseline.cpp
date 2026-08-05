#include "baseline.hpp"

#include <cstring>

namespace chp {
namespace lls {

std::size_t count_title_c_style(const CNode* head, const char* needle) {
    std::size_t count = 0;
    for (const CNode* node = head; node != nullptr; node = node->next) {
        if (std::strcmp(node->title, needle) == 0) {
            ++count;
        }
    }
    return count;
}

}  // namespace lls
}  // namespace chp
