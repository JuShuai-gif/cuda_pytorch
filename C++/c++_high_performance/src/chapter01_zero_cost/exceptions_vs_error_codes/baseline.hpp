#pragma once

namespace chp {
namespace evc {

enum class DivideError { none, division_by_zero };

// Error-code style: returns the division result through `out` and reports
// success/failure through the return value. The caller must check it.
DivideError divide_checked(int a, int b, int& out);

}  // namespace evc
}  // namespace chp
