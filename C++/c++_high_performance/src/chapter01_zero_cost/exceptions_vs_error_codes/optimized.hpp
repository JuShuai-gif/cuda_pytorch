#pragma once

namespace chp {
namespace evc {

// Exception style: throws std::runtime_error on division by zero and returns
// the result directly otherwise. The book argues that with modern compilers
// exceptions cost nothing on the non-throwing (success) path.
int divide_throwing(int a, int b);

}  // namespace evc
}  // namespace chp
