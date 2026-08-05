#include "baseline.hpp"

namespace chp {
namespace nomv {

int MoveNoexcept::copies = 0;
int MoveNoexcept::moves = 0;
int MoveThrowing::copies = 0;
int MoveThrowing::moves = 0;

}  // namespace nomv
}  // namespace chp
