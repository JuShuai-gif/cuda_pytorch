#include "CycleTimer.h"

#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>
#include <type_traits>

int main() {
  static_assert(!std::is_default_constructible<CycleTimer>::value,
                "CycleTimer should not be default constructible");
  static_assert(!std::is_copy_constructible<CycleTimer>::value,
                "CycleTimer should not be copy constructible");
  static_assert(!std::is_copy_assignable<CycleTimer>::value,
                "CycleTimer should not be copy assignable");

  double t1 = CycleTimer::currentSeconds();
  double t2 = CycleTimer::currentSeconds();

  assert(t1 >= 0.0);
  assert(t2 >= t1);

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  double t3 = CycleTimer::currentSeconds();

  assert(t3 >= t2);
  assert(t3 - t2 >= 0.08);   // 留一点误差空间
  assert(t3 - t2 < 1.0);     // 防止明显异常

  std::cout << "t1 = " << t1 << " seconds\n";
  std::cout << "t2 = " << t2 << " seconds\n";
  std::cout << "t3 = " << t3 << " seconds\n";
  std::cout << "All CycleTimer tests passed.\n";

  return 0;
}