#pragma once

#include <cuda_runtime.h>

#include <cute/config.hpp>
#include <cute/numeric/integral_constant.hpp>

namespace cutlass::fmha {

struct Pow2 {
  int n;
  int log2_n;

  explicit CUTE_DEVICE Pow2(int n) : n(n) {
#ifdef __CUDA_ARCH__
    log2_n = __ffs(n) - 1;
#endif
  }

  template <class T>
  CUTE_HOST_DEVICE T operator*(T const& b) const {
    return n * b;
  }

  template <int N>
  CUTE_HOST_DEVICE auto operator*(Int<N> const&) const {
    if constexpr (N & (N - 1) == 0) {
      return Pow2{n * N};
    }
    return n * N;
  }
};

template <class T>
CUTE_HOST_DEVICE auto operator/(T const& a, Pow2 const& b) {
  return a >> b.log2_n;
}

template <class T>
CUTE_HOST_DEVICE auto operator%(T const& a, Pow2 const& b) {
  return a & (b.n - 1);
}

template <class T>
CUTE_HOST_DEVICE bool operator<(T const& a, Pow2 const& b) {
  return a < b.n;
}

CUTE_HOST_DEVICE void print(Pow2 const& a) { printf("2^%d", a.log2_n); }

}  // end namespace cutlass::fmha

namespace cute {

template <>
struct is_integral<cutlass::fmha::Pow2> : true_type {};

}  // end namespace cute
