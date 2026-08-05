#pragma once

#include <atomic>

namespace chp {

// Prevents the compiler from moving or eliminating memory operations across
// this point. Used in benchmarks to keep the measured work alive so the
// compiler cannot hoist the call out of the loop or eliminate it entirely.
//
// `std::atomic_signal_fence` is a compiler barrier (not a CPU fence): it
// blocks compiler reordering but costs zero instructions at runtime.
inline void compiler_barrier() noexcept {
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

}  // namespace chp
