#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>

namespace chp {
namespace arena {

// A bump allocator over a fixed-size, stack-allocated buffer.
//
// Based on the book's Arena (PDF p.200-203, itself derived from Howard
// Hinnant's short_alloc). Key properties:
//  - alignment: hands out memory aligned to alignof(std::max_align_t)
//  - capacity: fixed at compile time; allocations that do not fit fall back
//    to ::operator new
//  - deallocate: only reclaims the most recent (top) block, like a stack;
//    other frees are ignored (caller must ensure correct ordering or accept
//    the fallback)
//  - non-copyable, non-movable (owns a buffer + a bump pointer)
template <std::size_t N>
class Arena {
public:
    static constexpr std::size_t kAlignment = alignof(std::max_align_t);

    Arena() noexcept : ptr_(buffer_) {}
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    // Reset the bump pointer (all previously handed-out memory is reusable).
    void reset() noexcept { ptr_ = buffer_; }

    static constexpr std::size_t size() noexcept { return N; }

    // Bytes currently handed out (0..N).
    std::size_t used() const noexcept {
        return static_cast<std::size_t>(ptr_ - buffer_);
    }

    // Hand out n bytes aligned to kAlignment. If the buffer cannot satisfy
    // the request, fall back to ::operator new (the book's behaviour).
    char* allocate(std::size_t n) {
        const std::size_t aligned_n = align_up(n);
        const std::size_t available = static_cast<std::size_t>(buffer_ + N - ptr_);
        if (available >= aligned_n) {
            char* r = ptr_;
            ptr_ += aligned_n;
            return r;
        }
        return static_cast<char*>(::operator new(n));
    }

    // Reclaim memory if p is the most recent block handed out from the
    // buffer; otherwise forward to ::operator delete.
    void deallocate(char* p, std::size_t n) noexcept {
        if (pointer_in_buffer(p)) {
            const std::size_t aligned_n = align_up(n);
            if (p + aligned_n == ptr_) {
                ptr_ = p;
            }
            // Non-top deallocations are ignored (book p.203).
        } else {
            ::operator delete(p);
        }
    }

private:
    static std::size_t align_up(std::size_t n) noexcept {
        return (n + (kAlignment - 1)) & ~(kAlignment - 1);
    }

    bool pointer_in_buffer(const char* p) const noexcept {
        return buffer_ <= p && p <= buffer_ + N;
    }

    alignas(kAlignment) char buffer_[N];
    char* ptr_;
};

// A stateful STL allocator that draws memory from an Arena<N>.
//
// Based on the book's ShortAlloc (PDF p.207-208). Two instances compare equal
// only if they reference the same arena. Objects are constructed/destroyed by
// the container; this allocator only supplies raw memory.
template <typename T, std::size_t N>
class ShortAlloc {
public:
    using value_type = T;
    using arena_type = Arena<N>;

    ShortAlloc() = delete;
    explicit ShortAlloc(arena_type& arena) noexcept : arena_(&arena) {}

    template <typename U>
    ShortAlloc(const ShortAlloc<U, N>& other) noexcept : arena_(other.arena_) {}

    // Rebind support for containers that allocate node types.
    template <typename U>
    struct rebind {
        using other = ShortAlloc<U, N>;
    };

    T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        return reinterpret_cast<T*>(arena_->allocate(n * sizeof(T)));
    }

    void deallocate(T* p, std::size_t n) noexcept {
        arena_->deallocate(reinterpret_cast<char*>(p), n * sizeof(T));
    }

    template <typename U, std::size_t M>
    bool operator==(const ShortAlloc<U, M>& other) const noexcept {
        if (N != M) {
            return false;
        }
        return static_cast<const void*>(arena_) ==
               static_cast<const void*>(other.arena_);
    }

    template <typename U, std::size_t M>
    bool operator!=(const ShortAlloc<U, M>& other) const noexcept {
        return !(*this == other);
    }

    template <typename U, std::size_t M>
    friend class ShortAlloc;

private:
    arena_type* arena_;
};

}  // namespace arena
}  // namespace chp
