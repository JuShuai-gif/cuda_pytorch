// Fixed-size object pool (synthesized from Ch1 RAII + Ch7 custom allocation).
//
// A pool pre-allocates N blocks of block_size bytes, aligned to
// max_align_t, and hands them out via a free list. Allocation is O(1) and
// never calls the system allocator after construction -- ideal when many
// same-size objects are created/destroyed in a loop.

#ifndef CHP_OBJECT_POOL_HPP
#define CHP_OBJECT_POOL_HPP

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace chp {

class ObjectPool {
public:
    ObjectPool(std::size_t block_size, std::size_t capacity)
        : block_size_(normalize_block_size(block_size)),
          capacity_(capacity),
          storage_(std::make_unique<std::byte[]>(
              normalize_block_size(block_size) * capacity +
              alignof(std::max_align_t))) {
        // Align the raw storage so every handed-out pointer is aligned.
        base_ = align_up(storage_.get(), alignof(std::max_align_t));
        // Build the free list: each block points to the next one.
        free_head_ = block_at(0);
        for (std::size_t i = 0; i < capacity_; ++i) {
            next_of(block_at(i)) = block_at(i + 1);
        }
        next_of(block_at(capacity_ - 1)) = nullptr;
    }

    ObjectPool(const ObjectPool&) = delete;
    ObjectPool& operator=(const ObjectPool&) = delete;

    // Allocate one block, or nullptr when the pool is exhausted.
    void* allocate() {
        if (free_head_ == nullptr) {
            return nullptr;
        }
        void* block = free_head_;
        free_head_ = next_of(free_head_);
        ++in_use_;
        return block;
    }

    // Return a block previously obtained from allocate().
    void deallocate(void* ptr) {
        assert(contains(ptr));
        std::byte* p = static_cast<std::byte*>(ptr);
        next_of(p) = free_head_;
        free_head_ = p;
        --in_use_;
    }

    std::size_t capacity() const { return capacity_; }
    std::size_t in_use() const { return in_use_; }
    std::size_t free_count() const { return capacity_ - in_use_; }

    // True if ptr points inside this pool's storage (debugging/assert helper).
    bool contains(const void* ptr) const {
        const auto p = static_cast<const std::byte*>(ptr);
        return p >= base_ &&
               p < base_ + block_size_ * capacity_ &&
               (p - base_) % block_size_ == 0;
    }

private:
    // Blocks must be big enough to hold a free-list pointer and stay
    // max_align_t aligned when stepped by block_size.
    static std::size_t normalize_block_size(std::size_t block_size) {
        std::size_t size = block_size < sizeof(std::byte*)
                               ? sizeof(std::byte*)
                               : block_size;
        const std::size_t alignment = alignof(std::max_align_t);
        if (size % alignment != 0) {
            size += alignment - (size % alignment);
        }
        return size;
    }

    static std::byte* align_up(std::byte* p, std::size_t alignment) {
        const auto addr = reinterpret_cast<std::uintptr_t>(p);
        const auto aligned = (addr + alignment - 1) & ~(alignment - 1);
        return reinterpret_cast<std::byte*>(aligned);
    }

    std::byte* block_at(std::size_t i) const {
        return base_ + i * block_size_;
    }

    // The first bytes of each free block hold a pointer to the next free block.
    std::byte*& next_of(std::byte* p) {
        return *reinterpret_cast<std::byte**>(p);
    }

    std::size_t block_size_;
    std::size_t capacity_;
    std::unique_ptr<std::byte[]> storage_;
    std::byte* base_ = nullptr;
    std::byte* free_head_ = nullptr;
    std::size_t in_use_ = 0;
};

// RAII wrapper: constructs/destroys a T inside a pooled block and returns it
// to the pool on destruction. Pointer-like, so it can wrap resources.
template <typename T>
class Pooled {
public:
    template <typename... Args>
    Pooled(ObjectPool& pool, Args&&... args) : pool_(&pool) {
        ptr_ = static_cast<T*>(pool_->allocate());
        if (ptr_ == nullptr) {
            throw std::bad_alloc{};
        }
        new (ptr_) T(std::forward<Args>(args)...);
    }

    ~Pooled() {
        if (ptr_ != nullptr) {
            ptr_->~T();
            pool_->deallocate(ptr_);
        }
    }

    Pooled(const Pooled&) = delete;
    Pooled& operator=(const Pooled&) = delete;
    Pooled(Pooled&& other) noexcept : pool_(other.pool_), ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }
    // Move assignment destroys the current object and takes the other's.
    Pooled& operator=(Pooled&& other) noexcept {
        if (this != &other) {
            if (ptr_ != nullptr) {
                ptr_->~T();
                pool_->deallocate(ptr_);
            }
            pool_ = other.pool_;
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    T& operator*() { return *ptr_; }
    const T& operator*() const { return *ptr_; }
    T* operator->() { return ptr_; }
    const T* operator->() const { return ptr_; }
    T* get() { return ptr_; }

private:
    ObjectPool* pool_;
    T* ptr_ = nullptr;
};

}  // namespace chp

#endif  // CHP_OBJECT_POOL_HPP
