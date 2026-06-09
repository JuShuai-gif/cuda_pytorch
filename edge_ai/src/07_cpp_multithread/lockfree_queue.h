#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>

template <typename T, size_t Capacity>
class LockFreeQueue {
    static_assert(Capacity >= 2, "容量必须至少为 2");
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "容量必须是 2 的幂");

public:
    LockFreeQueue() : head_(0), tail_(0) {
        for (size_t i = 0; i < Capacity; ++i) {
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
        }
    }

    LockFreeQueue(const LockFreeQueue &) = delete;
    LockFreeQueue &operator=(const LockFreeQueue &) = delete;

    bool try_push(T &&value) {
        Cell *cell;
        size_t pos = tail_.load(std::memory_order_relaxed);
        for (;;) {
            cell = &buffer_[pos & (Capacity - 1)];
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq)
                            - static_cast<intptr_t>(pos);
            if (diff == 0) {
                if (tail_.compare_exchange_weak(pos, pos + 1,
                                                std::memory_order_relaxed))
                    break;
            } else if (diff < 0) {
                return false; // 队列已满
            } else {
                pos = tail_.load(std::memory_order_relaxed);
            }
        }
        cell->data = std::move(value);
        cell->sequence.store(pos + 1, std::memory_order_release);
        return true;
    }

    bool try_push(const T &value) {
        T copy(value);
        return try_push(std::move(copy));
    }

    bool try_pop(T &result) {
        Cell *cell;
        size_t pos = head_.load(std::memory_order_relaxed);
        for (;;) {
            cell = &buffer_[pos & (Capacity - 1)];
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq)
                            - static_cast<intptr_t>(pos + 1);
            if (diff == 0) {
                if (head_.compare_exchange_weak(pos, pos + 1,
                                                std::memory_order_relaxed))
                    break;
            } else if (diff < 0) {
                return false; // 队列为空
            } else {
                pos = head_.load(std::memory_order_relaxed);
            }
        }
        result = std::move(cell->data);
        cell->sequence.store(pos + Capacity, std::memory_order_release);
        return true;
    }

    bool empty() const {
        return head_.load(std::memory_order_relaxed)
               == tail_.load(std::memory_order_relaxed);
    }

    size_t capacity() const {
        return Capacity;
    }

private:
    static constexpr size_t CACHE_LINE_SIZE = 64;

    struct Cell {
        std::atomic<size_t> sequence;
        T data;
    };

    // 使用填充使 head_ 和 tail_ 避免伪共享
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> head_;
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> tail_;

    // 索引字段与缓冲区数组之间的填充
    char padding_[CACHE_LINE_SIZE - sizeof(std::atomic<size_t>)];
    Cell buffer_[Capacity];
};
