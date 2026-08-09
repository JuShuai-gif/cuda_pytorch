#pragma once

// Single-producer / single-consumer lock-free queue (PDF p.309-311).
//
// Fixed-capacity ring buffer. Only size_ is atomic; read_pos_ belongs to
// the reader and write_pos_ to the writer, so no locks are needed. Both
// threads are lock-free -- ideal for a real-time audio thread that cannot
// block or allocate.

#include <array>
#include <atomic>
#include <cstddef>
#include <stdexcept>

namespace chp10 {

template <class T, std::size_t N>
class LockFreeQueue {
public:
    LockFreeQueue() : size_{0} {
        if (!size_.is_lock_free()) {
            throw std::runtime_error{"size_ is not lock-free"};
        }
    }

    std::size_t size() const { return size_.load(); }

    // Writer thread only.
    void push(const T& t) {
        if (size_.load() >= N) {
            throw std::overflow_error("Queue is full");
        }
        buffer_[write_pos_] = t;
        write_pos_ = (write_pos_ + 1) % N;
        size_.fetch_add(1);
    }

    // Reader thread only.
    const T& front() const {
        if (size_.load() == 0) {
            throw std::underflow_error("Queue is empty");
        }
        return buffer_[read_pos_];
    }

    // Reader thread only.
    void pop() {
        if (size_.load() == 0) {
            throw std::underflow_error("Queue is empty");
        }
        read_pos_ = (read_pos_ + 1) % N;
        size_.fetch_sub(1);
    }

private:
    std::array<T, N> buffer_{};  // shared, but threads never touch same slot
    std::atomic<std::size_t> size_{};  // the only shared mutable state
    std::size_t read_pos_ = 0;   // reader thread only
    std::size_t write_pos_ = 0;  // writer thread only
};

}  // namespace chp10
