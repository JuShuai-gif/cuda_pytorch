#pragma once

#include <cstddef>

namespace chp {
namespace cvm {

// A buffer type owning a heap allocation. Counts every copy/move so tests
// can verify which operation the compiler picked.
class Buffer {
public:
    Buffer() : size_(0), data_(nullptr) {}

    explicit Buffer(std::size_t size) : size_(size), data_(new double[size]{}) {}

    ~Buffer() { delete[] data_; }

    Buffer(const Buffer& other) : size_(other.size_), data_(new double[other.size_]) {
        ++copies;
        for (std::size_t i = 0; i < size_; ++i) {
            data_[i] = other.data_[i];
        }
    }

    Buffer& operator=(const Buffer& other) {
        ++copies;
        if (this != &other) {
            delete[] data_;
            size_ = other.size_;
            data_ = new double[size_];
            for (std::size_t i = 0; i < size_; ++i) {
                data_[i] = other.data_[i];
            }
        }
        return *this;
    }

    Buffer(Buffer&& other) noexcept : size_(other.size_), data_(other.data_) {
        ++moves;
        other.size_ = 0;
        other.data_ = nullptr;
    }

    Buffer& operator=(Buffer&& other) noexcept {
        ++moves;
        if (this != &other) {
            delete[] data_;
            size_ = other.size_;
            data_ = other.data_;
            other.size_ = 0;
            other.data_ = nullptr;
        }
        return *this;
    }

    std::size_t size() const { return size_; }
    double sum() const {
        double s = 0.0;
        for (std::size_t i = 0; i < size_; ++i) {
            s += data_[i];
        }
        return s;
    }

    static void reset_counters() { copies = 0; moves = 0; }
    static int copy_count() { return copies; }
    static int move_count() { return moves; }

private:
    std::size_t size_;
    double* data_;
    static int copies;
    static int moves;
};

}  // namespace cvm
}  // namespace chp
