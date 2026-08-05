#pragma once

#include <cassert>
#include <cstddef>
#include <iterator>

namespace chp {
namespace lr {

// Utility functions (book PDF p.138).
template <typename T>
T get_step_size(T start, T stop, std::size_t n) {
    assert(n >= 2);
    return (stop - start) / static_cast<T>(n - 1);
}

template <typename T>
T get_linear_value(T start, T step_size, std::size_t idx) {
    return start + step_size * static_cast<T>(idx);
}

// A bidirectional iterator over a linear range of values generated on the fly
// (book PDF p.139-140). reference is value_type because the value is computed,
// not stored; pointer is void because there is no addressable element.
template <typename T>
class LinearRangeIterator {
public:
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using reference = T;
    using pointer = void;
    using iterator_category = std::bidirectional_iterator_tag;

    LinearRangeIterator(T start, T step_size, std::size_t idx)
        : start_(start), step_size_(step_size), idx_(idx) {}

    bool operator==(const LinearRangeIterator& other) const {
        return idx_ == other.idx_;
    }
    bool operator!=(const LinearRangeIterator& other) const {
        return !(*this == other);
    }
    LinearRangeIterator& operator++() {
        ++idx_;
        return *this;
    }
    LinearRangeIterator operator++(int) {
        LinearRangeIterator tmp = *this;
        ++(*this);
        return tmp;
    }
    LinearRangeIterator& operator--() {
        --idx_;
        return *this;
    }
    LinearRangeIterator operator--(int) {
        LinearRangeIterator tmp = *this;
        --(*this);
        return tmp;
    }
    T operator*() const { return get_linear_value(start_, step_size_, idx_); }

private:
    T start_;
    T step_size_;
    std::size_t idx_;
};

// The range object that produces begin()/end() (book PDF p.141).
template <typename T>
class LinearRange {
public:
    using iterator = LinearRangeIterator<T>;

    LinearRange(T start, T stop, std::size_t num_values)
        : start_(start),
          step_size_(get_step_size(start, stop, num_values)),
          num_values_(num_values) {}

    iterator begin() const { return iterator{start_, step_size_, 0}; }
    iterator end() const { return iterator{start_, step_size_, num_values_}; }

private:
    T start_;
    T step_size_;
    std::size_t num_values_;
};

// Convenience factory that deduces T (book PDF p.141-142).
template <typename T>
LinearRange<T> make_linear_range(T start, T stop, std::size_t n) {
    return LinearRange<T>{start, stop, n};
}

}  // namespace lr
}  // namespace chp
