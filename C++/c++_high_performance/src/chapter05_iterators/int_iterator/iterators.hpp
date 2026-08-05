#pragma once

#include <cstddef>
#include <iterator>

namespace chp {
namespace iter {

// A forward iterator that generates integers on the fly (book PDF p.133).
class IntIterator {
public:
    using difference_type = std::ptrdiff_t;
    using value_type = int;
    using reference = int&;
    using pointer = int*;
    using iterator_category = std::forward_iterator_tag;

    explicit IntIterator(int v) : value_(v) {}

    bool operator==(const IntIterator& other) const {
        return value_ == other.value_;
    }
    bool operator!=(const IntIterator& other) const {
        return !(*this == other);
    }
    int& operator*() { return value_; }
    IntIterator& operator++() {
        ++value_;
        return *this;
    }
    IntIterator operator++(int) {
        IntIterator tmp = *this;
        ++(*this);
        return tmp;
    }

private:
    int value_;
};

// A bidirectional version: adds operator-- (book PDF p.136).
class BidirectionalIntIterator {
public:
    using difference_type = std::ptrdiff_t;
    using value_type = int;
    using reference = int&;
    using pointer = int*;
    using iterator_category = std::bidirectional_iterator_tag;

    explicit BidirectionalIntIterator(int v) : value_(v) {}

    bool operator==(const BidirectionalIntIterator& other) const {
        return value_ == other.value_;
    }
    bool operator!=(const BidirectionalIntIterator& other) const {
        return !(*this == other);
    }
    int& operator*() { return value_; }
    BidirectionalIntIterator& operator++() {
        ++value_;
        return *this;
    }
    BidirectionalIntIterator operator++(int) {
        BidirectionalIntIterator tmp = *this;
        ++(*this);
        return tmp;
    }
    BidirectionalIntIterator& operator--() {
        --value_;
        return *this;
    }
    BidirectionalIntIterator operator--(int) {
        BidirectionalIntIterator tmp = *this;
        --(*this);
        return tmp;
    }

private:
    int value_;
};

}  // namespace iter
}  // namespace chp
