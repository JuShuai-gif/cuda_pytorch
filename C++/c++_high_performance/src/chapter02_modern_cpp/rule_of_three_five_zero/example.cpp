// Demonstrates the Rule of Three, Rule of Five, and Rule of Zero.
//
// The book (PDF p.67-77) explains:
//  - Rule of Three: if a class manages a resource, you need the copy
//    constructor, copy assignment, and destructor.
//  - Rule of Five: add move constructor and move assignment (noexcept!).
//  - Rule of Zero: prefer classes whose special members are all implicitly
//    correct (no manual resource handling).

#include <cstddef>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace {

class RuleOfFive {
public:
    RuleOfFive() = default;
    explicit RuleOfFive(std::size_t size) : size_(size), data_(new int[size]{}), tagged_(size) {}

    ~RuleOfFive() { delete[] data_; }

    RuleOfFive(const RuleOfFive& other)
        : size_(other.size_), data_(new int[other.size_]), tagged_(other.tagged_) {
        for (std::size_t i = 0; i < size_; ++i) {
            data_[i] = other.data_[i];
        }
    }

    RuleOfFive& operator=(const RuleOfFive& other) {
        if (this != &other) {
            RuleOfFive tmp(other);
            std::swap(size_, tmp.size_);
            std::swap(data_, tmp.data_);
            std::swap(tagged_, tmp.tagged_);
        }
        return *this;
    }

    RuleOfFive(RuleOfFive&& other) noexcept
        : size_(other.size_), data_(other.data_), tagged_(other.tagged_) {
        other.size_ = 0;
        other.data_ = nullptr;
    }

    RuleOfFive& operator=(RuleOfFive&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            size_ = other.size_;
            data_ = other.data_;
            tagged_ = other.tagged_;
            other.size_ = 0;
            other.data_ = nullptr;
        }
        return *this;
    }

    std::size_t size() const { return size_; }
    std::size_t tag() const { return tagged_; }

private:
    std::size_t size_ = 0;
    int* data_ = nullptr;
    std::size_t tagged_ = 0;
};

// Rule of Zero: no user-declared special members; the members manage their
// own resources and the compiler-generated copy/move/destructor are correct.
struct RuleOfZero {
    std::vector<int> values;
    std::string name;
};

}  // namespace

int main() {
    std::printf("== rule_of_three_five_zero ==\n");

    RuleOfFive a(100);
    RuleOfFive b = a;          // copy
    b = a;                     // copy assignment
    RuleOfFive c = std::move(b);  // move
    RuleOfFive d(10);
    d = std::move(c);          // move assignment
    std::printf("sizes after moves: c=%zu d=%zu\n", c.size(), d.size());

    RuleOfZero za{{1, 2, 3}, "hello"};
    RuleOfZero zb = za;  // implicit copy
    zb.values.push_back(4);
    std::printf("rule of zero: za.size=%zu zb.size=%zu\n", za.values.size(),
                zb.values.size());
    return 0;
}
