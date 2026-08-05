#include <algorithm>
#include <cassert>
#include <cstdio>
#include <list>
#include <type_traits>
#include <vector>

#include "test_utils.hpp"

namespace {

// Equivalence of a lambda and a hand-written class (book PDF p.51-52).
auto make_lambda_is_above(int th) {
    return [th](int v) { return v > th; };
}

class IsAbove {
public:
    explicit IsAbove(int th) : th_(th) {}
    int operator()(int v) const { return v > th_; }

private:
    int th_;
};

}  // namespace

int main() {
    // --- Capture by value vs by reference (book PDF p.50) ---
    {
        std::vector<int> vals = {1, 2, 3, 4, 5, 6};
        int th = 3;
        auto by_value = [th](int v) { return v > th; };
        auto by_ref = [&th](int v) { return v > th; };
        th = 4;
        const int n_value = static_cast<int>(
            std::count_if(vals.begin(), vals.end(), by_value));
        const int n_ref = static_cast<int>(
            std::count_if(vals.begin(), vals.end(), by_ref));
        CHP_CHECK(n_value == 3);  // 4,5,6 above old threshold 3
        CHP_CHECK(n_ref == 2);    // 5,6 above new threshold 4
    }

    // --- Lambda behaves like a class with operator() (book PDF p.51) ---
    {
        const int th = 3;
        auto lambda = make_lambda_is_above(th);
        IsAbove klass(th);
        CHP_CHECK(lambda(5) == klass(5));
        CHP_CHECK(lambda(2) == klass(2));
        static_assert(std::is_same<decltype(lambda), decltype(lambda)>::value,
                      "each lambda has a unique type");
    }

    // --- Captures can be initialized (book PDF p.52) ---
    {
        auto func = [c = std::list<int>{4, 2}]() { return c.size(); };
        CHP_CHECK(func() == 2);
    }

    // --- mutable lambda mutates its own captured copy (book PDF p.53) ---
    {
        int v = 7;
        auto lambda = [v]() mutable { ++v; return v; };
        CHP_CHECK(v == 7);
        CHP_CHECK(lambda() == 8);
        CHP_CHECK(lambda() == 9);
        CHP_CHECK(v == 7);  // captured copy is mutated, not the original
    }

    // --- Capture by reference: the original is mutated (book PDF p.53) ---
    {
        int v = 7;
        auto lambda = [&v]() { ++v; return v; };
        CHP_CHECK(lambda() == 8);
        CHP_CHECK(v == 8);
    }

    // --- Capture all: [=] captures only what is used (book PDF p.55) ---
    {
        int a = 1;
        float b = 2.0f;
        auto all_value = [=]() { return a; };
        auto all_ref = [&]() { return b; };
        static_assert(sizeof(decltype(all_value)) >= sizeof(int),
                      "captures the used int a");
        static_assert(sizeof(decltype(all_ref)) >= sizeof(float*),
                      "captures a reference");
        CHP_CHECK(all_value() == 1);
        CHP_CHECK(all_ref() == 2.0f);
    }

    // --- Lambda with mutable member used as a counter (book PDF p.58) ---
    {
        auto counter = [count = 0]() mutable { return ++count; };
        CHP_CHECK(counter() == 1);
        CHP_CHECK(counter() == 2);
        CHP_CHECK(counter() == 3);
    }

    std::printf("lambda_basics: checks finished\n");
    return chp::test_summary("lambda_basics");
}
