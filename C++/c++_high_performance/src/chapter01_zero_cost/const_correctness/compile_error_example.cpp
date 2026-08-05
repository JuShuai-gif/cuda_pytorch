// This file is INTENTIONALLY ill-formed.
//
// Purpose: demonstrate that const correctness makes certain bugs
// compile-time errors instead of runtime surprises.
//
// It is NOT part of the normal build. To see the expected diagnostics run:
//
//   g++ -std=c++17 -c const_correctness/compile_error_example.cpp
//   clang++ -std=c++17 -c const_correctness/compile_error_example.cpp
//
// Expected: "passing 'const Person' as 'this' argument discards qualifiers".

#include <vector>

namespace {

class Person {
public:
    int age() const { return age_; }
    void set_age(int age) { age_ = age; }

private:
    int age_ = 0;
};

class Team {
public:
    const Person& leader() const { return leader_; }
    Person& leader() { return leader_; }

private:
    Person leader_;
};

}  // namespace

// Calling a non-const member function through a const object must not compile.
void cannot_mutate_const(const std::vector<Team>& teams) {
    for (auto& team : teams) {  // Error: cannot bind non-const lvalue to const
        team.leader().set_age(20);  // Error: set_age() discards const
    }
}
