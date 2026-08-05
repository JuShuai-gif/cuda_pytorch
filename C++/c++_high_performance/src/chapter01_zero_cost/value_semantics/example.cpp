#include <cstdio>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

// Value semantics: the toppings set is copied on construction, so each
// instance is fully isolated from the set it was created from.
class BagelValue {
public:
    explicit BagelValue(const std::set<std::string>& toppings)
        : toppings_(toppings) {}
    const std::set<std::string>& toppings() const { return toppings_; }

private:
    std::set<std::string> toppings_;
};

// Reference semantics (Java-like): the toppings set is shared through a
// pointer, so mutations made through any handle are visible everywhere.
class BagelShared {
public:
    explicit BagelShared(std::shared_ptr<std::set<std::string>> toppings)
        : toppings_(std::move(toppings)) {}
    std::set<std::string>& toppings() { return *toppings_; }

private:
    std::shared_ptr<std::set<std::string>> toppings_;
};

void print_toppings(const char* label, const std::set<std::string>& s) {
    std::printf("%-20s {", label);
    for (const auto& t : s) {
        std::printf(" %s", t.c_str());
    }
    std::printf(" }\n");
}

void change_value(std::set<std::string> s) { s.insert("local"); }

void change_reference(std::set<std::string>& s) { s.insert("local"); }

}  // namespace

int main() {
    std::printf("== value semantics vs reference semantics ==\n\n");

    // --- Bagel isolation (value semantics) ---
    auto toppings = std::set<std::string>{"salt"};
    auto a = BagelValue{toppings};
    toppings.insert("pepper");
    auto b = BagelValue{toppings};
    toppings.insert("oregano");

    print_toppings("value bagel 'a':", a.toppings());
    print_toppings("value bagel 'b':", b.toppings());

    // --- Bagel sharing (reference semantics, Java-like) ---
    auto shared = std::make_shared<std::set<std::string>>();
    shared->insert("salt");
    auto c = BagelShared{shared};
    shared->insert("pepper");
    auto d = BagelShared{shared};
    shared->insert("oregano");

    print_toppings("shared bagel 'c':", c.toppings());
    print_toppings("shared bagel 'd':", d.toppings());

    // --- Passing arguments by value vs by reference ---
    std::set<std::string> local{"original"};
    change_value(local);
    print_toppings("after change_value:", local);
    change_reference(local);
    print_toppings("after change_reference:", local);

    return 0;
}
