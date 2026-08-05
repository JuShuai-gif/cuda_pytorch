#include <cstdio>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "test_utils.hpp"

namespace {

class BagelValue {
public:
    explicit BagelValue(const std::set<std::string>& toppings)
        : toppings_(toppings) {}
    const std::set<std::string>& toppings() const { return toppings_; }

private:
    std::set<std::string> toppings_;
};

class BagelShared {
public:
    explicit BagelShared(std::shared_ptr<std::set<std::string>> toppings)
        : toppings_(std::move(toppings)) {}
    std::set<std::string>& toppings() { return *toppings_; }

private:
    std::shared_ptr<std::set<std::string>> toppings_;
};

}  // namespace

int main() {
    // Value semantics: later mutations of the source set must not leak in.
    auto toppings = std::set<std::string>{"salt"};
    auto a = BagelValue{toppings};
    toppings.insert("pepper");
    auto b = BagelValue{toppings};
    toppings.insert("oregano");

    CHP_CHECK(a.toppings().count("salt") == 1);
    CHP_CHECK(a.toppings().count("pepper") == 0);
    CHP_CHECK(a.toppings().count("oregano") == 0);
    CHP_CHECK(b.toppings().count("salt") == 1);
    CHP_CHECK(b.toppings().count("pepper") == 1);
    CHP_CHECK(b.toppings().count("oregano") == 0);
    CHP_CHECK(a.toppings() != b.toppings());

    // Reference semantics: mutations through any handle must be visible.
    auto shared = std::make_shared<std::set<std::string>>();
    shared->insert("salt");
    auto c = BagelShared{shared};
    shared->insert("pepper");
    auto d = BagelShared{shared};
    shared->insert("oregano");

    CHP_CHECK(c.toppings().count("oregano") == 1);
    CHP_CHECK(d.toppings().count("pepper") == 1);
    CHP_CHECK(c.toppings() == d.toppings());
    CHP_CHECK(&c.toppings() == &d.toppings());

    return chp::test_summary("value_semantics");
}
