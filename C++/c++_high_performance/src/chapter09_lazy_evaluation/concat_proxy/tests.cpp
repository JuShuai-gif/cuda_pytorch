// Correctness checks for concat_proxy.

#include <cstdio>
#include <string>
#include <utility>

#include "test_utils.hpp"

namespace {

class String {
public:
    String() = default;
    String(std::string str) : str_(std::move(str)) {}
    const std::string& str() const { return str_; }

private:
    std::string str_;
};

struct ConcatProxy {
    const std::string& a;
    const std::string& b;
    operator String() const&& { return String{a + b}; }
};

bool is_concat_equal(const std::string& a, const std::string& b,
                     const std::string& c) {
    return a.size() + b.size() == c.size() &&
           std::equal(a.begin(), a.end(), c.begin()) &&
           std::equal(b.begin(), b.end(), c.begin() + a.size());
}

auto operator+(const String& a, const String& b) {
    return ConcatProxy{a.str(), b.str()};
}

auto operator==(ConcatProxy&& concat, const String& rhs) -> bool {
    return is_concat_equal(concat.a, concat.b, rhs.str());
}

}  // namespace

int main() {
    const String a{"Cole"};
    const String b{"Porter"};

    CHP_CHECK((a + b) == String{"ColePorter"});
    CHP_CHECK(!((a + b) == String{"Cole"}));
    CHP_CHECK(!((a + b) == String{"ColePorterX"}));
    CHP_CHECK(!((a + b) == String{"ColePorterLONGTARGET"}));

    // Assigning the concatenation requires an explicit String target.
    const String c = String{"Marc"} + String{"Chagall"};
    CHP_CHECK(c.str() == "MarcChagall");

    return chp::test_summary("concat_proxy");
}
