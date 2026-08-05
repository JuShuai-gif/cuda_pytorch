// std::any: stores a value of any type at runtime, at the cost of type
// erasure. The book (PDF p.83-84) notes std::any heap-allocates its stored
// value (implementations are encouraged to use a small-object optimization)
// and that any_cast is slower than a direct typed access or std::variant.

#include <any>
#include <cstddef>
#include <cstdio>
#include <string>

namespace {

template <typename T>
bool is_withheld_type(const std::any& a) {
    return typeid(T) == a.type();
}

}  // namespace

int main() {
    std::printf("== any_demo ==\n");
    std::printf("sizeof(std::any) = %zu\n", sizeof(std::any));

    std::any a;  // empty
    a = std::string{"something"};
    auto& str_ref = std::any_cast<std::string&>(a);
    auto str_copy = std::any_cast<std::string>(a);
    str_ref += "!";  // modifies the value inside the any
    std::printf("any holds string: \"%s\" (copy was \"%s\")\n",
                std::any_cast<std::string>(a).c_str(), str_copy.c_str());

    a = 135.246F;
    auto flt = std::any_cast<float>(a);
    std::printf("any holds float: %g\n", flt);

    // Copying an any object.
    auto b = a;
    const bool is_same = (a.type() == b.type()) &&
                         (std::any_cast<float>(a) == std::any_cast<float>(b));
    std::printf("a equals b: %d\n", is_same);

    // Type checking helper.
    std::any d = 32.0;
    std::printf("is int? %d, is double? %d\n", is_withheld_type<int>(d),
                is_withheld_type<double>(d));

    // Wrong type throws std::bad_any_cast.
    try {
        (void)std::any_cast<int>(d);
    } catch (const std::bad_any_cast&) {
        std::printf("any_cast<int> on a double threw bad_any_cast (expected)\n");
    }
    return 0;
}
