// Heterogeneous containers: vector<any> vs vector<variant>.
//
// The book (PDF p.236-242): std::vector<std::any> can store anything but must
// be type-checked at runtime on every access. std::vector<std::variant<...>>
// stores a fixed set of types, lives on the stack (no heap per element), and
// supports std::visit with a polymorphic lambda.

#include <algorithm>
#include <any>
#include <cstdio>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

int main() {
    std::printf("== heterogeneous_containers ==\n");

    // --- vector<any>: store anything, but runtime type checks ---
    std::vector<std::any> any_container{42, "hi", true};
    std::printf("any container:");
    for (const auto& a : any_container) {
        if (a.type() == typeid(int)) {
            std::printf(" %d", std::any_cast<int>(a));
        } else if (a.type() == typeid(const char*)) {
            std::printf(" %s", std::any_cast<const char*>(a));
        } else if (a.type() == typeid(bool)) {
            std::printf(" %d", std::any_cast<bool>(a));
        }
    }
    std::printf("\n");

    // --- vector<variant<...>>: fixed set of types, stack storage ---
    using VariantType = std::variant<int, std::string, bool>;
    std::vector<VariantType> vc{42, std::string{"needle"}, true};

    // Print via std::visit + polymorphic lambda (PDF p.241).
    std::printf("variant container:");
    for (const auto& val : vc) {
        std::visit(
            [](const auto& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, std::string>) {
                    std::printf(" [%s]", v.c_str());
                } else {
                    std::printf(" [%d]", static_cast<int>(v));
                }
            },
            val);
    }
    std::printf("\n");

    // holds_alternative inspection (PDF p.241).
    const auto num_bools = static_cast<int>(std::count_if(
        vc.begin(), vc.end(),
        [](const auto& v) { return std::holds_alternative<bool>(v); }));
    std::printf("num bools: %d\n", num_bools);

    // Find by type AND value (PDF p.241-242).
    const bool has_needle = std::any_of(
        vc.begin(), vc.end(), [](const auto& v) {
            return std::holds_alternative<std::string>(v) &&
                   std::get<std::string>(v) == "needle";
        });
    std::printf("has string \"needle\": %d\n", has_needle);

    // sizeof: variant is the max of its alternatives (PDF p.240).
    std::printf("sizeof(any)=%zu sizeof(variant<int,string,bool>)=%zu\n",
                sizeof(std::any), sizeof(VariantType));

    return 0;
}
