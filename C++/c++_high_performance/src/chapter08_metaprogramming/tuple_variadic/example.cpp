// std::tuple, structured bindings, variadic packs, and tuple algorithms.
//
// The book (PDF p.227-236): tuple is a static heterogeneous container.
// tuple_for_each unrolls a loop at compile time with if constexpr; a variadic
// parameter pack can be wrapped into a tuple and iterated the same way.

#include <cstddef>
#include <cstdio>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace {

// tuple_at: apply a functor to one element (PDF p.230).
template <std::size_t Index, typename Tuple, typename Functor>
void tuple_at(const Tuple& tpl, const Functor& func) {
    const auto& v = std::get<Index>(tpl);
    func(v);
}

// tuple_for_each: unroll the tuple (PDF p.230-231).
template <typename Tuple, typename Functor, std::size_t Index = 0>
void tuple_for_each(const Tuple& tpl, const Functor& f) {
    constexpr auto tuple_size = std::tuple_size_v<Tuple>;
    if constexpr (Index < tuple_size) {
        tuple_at<Index>(tpl, f);
        tuple_for_each<Tuple, Functor, Index + 1>(tpl, f);
    }
}

// tuple_any_of (PDF p.231).
template <typename Tuple, typename Functor, std::size_t Index = 0>
bool tuple_any_of(const Tuple& tpl, const Functor& f) {
    constexpr auto tuple_size = std::tuple_size_v<Tuple>;
    if constexpr (Index < tuple_size) {
        return f(std::get<Index>(tpl)) ? true
                                       : tuple_any_of<Tuple, Functor,
                                                      Index + 1>(tpl, f);
    } else {
        return false;
    }
}

// Variadic parameter pack -> tuple -> iterate (PDF p.236).
template <typename... Ts>
std::string make_string(const Ts&... values) {
    std::ostringstream sstr;
    auto tuple = std::tie(values...);
    tuple_for_each(tuple, [&sstr](const auto& v) { sstr << v << " "; });
    return sstr.str();
}

}  // namespace

int main() {
    std::printf("== tuple_variadic ==\n");

    // Construct and access (PDF p.227-228).
    auto tuple0 = std::make_tuple(42, std::string{"hi"}, true);
    std::printf("get<0>=%d get<1>=%s get<2>=%d\n", std::get<0>(tuple0),
                std::get<1>(tuple0).c_str(), std::get<2>(tuple0));

    // Structured bindings (PDF p.232-233).
    const auto& [num, str, flag] = tuple0;
    std::printf("structured: %d %s %d\n", num, str.c_str(), flag);

    // tuple_for_each (PDF p.230-231).
    auto tpl = std::make_tuple(1, true, std::string{"Jedi"});
    std::printf("for_each: ");
    tuple_for_each(tpl, [](const auto& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::string>) {
            std::printf("[%s] ", v.c_str());
        } else {
            std::printf("[%d] ", static_cast<int>(v));
        }
    });
    std::printf("\n");

    // tuple_any_of (PDF p.231).
    auto t2 = std::make_tuple(42, 43.0F, 44.0);
    const bool has_44 = tuple_any_of(t2, [](auto v) { return v == 44; });
    std::printf("any_of == 44: %d\n", has_44);

    // Variadic make_string (PDF p.234-236).
    std::printf("make_string: %s\n", make_string(42, "hi", true).c_str());

    return 0;
}
