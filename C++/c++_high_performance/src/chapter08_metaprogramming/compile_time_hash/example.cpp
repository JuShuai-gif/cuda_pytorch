// PrehashedString: compute a hash at compile time.
//
// The book (PDF p.252-257): a resource cache uses unordered_map<string,...>.
// Each lookup hashes the string at runtime. PrehashedString precomputes the
// hash when constructed from a string literal, and forces the input to be a
// compile-time literal via a template <size_t N> const char(&)[N] constructor.

#include <cstddef>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <utility>

namespace {

// A compile-time hash function (PDF p.254). Summing chars is a BAD hash for
// real use, but is easy to verify in assembly (book uses it for clarity).
constexpr std::size_t hash_function(const char* str) {
    std::size_t sum = 0;
    for (auto ptr = str; *ptr != '\0'; ++ptr) {
        sum += static_cast<std::size_t>(*ptr);
    }
    return sum;
}

class PrehashedString {
public:
    // Only accepts string literals (compile-time arrays).
    template <std::size_t N>
    constexpr PrehashedString(const char (&str)[N])
        : hash_(hash_function(&str[0])), size_(N - 1), strptr_(&str[0]) {}

    bool operator==(const PrehashedString& s) const {
        if (size_ != s.size_) {
            return false;
        }
        for (std::size_t i = 0; i < size_; ++i) {
            if (strptr_[i] != s.strptr_[i]) {
                return false;
            }
        }
        return true;
    }
    bool operator!=(const PrehashedString& s) const { return !(*this == s); }

    constexpr std::size_t size() const { return size_; }
    constexpr std::size_t get_hash() const { return hash_; }
    constexpr const char* c_str() const { return strptr_; }

private:
    std::size_t hash_;
    std::size_t size_;
    const char* strptr_;
};

}  // namespace

namespace std {
template <>
struct hash<PrehashedString> {
    constexpr std::size_t operator()(const PrehashedString& s) const {
        return s.get_hash();
    }
};
}  // namespace std

int main() {
    std::printf("== compile_time_hash ==\n");

    // Compile-time verification.
    constexpr auto h = hash_function("abc");
    static_assert(h == 294, "97+98+99");
    std::printf("hash_function(\"abc\") = %zu\n", h);

    // PrehashedString: hash is a compile-time constant.
    constexpr PrehashedString ps{"abc"};
    static_assert(ps.get_hash() == 294, "compile-time hash");
    std::printf("PrehashedString{\"abc\"} hash = %zu size = %zu\n",
                ps.get_hash(), ps.size());

    // Usable as a key in unordered_map with the std::hash specialization.
    std::unordered_map<PrehashedString, int> cache;
    cache.emplace(PrehashedString{"my_bitmap.png"}, 1);
    const int found = cache.count(PrehashedString{"my_bitmap.png"});
    std::printf("cache lookup \"my_bitmap.png\": %d\n", found);

    return 0;
}
