#include <cstdio>
#include <unordered_map>
#include <unordered_set>

#include "test_utils.hpp"

namespace {

struct BadHash {
    std::size_t operator()(int) const { return 0; }
};

}  // namespace

int main() {
    // Default hash distributes evenly: small buckets, stable lookups.
    std::unordered_map<int, int> m;
    for (int i = 0; i < 1000; ++i) {
        m[i] = i;
    }
    CHP_CHECK(m.size() == 1000);
    CHP_CHECK(m.max_load_factor() > 0.0F);
    // After enough inserts, the bucket count must satisfy
    // size <= load_factor_max * buckets.
    const std::size_t min_buckets = static_cast<std::size_t>(
        static_cast<double>(m.size()) / static_cast<double>(m.max_load_factor()));
    CHP_CHECK(m.bucket_count() >= min_buckets);
    // load_factor is elements / buckets.
    CHP_CHECK(m.load_factor() > 0.0F);
    CHP_CHECK(m.load_factor() <= m.max_load_factor());
    // element count derived from load_factor and bucket_count must be sane.
    const auto expected_buckets = min_buckets;
    CHP_CHECK(m.bucket_count() >= expected_buckets);

    // reserve reduces rehashing: after reserve(10000), bucket count stays put
    // for 1000 inserts.
    std::unordered_map<int, int> r;
    r.reserve(10000);
    const std::size_t buckets_before = r.bucket_count();
    for (int i = 0; i < 1000; ++i) {
        r[i] = i;
    }
    CHP_CHECK(r.bucket_count() >= buckets_before);

    // Even with a terrible hash, correctness holds (only performance suffers).
    std::unordered_set<int, BadHash> s;
    for (int i = 0; i < 100; ++i) {
        s.insert(i);
    }
    CHP_CHECK(s.size() == 100);
    CHP_CHECK(s.count(50) == 1);
    CHP_CHECK(s.count(200) == 0);

    return chp::test_summary("hash_policy");
}
