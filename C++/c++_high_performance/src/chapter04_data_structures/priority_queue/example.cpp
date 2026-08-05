// priority_queue as a partial sort (book PDF p.117-120).
//
// To find the top-m highest ranking hits we keep a min-heap of size m. Each
// new hit replaces the current minimum if it ranks higher. Complexity: O(n *
// log m) time, O(m) memory - cheaper than sorting all n hits.

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <queue>
#include <random>
#include <string>
#include <vector>

namespace {

struct Document {
    std::string title;
};

struct Hit {
    float rank = 0.0F;
    std::shared_ptr<Document> document;
};

// Find the top-m hits by rank, using a min-heap (priority_queue).
std::vector<Hit> sort_hits(const std::vector<Hit>& hits, std::size_t m) {
    // Min-heap: the top of the queue is the LOWEST rank kept so far.
    auto cmp = [](const Hit& a, const Hit& b) { return a.rank > b.rank; };
    std::priority_queue<Hit, std::vector<Hit>, decltype(cmp)> queue(cmp);

    for (const auto& hit : hits) {
        if (queue.size() < m) {
            queue.push(hit);
        } else if (hit.rank > queue.top().rank) {
            queue.pop();
            queue.push(hit);
        }
    }

    std::vector<Hit> result;
    result.reserve(queue.size());
    while (!queue.empty()) {
        result.push_back(queue.top());
        queue.pop();
    }
    std::reverse(result.begin(), result.end());
    return result;
}

}  // namespace

int main() {
    std::printf("== priority_queue (partial sort) ==\n");

    // 100k random hits with ranks in [0, 1000).
    std::mt19937 gen(42u);
    std::uniform_real_distribution<float> dist(0.0F, 1000.0F);
    std::vector<Hit> hits(100'000);
    for (auto& h : hits) {
        h.rank = dist(gen);
        h.document = std::make_shared<Document>();
        h.document->title = "doc";
    }

    const std::size_t m = 10;
    const auto top = sort_hits(hits, m);
    std::printf("top-%zu ranks:", m);
    for (const auto& h : top) {
        std::printf(" %.1f", h.rank);
    }
    std::printf("\n");

    // Verify: the top-m must be the m largest ranks in descending order.
    std::vector<float> all;
    all.reserve(hits.size());
    for (const auto& h : hits) {
        all.push_back(h.rank);
    }
    std::sort(all.begin(), all.end(), std::greater<float>{});
    bool ok = top.size() == m;
    for (std::size_t i = 0; i < m && ok; ++i) {
        if (top[i].rank != all[i]) {
            ok = false;
        }
    }
    std::printf("top-m matches full sort: %s\n", ok ? "yes" : "NO");
    return ok ? 0 : 1;
}
