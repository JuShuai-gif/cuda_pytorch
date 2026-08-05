#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <list>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "benchmark.hpp"
#include "baseline.hpp"
#include "optimized.hpp"

namespace {

constexpr std::size_t kBookCount = 1'000'000;
constexpr std::size_t kPoolSize = 200;
constexpr std::size_t kIterations = 3;
constexpr std::size_t kRounds = 5;
constexpr std::size_t kWarmup = 2;

}  // namespace

int main() {
    std::printf("== linked_list_search benchmark ==\n");

    std::mt19937 gen(12345u);
    std::vector<std::string> pool;
    pool.reserve(kPoolSize);
    for (std::size_t i = 0; i < kPoolSize; ++i) {
        pool.push_back("Title-" + std::to_string(i));
    }
    pool[0] = "Hamlet";

    std::vector<std::string> books;
    books.reserve(kBookCount);
    std::uniform_int_distribution<std::size_t> dist(0, pool.size() - 1);
    for (std::size_t i = 0; i < kBookCount; ++i) {
        books.push_back(pool[dist(gen)]);
    }

    std::list<std::string> book_list(books.begin(), books.end());

    // C-style linked list: every node is heap allocated individually and
    // chained through the `next` pointer, like the C code in the book.
    std::vector<std::unique_ptr<chp::lls::CNode>> nodes;
    nodes.reserve(kBookCount);
    for (std::size_t i = 0; i < kBookCount; ++i) {
        auto node = std::make_unique<chp::lls::CNode>();
        node->title = books[i].c_str();
        if (!nodes.empty()) {
            nodes.back()->next = node.get();
        }
        nodes.push_back(std::move(node));
    }
    const chp::lls::CNode* head = nodes.front().get();
    const std::string hamlet = "Hamlet";

    const auto r_c = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(
                chp::lls::count_title_c_style(head, hamlet.c_str()));
        });
    const auto r_v = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(
                chp::lls::count_title_stl_vector(books, hamlet));
        });
    const auto r_l = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(
                chp::lls::count_title_stl_list(book_list, hamlet));
        });

    std::printf("Data: %zu books, \"Hamlet\" occurs %zu times\n", kBookCount,
                chp::lls::count_title_stl_vector(books, hamlet));
    std::printf("(checksum across impls should be equal)\n\n");

    chp::print_result("C-style linked list + manual loop", r_c);
    chp::print_result("std::list + std::count", r_l);
    chp::print_result("std::vector + std::count", r_v);

    if (r_c.checksum == r_v.checksum && r_v.checksum == r_l.checksum) {
        std::printf("Checksums identical: all implementations agreed.\n");
        return 0;
    }
    std::printf("ERROR: checksums differ!\n");
    return 1;
}
