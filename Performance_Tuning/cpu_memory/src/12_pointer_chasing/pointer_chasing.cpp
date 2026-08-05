// Experiment 12: Pointer chasing (linked list vs contiguous array).
//
// Compares std::list, a randomly-permuted node pool, a contiguous node
// pool, and a plain std::vector. Shows that linked lists are slow not just
// because of pointer overhead but because allocation scatters nodes,
// destroying spatial locality and prefetching.
//
// Reference: PDF 6.2.1, 7.3 (allocation patterns), Figure 3.11.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <list>
#include <random>
#include <vector>

#include "benchmark.h"

static constexpr int N = 1 << 20;
static constexpr int kRounds = 5;

struct Node {
    Node* next;
    int data;
};

int main() {
    std::printf("Experiment 12: pointer chasing (%d nodes)\n", N);
    std::mt19937 rng(42);

    // std::list: nodes allocated individually (scattered).
    std::list<int> lst;
    for (int i = 0; i < N; ++i) lst.push_back(i);
    auto walk_list = [&] {
        long long s = 0;
        for (int v : lst) s += v;
        bm::do_not_optimize(s);
    };

    // Randomly permuted node pool.
    std::vector<Node> pool(N);
    std::vector<Node*> perm(N);
    for (int i = 0; i < N; ++i) perm[i] = &pool[i];
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int i = 0; i < N; ++i) {
        perm[i]->data = i;
        perm[i]->next = perm[(i + 1) % N];
    }
    auto walk_randpool = [&] {
        long long s = 0;
        Node* cur = &pool[0];
        for (int i = 0; i < N; ++i) {
            s += cur->data;
            cur = cur->next;
        }
        bm::do_not_optimize(s);
    };

    // Contiguous node pool (sequential order).
    std::vector<Node> cont(N);
    for (int i = 0; i < N; ++i) {
        cont[i].data = i;
        cont[i].next = &cont[(i + 1) % N];
    }
    auto walk_cont = [&] {
        long long s = 0;
        Node* cur = &cont[0];
        for (int i = 0; i < N; ++i) {
            s += cur->data;
            cur = cur->next;
        }
        bm::do_not_optimize(s);
    };

    // Plain vector (no pointers).
    std::vector<int> vec(N);
    for (int i = 0; i < N; ++i) vec[i] = i;
    auto walk_vec = [&] {
        long long s = 0;
        for (int v : vec) s += v;
        bm::do_not_optimize(s);
    };

    struct Mode { const char* name; std::function<void()> fn; };
    Mode modes[] = {{"std_list", walk_list},
                    {"rand_pool", walk_randpool},
                    {"contiguous_pool", walk_cont},
                    {"std_vector", walk_vec}};

    std::printf("%-18s %-12s %-12s\n", "mode", "time_ms", "ns/elem");
    for (auto& m : modes) {
        m.fn();
        auto res = bm::time_rounds(kRounds, m.fn);
        std::printf("%-18s %-12.3f %-12.2f\n", m.name, res.median_ms,
                    res.median_ms * 1e6 / (double)N);
    }
    return 0;
}
