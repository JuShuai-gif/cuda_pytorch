// 07_container: traverse/insert costs of std containers, reserve effect,
// and a simple memory pool vs per-object new/delete.
//
// PDF 9.6-9.7 (p95-105).
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <list>
#include <map>
#include <unordered_map>
#include <vector>

#include "common/benchmark.h"

// Simple fixed-size memory pool: one big block, bump allocator.
struct Pool {
    std::vector<char> mem;
    size_t used = 0;
    explicit Pool(size_t bytes) : mem(bytes) {}
    void* alloc(size_t n) {
        void* p = mem.data() + used;
        used += n;
        return p;
    }
};

struct Node { int val; Node* next; };

int main() {
    const int n = 2'000'000;

    // --- traversal: contiguous vs linked --------------------------------
    std::vector<int> vec(n);
    for (int i = 0; i < n; ++i) vec[i] = i;
    bench("traverse_vector", [&] {
        long long s = 0; for (int x : vec) s += x; return s;
    });

    std::list<int> lst;
    for (int i = 0; i < 200'000; ++i) lst.push_back(i);  // keep it smaller
    bench("traverse_list", [&] {
        long long s = 0; for (int x : lst) s += x; return s;
    });

    std::deque<int> dq;
    for (int i = 0; i < 200'000; ++i) dq.push_back(i);
    bench("traverse_deque", [&] {
        long long s = 0; for (int x : dq) s += x; return s;
    });

    // --- lookup: map vs unordered_map -----------------------------------
    std::map<int, int> m;
    for (int i = 0; i < 200'000; ++i) m[i] = i;
    bench("map_lookup", [&] {
        long long s = 0; for (int i = 0; i < 200'000; ++i) s += m[i % 200'000]; return s;
    });

    std::unordered_map<int, int> um;
    um.reserve(300'000);
    for (int i = 0; i < 200'000; ++i) um[i] = i;
    bench("umap_lookup", [&] {
        long long s = 0; for (int i = 0; i < 200'000; ++i) s += um[i % 200'000]; return s;
    });

    // --- allocation: new/delete in a loop vs pool -----------------------
    bench("new_delete_loop", [&] {
        volatile long long s = 0;
        for (int i = 0; i < 100'000; ++i) {
            int* p = new int(i);
            s += *p;
            delete p;
        }
        return s;
    });

    Pool pool(256 * 1024 * 1024);
    bench("pool_alloc", [&] {
        volatile long long s = 0;
        for (int i = 0; i < 100'000; ++i) {
            int* p = (int*)pool.alloc(sizeof(int));
            *p = i;
            s += *p;
        }
        return s;
    });

    std::printf("\nresults: vec=%lld map=%d umap=%d\n",
                (long long)(vec.size()), m[0], um[0]);
    return 0;
}
