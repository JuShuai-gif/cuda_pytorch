/**
 * Mini PagedAttention implementation in C++.
 *
 * Core idea: KV Cache is split into fixed-size blocks.
 * Each request has a block_table mapping logical → physical blocks.
 * Physical blocks don't need to be contiguous.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <set>
#include <vector>

constexpr int BLOCK_SIZE = 16; // tokens per block

// ----------------------------------------------------------------------
// Physical KV Cache
// ----------------------------------------------------------------------
struct PhysicalKVCache {
    std::vector<float> k_blocks; // [num_blocks * BLOCK_SIZE, n_heads, head_dim]
    std::vector<float> v_blocks;
    std::vector<bool> allocated; // Which physical blocks are in use
    int num_blocks;
    int n_heads;
    int head_dim;

    PhysicalKVCache(int total_blocks, int nh, int hd) : num_blocks(total_blocks), n_heads(nh), head_dim(hd) {
        int block_elems = BLOCK_SIZE * nh * hd;
        k_blocks.resize(total_blocks * block_elems, 0.0f);
        v_blocks.resize(total_blocks * block_elems, 0.0f);
        allocated.resize(total_blocks, false);
    }

    int allocate_block() {
        for (int i = 0; i < num_blocks; ++i) {
            if (!allocated[i]) {
                allocated[i] = true;
                return i;
            }
        }
        return -1; // No free block
    }

    void free_block(int block_idx) {
        assert(block_idx >= 0 && block_idx < num_blocks);
        allocated[block_idx] = false;
    }

    int num_free() const {
        int count = 0;
        for (bool a : allocated)
            if (!a) count++;
        return count;
    }

    float *k_block_ptr(int block_idx) {
        return k_blocks.data() + block_idx * BLOCK_SIZE * n_heads * head_dim;
    }
    float *v_block_ptr(int block_idx) {
        return v_blocks.data() + block_idx * BLOCK_SIZE * n_heads * head_dim;
    }
};

// ----------------------------------------------------------------------
// Block Table for one request
// ----------------------------------------------------------------------
struct BlockTable {
    std::vector<int> physical_blocks; // logical block → physical block idx
    int n_tokens;                     // total tokens in this request

    BlockTable() : n_tokens(0) {
    }

    int num_blocks() const {
        return physical_blocks.size();
    }
    int num_empty_slots() const {
        return num_blocks() * BLOCK_SIZE - n_tokens;
    }
};

// ----------------------------------------------------------------------
// Block Allocator
// ----------------------------------------------------------------------
struct BlockAllocator {
    PhysicalKVCache &cache;
    std::set<int> free_set; // Track free blocks for O(log N) allocation

    BlockAllocator(PhysicalKVCache &c) : cache(c) {
        for (int i = 0; i < cache.num_blocks; ++i)
            free_set.insert(i);
    }

    int allocate_block() {
        if (free_set.empty()) return -1;
        int block = *free_set.begin();
        free_set.erase(free_set.begin());
        cache.allocated[block] = true;
        return block;
    }

    void free_block(int block) {
        cache.allocated[block] = false;
        free_set.insert(block);
    }

    BlockTable allocate_request(int n_tokens) {
        BlockTable bt;
        int needed = (n_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (int i = 0; i < needed; ++i) {
            int block = allocate_block();
            if (block < 0) {
                // Rollback
                for (int b : bt.physical_blocks) free_block(b);
                return BlockTable(); // Failed
            }
            bt.physical_blocks.push_back(block);
        }
        bt.n_tokens = n_tokens;
        return bt;
    }

    void free_request(BlockTable &bt) {
        for (int block : bt.physical_blocks)
            free_block(block);
        bt.physical_blocks.clear();
        bt.n_tokens = 0;
    }
};

// ----------------------------------------------------------------------
// PagedAttention: compute attention using block table
// ----------------------------------------------------------------------
static void paged_attention(
    const float *Q, // [1, n_heads, d]
    const PhysicalKVCache &cache,
    const BlockTable &block_table,
    float *O, // [1, n_heads, d]
    int n_heads, int d) {
    float scale = 1.0f / std::sqrt(static_cast<float>(d));
    int total_tokens = block_table.n_tokens;

    for (int h = 0; h < n_heads; ++h) {
        // Online softmax state
        float m = -INFINITY;
        float l = 0.0f;
        std::vector<float> O_acc(d, 0.0f);

        for (int logical_block = 0; logical_block < block_table.num_blocks(); ++logical_block) {
            int phys_block = block_table.physical_blocks[logical_block];
            int tokens_in_block = std::min(BLOCK_SIZE, total_tokens - logical_block * BLOCK_SIZE);

            float *k_block = cache.k_blocks.data()
                             + phys_block * BLOCK_SIZE * n_heads * d;
            float *v_block = cache.v_blocks.data()
                             + phys_block * BLOCK_SIZE * n_heads * d;

            // Process each token in this block
            for (int t = 0; t < tokens_in_block; ++t) {
                // Q @ K^T
                float dot = 0.0f;
                for (int dd = 0; dd < d; ++dd)
                    dot += Q[h * d + dd] * k_block[t * n_heads * d + h * d + dd];
                float score = dot * scale;

                // Online softmax update
                float m_prev = m;
                m = std::max(m, score);
                float rescale = std::exp(m_prev - m);
                float P = std::exp(score - m);

                l = l * rescale + P;

                for (int dd = 0; dd < d; ++dd) {
                    O_acc[dd] = O_acc[dd] * rescale
                                + P * v_block[t * n_heads * d + h * d + dd];
                }
            }
        }

        // Normalize
        float inv_l = 1.0f / l;
        for (int dd = 0; dd < d; ++dd)
            O[h * d + dd] = O_acc[dd] * inv_l;
    }
}

// ----------------------------------------------------------------------
// Main: demonstrate block allocation and paged attention
// ----------------------------------------------------------------------
int main() {
    std::cout << "Mini PagedAttention - C++ Implementation\n";
    std::cout << std::string(60, '=') << "\n";

    const int n_heads = 4;
    const int head_dim = 64;
    const int total_blocks = 32;

    PhysicalKVCache cache(total_blocks, n_heads, head_dim);
    BlockAllocator allocator(cache);

    // Simulate 3 concurrent requests
    std::cout << "Initial free blocks: " << cache.num_free() << "\n\n";

    // Request 1: 100 tokens → needs ceil(100/16) = 7 blocks
    auto bt1 = allocator.allocate_request(100);
    std::cout << "Request 1: 100 tokens → " << bt1.num_blocks() << " blocks (";
    for (size_t i = 0; i < bt1.physical_blocks.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << bt1.physical_blocks[i];
    }
    std::cout << ")\n";
    std::cout << "  Free blocks remaining: " << cache.num_free() << "\n";

    // Request 2: 50 tokens → needs ceil(50/16) = 4 blocks
    auto bt2 = allocator.allocate_request(50);
    std::cout << "Request 2: 50 tokens → " << bt2.num_blocks() << " blocks (";
    for (size_t i = 0; i < bt2.physical_blocks.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << bt2.physical_blocks[i];
    }
    std::cout << ")\n";
    std::cout << "  Free blocks remaining: " << cache.num_free() << "\n";

    // Request 3: 200 tokens → needs ceil(200/16) = 13 blocks
    auto bt3 = allocator.allocate_request(200);
    std::cout << "Request 3: 200 tokens → " << bt3.num_blocks() << " blocks (";
    for (size_t i = 0; i < bt3.physical_blocks.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << bt3.physical_blocks[i];
    }
    std::cout << ")\n";
    std::cout << "  Free blocks remaining: " << cache.num_free() << "\n";

    // Free Request 1
    std::cout << "\nFreeing Request 1...\n";
    allocator.free_request(bt1);
    std::cout << "  Free blocks: " << cache.num_free() << "\n";

    // Verify fragmentation is minimal
    std::cout << "\n"
              << std::string(60, '=') << "\n";
    std::cout << "Key advantages of PagedAttention:\n";
    std::cout << "  1. Blocks can be allocated non-contiguously\n";
    std::cout << "  2. No external fragmentation (all blocks are same size)\n";
    std::cout << "  3. Internal fragmentation ≤ BLOCK_SIZE-1 tokens per request\n";
    std::cout << "  4. Memory utilization improved from ~30% to ~90%+\n";
    std::cout << "  5. This is the core innovation behind vLLM\n";

    // Clean up remaining
    allocator.free_request(bt2);
    allocator.free_request(bt3);

    return 0;
}
