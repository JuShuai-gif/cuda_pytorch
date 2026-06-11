/***
 * KV Cache implementation in C++ - Chapter 07.
 *
 * Implements:
 * - KV Cache data structure
 * - Append operation
 * - Single-token decode attention with cached K,V
 * - Prefill vs Decode performance comparison
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ----------------------------------------------------------------------
// KV Cache data structure
// ----------------------------------------------------------------------
struct KVCache {
    std::vector<float> k_cache; // [max_seq_len, n_heads, head_dim]
    std::vector<float> v_cache; // [max_seq_len, n_heads, head_dim]
    int max_seq_len;
    int n_heads;
    int head_dim;
    int cur_len;

    KVCache(int max_len, int nh, int hd) : max_seq_len(max_len), n_heads(nh), head_dim(hd), cur_len(0) {
        k_cache.resize(max_len * nh * hd, 0.0f);
        v_cache.resize(max_len * nh * hd, 0.0f);
    }

    void append(const float *new_k, const float *new_v, int n_tokens) {
        int offset = cur_len * n_heads * head_dim;
        int size = n_tokens * n_heads * head_dim;
        std::memcpy(k_cache.data() + offset, new_k, size * sizeof(float));
        std::memcpy(v_cache.data() + offset, new_v, size * sizeof(float));
        cur_len += n_tokens;
    }

    void clear() {
        cur_len = 0;
    }
};

// ----------------------------------------------------------------------
// Naive softmax
// ----------------------------------------------------------------------
static void softmax(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; ++i)
        if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < n; ++i)
        x[i] /= sum;
}

// ----------------------------------------------------------------------
// Decode attention: single token query with KV cache
//
// Q_single: [1, n_heads, head_dim]
// K_cache:  [cur_len, n_heads, head_dim]
// V_cache:  [cur_len, n_heads, head_dim]
// O:        [1, n_heads, head_dim]
// ----------------------------------------------------------------------
static void decode_attention_kv_cache(
    const float *Q_single,
    const KVCache &cache,
    float *O) {
    int N = cache.n_heads;
    int d = cache.head_dim;
    int L = cache.cur_len;
    float scale = 1.0f / std::sqrt(static_cast<float>(d));

    for (int h = 0; h < N; ++h) {
        // Step 1: Q @ K^T → [L] scores
        std::vector<float> scores(L);
        for (int t = 0; t < L; ++t) {
            float dot = 0.0f;
            for (int dd = 0; dd < d; ++dd) {
                dot += Q_single[h * d + dd] * cache.k_cache[t * N * d + h * d + dd];
            }
            scores[t] = dot * scale;
        }

        // Step 2: Softmax
        softmax(scores.data(), L);

        // Step 3: P @ V → [d]
        for (int dd = 0; dd < d; ++dd) {
            float accum = 0.0f;
            for (int t = 0; t < L; ++t) {
                accum += scores[t] * cache.v_cache[t * N * d + h * d + dd];
            }
            O[h * d + dd] = accum;
        }
    }
}

// ----------------------------------------------------------------------
// Benchmark
// ----------------------------------------------------------------------
#include <chrono>
#include <iomanip>
#include <iostream>

int main() {
    std::cout << "KV Cache - C++ Implementation\n";
    std::cout << std::string(60, '=') << "\n";

    const int n_heads = 8;
    const int head_dim = 64;
    const int max_len = 4096;

    KVCache cache(max_len, n_heads, head_dim);

    // Prefill: populate cache with random data
    int prefill_len = 512;
    std::vector<float> Q_prefill(prefill_len * n_heads * head_dim);
    std::vector<float> K_prefill(prefill_len * n_heads * head_dim);
    std::vector<float> V_prefill(prefill_len * n_heads * head_dim);
    for (auto &x : K_prefill) x = static_cast<float>(rand()) / RAND_MAX;
    for (auto &x : V_prefill) x = static_cast<float>(rand()) / RAND_MAX;
    cache.append(K_prefill.data(), V_prefill.data(), prefill_len);

    std::cout << "Prefilled " << prefill_len << " tokens\n";
    std::cout << "Cache memory: "
              << (cache.k_cache.size() + cache.v_cache.size()) * sizeof(float) / (1024.0 * 1024.0)
              << " MB\n\n";

    // Decode benchmark: single token
    std::vector<float> Q_single(1 * n_heads * head_dim);
    std::vector<float> O(1 * n_heads * head_dim);
    for (auto &x : Q_single) x = static_cast<float>(rand()) / RAND_MAX;

    // Warmup
    for (int w = 0; w < 5; ++w)
        decode_attention_kv_cache(Q_single.data(), cache, O.data());

    int iters = 500;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
        decode_attention_kv_cache(Q_single.data(), cache, O.data());
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    double flops = 2.0 * prefill_len * n_heads * head_dim + 4.0 * prefill_len * n_heads
                   + 2.0 * prefill_len * n_heads * head_dim;
    double gflops = (flops / 1e9) / (ms / 1000.0);

    std::cout << "Decode benchmark (CPU, single-threaded):\n";
    std::cout << "  Cache length:   " << prefill_len << "\n";
    std::cout << "  Latency:        " << ms << " ms\n";
    std::cout << "  Throughput:     " << gflops << " GFLOPS\n";
    std::cout << "  I/O volume:     "
              << (prefill_len * n_heads * head_dim * 2 * sizeof(float) / (1024.0 * 1024.0))
              << " MB read per decode step\n";

    std::cout << "\n"
              << std::string(60, '=') << "\n";
    std::cout << "Key insight: Decode is memory-bound.\n";
    std::cout << "Each decode step must read the ENTIRE KV cache.\n";
    std::cout << "This is why KV cache compression (GQA/MQA/quantization) matters.\n";

    return 0;
}
