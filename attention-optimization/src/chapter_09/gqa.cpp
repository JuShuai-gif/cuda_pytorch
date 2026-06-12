#include <iostream>

static double kv_cache_mb(int layers, int kv_heads, int seq_len, int head_dim, int bytes) {
  return layers * 2.0 * kv_heads * seq_len * head_dim * bytes / (1024.0 * 1024.0);
}

int main() {
  int layers = 80, q_heads = 64, kv_heads = 8, seq = 4096, head_dim = 128, bytes = 2;
  double mha = kv_cache_mb(layers, q_heads, seq, head_dim, bytes);
  double gqa = kv_cache_mb(layers, kv_heads, seq, head_dim, bytes);
  std::cout << "GQA KV cache demo\n";
  std::cout << "Q heads: " << q_heads << ", KV heads: " << kv_heads << "\n";
  std::cout << "MHA KV cache: " << mha << " MB\n";
  std::cout << "GQA KV cache: " << gqa << " MB\n";
  std::cout << "Reduction: " << (mha / gqa) << "x\n";
  return 0;
}
