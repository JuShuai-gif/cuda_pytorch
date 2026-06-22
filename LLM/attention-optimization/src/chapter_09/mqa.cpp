#include <cmath>
#include <iostream>

static double kv_cache_mb(int layers, int kv_heads, int seq_len, int head_dim, int bytes) {
  return layers * 2.0 * kv_heads * seq_len * head_dim * bytes / (1024.0 * 1024.0);
}

int main() {
  int layers = 80, q_heads = 64, seq = 4096, head_dim = 128, bytes = 2;
  double mha = kv_cache_mb(layers, q_heads, seq, head_dim, bytes);
  double mqa = kv_cache_mb(layers, 1, seq, head_dim, bytes);
  std::cout << "MQA KV cache demo\n";
  std::cout << "MHA KV cache: " << mha << " MB\n";
  std::cout << "MQA KV cache: " << mqa << " MB\n";
  std::cout << "Reduction: " << (mha / mqa) << "x\n";
  return 0;
}
