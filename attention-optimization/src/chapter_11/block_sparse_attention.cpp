#include <cmath>
#include <iostream>
#include <vector>

int main() {
  int N = 256, block = 16;
  int nb = N / block;
  std::vector<std::vector<int>> pattern(nb);
  for (int i = 0; i < nb; ++i) {
    pattern[i].push_back(i);              // diagonal/local block
    if (i > 0) pattern[i].push_back(i-1); // previous local block
    pattern[i].push_back(0);              // global block
  }
  long long dense_blocks = 1LL * nb * nb;
  long long sparse_blocks = 0;
  for (auto& row : pattern) sparse_blocks += row.size();
  std::cout << "Block Sparse Attention pattern demo\n";
  std::cout << "N=" << N << " block=" << block << " dense_blocks=" << dense_blocks
            << " sparse_blocks=" << sparse_blocks << " density="
            << (100.0 * sparse_blocks / dense_blocks) << "%\n";
  std::cout << "Teaching point: speedup requires block-level regularity, not random element sparsity.\n";
  return 0;
}
