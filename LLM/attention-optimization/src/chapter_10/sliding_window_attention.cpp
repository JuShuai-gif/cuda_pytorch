#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

static void sliding_window_attention(const std::vector<float>& Q, const std::vector<float>& K,
                                     const std::vector<float>& V, std::vector<float>* O,
                                     int N, int d, int window) {
  O->assign(N * d, 0.0f);
  float scale = 1.0f / std::sqrt((float)d);
  for (int i = 0; i < N; ++i) {
    int begin = std::max(0, i - window);
    int end = std::min(N, i + window + 1);
    std::vector<float> scores(end - begin);
    float m = -INFINITY;
    for (int j = begin; j < end; ++j) {
      float dot = 0.0f;
      for (int k = 0; k < d; ++k) dot += Q[i*d+k] * K[j*d+k];
      scores[j-begin] = dot * scale;
      m = std::max(m, scores[j-begin]);
    }
    float l = 0.0f;
    for (float& s : scores) { s = std::exp(s - m); l += s; }
    for (int j = begin; j < end; ++j)
      for (int k = 0; k < d; ++k) (*O)[i*d+k] += scores[j-begin] / l * V[j*d+k];
  }
}

int main() {
  int N = 128, d = 32, window = 16;
  std::vector<float> Q(N*d), K(N*d), V(N*d), O;
  for (int i=0;i<N*d;++i) { Q[i]=std::sin(i*0.01f); K[i]=std::cos(i*0.02f); V[i]=std::sin(i*0.03f); }
  sliding_window_attention(Q,K,V,&O,N,d,window);
  long long full_pairs = 1LL * N * N;
  long long window_pairs = 0;
  for (int i=0;i<N;++i) window_pairs += std::min(N, i+window+1) - std::max(0, i-window);
  std::cout << "Sliding Window Attention demo\n";
  std::cout << "N=" << N << " d=" << d << " window=" << window << " check=" << O[0] << "\n";
  std::cout << "Pairs full=" << full_pairs << " windowed=" << window_pairs
            << " reduction=" << (double)full_pairs/window_pairs << "x\n";
  return 0;
}
