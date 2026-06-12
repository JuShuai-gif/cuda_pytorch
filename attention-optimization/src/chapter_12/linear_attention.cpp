#include <cmath>
#include <iostream>
#include <vector>

static float phi(float x) { return std::exp(x); }

int main() {
  int N = 128, d = 16;
  std::vector<float> Q(N*d), K(N*d), V(N*d), kv(d*d, 0.0f), ksum(d, 0.0f), O(N*d, 0.0f);
  for (int i=0;i<N*d;++i) { Q[i]=0.01f*std::sin(i); K[i]=0.01f*std::cos(i); V[i]=std::sin(i*0.03f); }
  for (int t=0;t<N;++t) {
    std::vector<float> pk(d);
    for (int i=0;i<d;++i) { pk[i]=phi(K[t*d+i]); ksum[i]+=pk[i]; }
    for (int i=0;i<d;++i) for (int j=0;j<d;++j) kv[i*d+j] += pk[i] * V[t*d+j];
  }
  for (int t=0;t<N;++t) {
    std::vector<float> pq(d); float denom = 1e-6f;
    for (int i=0;i<d;++i) { pq[i]=phi(Q[t*d+i]); denom += pq[i]*ksum[i]; }
    for (int j=0;j<d;++j) {
      float num = 0.0f;
      for (int i=0;i<d;++i) num += pq[i] * kv[i*d+j];
      O[t*d+j] = num / denom;
    }
  }
  std::cout << "Linear Attention demo\n";
  std::cout << "N=" << N << " d=" << d << " memory_state=" << (kv.size()+ksum.size())*sizeof(float)
            << " bytes check=" << O[0] << "\n";
  std::cout << "Complexity: O(N*d*d) instead of O(N*N*d) when N >> d.\n";
  return 0;
}
