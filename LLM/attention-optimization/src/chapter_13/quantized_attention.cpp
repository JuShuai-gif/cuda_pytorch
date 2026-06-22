#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
  int N = 256, d = 64;
  std::vector<float> K(N*d), scale(d, 1.0f);
  std::vector<int8_t> Kq(N*d);
  for (int i=0;i<N*d;++i) K[i] = std::sin(i*0.01f);
  for (int c=0;c<d;++c) {
    float amax = 1e-6f;
    for (int r=0;r<N;++r) amax = std::max(amax, std::fabs(K[r*d+c]));
    scale[c] = amax / 127.0f;
    for (int r=0;r<N;++r) Kq[r*d+c] = (int8_t)std::lrint(std::max(-127.0f, std::min(127.0f, K[r*d+c] / scale[c])));
  }
  double mse = 0.0;
  for (int i=0;i<N*d;++i) {
    float deq = Kq[i] * scale[i % d];
    double e = deq - K[i]; mse += e*e;
  }
  mse /= (N*d);
  std::cout << "Quantized KV cache demo\n";
  std::cout << "FP16 bytes=" << N*d*2 << " INT8 bytes=" << N*d << " scale_bytes=" << d*4 << "\n";
  std::cout << "Per-channel INT8 MSE=" << mse << "\n";
  return 0;
}
