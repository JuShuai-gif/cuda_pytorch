#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(err) do { cudaError_t e=(err); if(e!=cudaSuccess){fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

extern "C" void print_plugin_contract();

__global__ void fused_attention_demo(const float* Q, const float* K, const float* V, float* O, int N, int d) {
  int row = blockIdx.x;
  int col = threadIdx.x;
  if (row >= N || col >= d) return;
  float scale = rsqrtf((float)d), m = -INFINITY, l = 0.0f, acc = 0.0f;
  for (int j=0;j<N;++j) {
    float dot = 0.0f;
    for (int k=0;k<d;++k) dot += Q[row*d+k] * K[j*d+k];
    float s = dot * scale;
    float mn = fmaxf(m, s);
    float p = expf(s - mn);
    float r = expf(m - mn);
    acc = acc * r + p * V[j*d+col];
    l = l * r + p;
    m = mn;
  }
  O[row*d+col] = acc / l;
}

int main() {
  print_plugin_contract();
  int N=64,d=32; std::vector<float> h(N*d,0.1f), out(N*d);
  float *Q,*K,*V,*O; CUDA_CHECK(cudaMalloc(&Q,N*d*sizeof(float))); CUDA_CHECK(cudaMalloc(&K,N*d*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&V,N*d*sizeof(float))); CUDA_CHECK(cudaMalloc(&O,N*d*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(Q,h.data(),N*d*sizeof(float),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(K,h.data(),N*d*sizeof(float),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(V,h.data(),N*d*sizeof(float),cudaMemcpyHostToDevice));
  fused_attention_demo<<<N,d>>>(Q,K,V,O,N,d); CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(out.data(),O,N*d*sizeof(float),cudaMemcpyDeviceToHost));
  printf("Mini TensorRT-LLM fused attention demo: check=%.6f\n", out[0]);
  CUDA_CHECK(cudaFree(Q)); CUDA_CHECK(cudaFree(K)); CUDA_CHECK(cudaFree(V)); CUDA_CHECK(cudaFree(O));
  return 0;
}
