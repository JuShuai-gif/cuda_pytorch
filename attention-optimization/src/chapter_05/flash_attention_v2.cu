/**
 * Chapter 05 FlashAttention V2 work-partition simulator.
 *
 * This is a correctness-first CUDA demo, not the production FA2 kernel. It keeps
 * one query row per block and uses online softmax so readers can compare the
 * work partition and IO model with Chapter04.
 */
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(err) do { cudaError_t e=(err); if(e!=cudaSuccess){fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

__global__ void fa2_rowwise_demo(const float* Q, const float* K, const float* V,
                                 float* O, int N, int d) {
  int row = blockIdx.x;
  int lane = threadIdx.x;
  if (row >= N || lane >= d) return;
  float scale = rsqrtf((float)d);
  float m = -INFINITY;
  float l = 0.0f;
  float acc = 0.0f;
  for (int j = 0; j < N; ++j) {
    float dot = 0.0f;
    for (int k = 0; k < d; ++k) dot += Q[row*d+k] * K[j*d+k];
    float s = dot * scale;
    float m_new = fmaxf(m, s);
    float p = expf(s - m_new);
    float rescale = expf(m - m_new);
    acc = acc * rescale + p * V[j*d+lane];
    l = l * rescale + p;
    m = m_new;
  }
  O[row*d+lane] = acc / l;
}

int main() {
  int N = 128, d = 64;
  std::vector<float> hQ(N*d), hK(N*d), hV(N*d), hO(N*d);
  for (int i=0;i<N*d;++i) { hQ[i]=std::sin(i*0.01f); hK[i]=std::cos(i*0.02f); hV[i]=std::sin(i*0.03f); }
  float *Q,*K,*V,*O;
  CUDA_CHECK(cudaMalloc(&Q,N*d*sizeof(float))); CUDA_CHECK(cudaMalloc(&K,N*d*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&V,N*d*sizeof(float))); CUDA_CHECK(cudaMalloc(&O,N*d*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(Q,hQ.data(),N*d*sizeof(float),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(K,hK.data(),N*d*sizeof(float),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(V,hV.data(),N*d*sizeof(float),cudaMemcpyHostToDevice));
  cudaEvent_t a,b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaEventRecord(a));
  for (int i=0;i<50;++i) fa2_rowwise_demo<<<N,d>>>(Q,K,V,O,N,d);
  CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b));
  float ms; CUDA_CHECK(cudaEventElapsedTime(&ms,a,b)); ms/=50.0f;
  CUDA_CHECK(cudaMemcpy(hO.data(),O,N*d*sizeof(float),cudaMemcpyDeviceToHost));
  printf("FlashAttention V2 demo: N=%d d=%d latency=%.4f ms check=%.6f\n", N,d,ms,hO[0]);
  printf("Teaching point: FA2 improves work partitioning and reduces non-matmul overhead; this demo isolates online softmax correctness.\n");
  CUDA_CHECK(cudaFree(Q)); CUDA_CHECK(cudaFree(K)); CUDA_CHECK(cudaFree(V)); CUDA_CHECK(cudaFree(O));
  return 0;
}
