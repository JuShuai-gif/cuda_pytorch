#include "common.cuh"
__global__ void compute(float*x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=x[i];
#pragma unroll 1
for(int k=0;k<2000;++k)v=fmaf(v,1.000001f,.000001f);x[i]=v;}}
int main(){int n=1<<22;float*x;CUDA_OK(cudaMalloc(&x,n*4));Timer t;t.start();compute<<<(n+255)/256,256>>>(x,n);std::cout<<"compute_bound ms="<<t.stop()<<'\n';cudaFree(x);}

