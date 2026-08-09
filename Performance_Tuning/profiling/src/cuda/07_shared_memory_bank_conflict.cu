#include "common.cuh"
__global__ void banks(float*out,int stride){__shared__ float s[32*33];int t=threadIdx.x;s[t*stride]=t;__syncthreads();float v=0;for(int k=0;k<2000;++k)v+=s[t*stride];out[blockIdx.x*32+t]=v;}
int main(){float*x;cudaMalloc(&x,4096*32*4);Timer t;t.start();banks<<<4096,32>>>(x,32);float bad=t.stop();t.start();banks<<<4096,32>>>(x,33);std::cout<<"conflict_ms="<<bad<<" padded_ms="<<t.stop()<<'\n';cudaFree(x);}

