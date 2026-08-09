#include "common.cuh"
__global__ void heavy(float*x){float v=x[blockIdx.x*blockDim.x+threadIdx.x];float r[64];
#pragma unroll
for(int i=0;i<64;++i)r[i]=v+i;for(int k=0;k<100;++k)for(int i=0;i<64;++i)r[i]=fmaf(r[i],1.0001f,v);for(int i=0;i<64;++i)v+=r[i];x[blockIdx.x*blockDim.x+threadIdx.x]=v;}
int main(){float*x;cudaMalloc(&x,(1<<24)*4);Timer t;t.start();heavy<<<(1<<24)/1024,1024>>>(x);std::cout<<"low_occupancy_candidate ms="<<t.stop()<<'\n';cudaFree(x);}

