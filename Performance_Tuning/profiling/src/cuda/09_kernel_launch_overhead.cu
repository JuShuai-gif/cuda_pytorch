#include "common.cuh"
__global__ void tiny(int*x){if(threadIdx.x==0)atomicAdd(x,1);}
int main(){int*x;cudaMalloc(&x,4);cudaMemset(x,0,4);Timer t;t.start();for(int i=0;i<100000;++i)tiny<<<1,32>>>(x);std::cout<<"100k_launches_ms="<<t.stop()<<'\n';cudaFree(x);}

