#include "common.cuh"
__global__ void tiny(float*x){int i=blockIdx.x*blockDim.x+threadIdx.x;x[i]+=1;}
int main(){float*x;cudaMalloc(&x,(1<<20)*4);Timer t;t.start();for(int i=0;i<1000;++i){tiny<<<4096,256>>>(x);cudaDeviceSynchronize();}float bad=t.stop();t.start();for(int i=0;i<1000;++i)tiny<<<4096,256>>>(x);std::cout<<"sync_each_ms="<<bad<<" async_batch_ms="<<t.stop()<<'\n';cudaFree(x);}

