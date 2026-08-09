#include "common.cuh"
__global__ void good(const float*a,float*b,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)b[i]=a[i];}
int main(){int n=1<<25;float*a,*b;cudaMalloc(&a,n*4);cudaMalloc(&b,n*4);Timer t;t.start();for(int r=0;r<100;++r)good<<<(n+255)/256,256>>>(a,b,n);std::cout<<"coalesced ms="<<t.stop()<<'\n';cudaFree(a);cudaFree(b);}

