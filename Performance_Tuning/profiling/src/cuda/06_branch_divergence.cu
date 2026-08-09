#include "common.cuh"
__global__ void branch(float*x,int n,bool bad){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){bool p=bad?(threadIdx.x&1):(i<n/2);float v=x[i];for(int k=0;k<200;++k)v=p?sinf(v):cosf(v);x[i]=v;}}
int main(){int n=1<<23;float*x;cudaMalloc(&x,n*4);Timer t;t.start();branch<<<(n+255)/256,256>>>(x,n,true);float a=t.stop();t.start();branch<<<(n+255)/256,256>>>(x,n,false);std::cout<<"divergent_ms="<<a<<" coherent_ms="<<t.stop()<<'\n';cudaFree(x);}

