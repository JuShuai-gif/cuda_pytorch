#include "common.cuh"
__global__ void triad(const float*a,const float*b,float*c,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)c[i]=a[i]+2*b[i];}
int main(){int n=1<<25;float*a,*b,*c;CUDA_OK(cudaMalloc(&a,n*4));CUDA_OK(cudaMalloc(&b,n*4));CUDA_OK(cudaMalloc(&c,n*4));Timer t;t.start();for(int r=0;r<200;++r)triad<<<(n+255)/256,256>>>(a,b,c,n);float ms=t.stop();std::cout<<"memory_bound ms="<<ms<<" effective_GB/s="<<200.0*n*12/ms/1e6<<'\n';cudaFree(a);cudaFree(b);cudaFree(c);}

