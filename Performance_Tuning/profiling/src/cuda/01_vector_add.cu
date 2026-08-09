#include "common.cuh"
__global__ void add(const float*a,const float*b,float*c,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)c[i]=a[i]+b[i];}
int main(){int n=1<<24;float *a,*b,*c;CUDA_OK(cudaMallocManaged(&a,n*4));CUDA_OK(cudaMallocManaged(&b,n*4));CUDA_OK(cudaMallocManaged(&c,n*4));for(int i=0;i<n;++i)a[i]=1,b[i]=2;add<<<(n+255)/256,256>>>(a,b,c,n);CUDA_OK(cudaDeviceSynchronize());Timer t;t.start();for(int r=0;r<100;++r)add<<<(n+255)/256,256>>>(a,b,c,n);float ms=t.stop();std::cout<<"ms="<<ms<<" GB/s="<<100.0*n*12/ms/1e6<<" check="<<c[n/2]<<'\n';cudaFree(a);cudaFree(b);cudaFree(c);}

