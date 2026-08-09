// Purpose: nsys找slow kernel，再用ncu过滤；包含NVTX与correctness。
#include "common.cuh"
#include <cmath>
#ifdef HAVE_NVTX
#include <nvToolsExt.h>
#define PUSH(x) nvtxRangePushA(x)
#define POP() nvtxRangePop()
#else
#define PUSH(x) ((void)0)
#define POP() ((void)0)
#endif
__global__ void fast_kernel(float*x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]+=1;}
__global__ void medium_kernel(float*x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=x[i];for(int k=0;k<20;++k)v=fmaf(v,1.00001f,.00001f);x[i]=v;}}
__global__ void slow_kernel(float*x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=x[i];for(int k=0;k<500;++k)v=fmaf(v,1.00001f,.00001f);x[i]=v;}}
int main(){int n=1<<22;float*x;CUDA_OK(cudaMalloc(&x,n*sizeof(float)));CUDA_OK(cudaMemset(x,0,n*sizeof(float)));for(int w=0;w<2;++w)slow_kernel<<<(n+255)/256,256>>>(x,n);CUDA_OK(cudaDeviceSynchronize());Timer t;t.start();for(int r=0;r<10;++r){PUSH("preprocess/kernel_fast");fast_kernel<<<(n+255)/256,256>>>(x,n);POP();PUSH("kernel_A/medium");medium_kernel<<<(n+255)/256,256>>>(x,n);POP();PUSH("kernel_B/slow");slow_kernel<<<(n+255)/256,256>>>(x,n);POP();}float ms=t.stop(),check;CUDA_OK(cudaMemcpy(&check,x,sizeof(float),cudaMemcpyDeviceToHost));if(!std::isfinite(check))return 2;std::cout<<"total_ms="<<ms<<" checksum="<<check<<"\n";cudaFree(x);}
