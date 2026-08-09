#include "common.cuh"
#ifdef HAVE_NVTX
#include <nvToolsExt.h>
#define RANGE(x) nvtxRangePushA(x)
#define END() nvtxRangePop()
#else
#define RANGE(x) ((void)0)
#define END() ((void)0)
#endif
__global__ void work(float*x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)for(int k=0;k<100;++k)x[i]=tanhf(x[i]);}
int main(){int n=1<<24;std::size_t z=n*4;float*h,*d;cudaMallocHost(&h,z);cudaMalloc(&d,z);for(int r=0;r<10;++r){RANGE("preprocess");for(int i=0;i<n;++i)h[i]=i*.001f;END();RANGE("H2D");cudaMemcpy(d,h,z,cudaMemcpyHostToDevice);END();RANGE("GEMM/activation");work<<<(n+255)/256,256>>>(d,n);END();RANGE("D2H");cudaMemcpy(h,d,z,cudaMemcpyDeviceToHost);END();RANGE("postprocess");volatile float v=h[r];(void)v;END();}cudaDeviceSynchronize();std::cout<<"NVTX pipeline done\n";cudaFree(d);cudaFreeHost(h);}
