#include "common.cuh"
__global__ void bad(const float*a,float*b,int n,int stride){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)b[i]=a[(static_cast<long long>(i)*stride)%n];}
int main(){int n=1<<25;float*a,*b;cudaMalloc(&a,n*4);cudaMalloc(&b,n*4);Timer t;t.start();for(int r=0;r<100;++r)bad<<<(n+255)/256,256>>>(a,b,n,33);std::cout<<"uncoalesced ms="<<t.stop()<<'\n';cudaFree(a);cudaFree(b);}

