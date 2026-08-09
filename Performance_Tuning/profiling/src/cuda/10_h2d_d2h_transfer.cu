#include "common.cuh"
int main(){std::size_t n=256ull<<20;void*h;void*d;cudaMallocHost(&h,n);cudaMalloc(&d,n);Timer t;t.start();cudaMemcpy(d,h,n,cudaMemcpyHostToDevice);float h2d=t.stop();t.start();cudaMemcpy(h,d,n,cudaMemcpyDeviceToHost);std::cout<<"H2D_ms="<<h2d<<" D2H_ms="<<t.stop()<<'\n';cudaFree(d);cudaFreeHost(h);}

