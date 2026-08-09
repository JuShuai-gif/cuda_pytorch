#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#define CUDA_OK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){std::cerr<<cudaGetErrorString(e)<<" @"<<__LINE__<<'\n';std::exit(1);}}while(0)
struct Timer{cudaEvent_t a,b;Timer(){CUDA_OK(cudaEventCreate(&a));CUDA_OK(cudaEventCreate(&b));}~Timer(){cudaEventDestroy(a);cudaEventDestroy(b);}void start(){CUDA_OK(cudaEventRecord(a));}float stop(){CUDA_OK(cudaEventRecord(b));CUDA_OK(cudaEventSynchronize(b));float ms;CUDA_OK(cudaEventElapsedTime(&ms,a,b));return ms;}};

