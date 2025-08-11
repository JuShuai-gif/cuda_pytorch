#ifndef _COMMON_H_
#define _COMMON_H_

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(call)                                     \
    {                                                   \
        const cudaError_t error = call;                 \
        if (error != cudaSuccess) {                     \
            printf("CUDA Error:\n");                    \
            printf("    File:       %s\n", __FILE__);   \
            printf("    Line:       %d\n", __LINE__);   \
            printf("    Error code: %d\n", error); \
            printf("    Error text: %s\n",              \
                   cudaGetErrorString(error));     \
            exit(1);                                    \
        }                                               \
    }

#include <time.h>

#include <sys/time.h>

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

void initialData(float *ip, int size) {
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xffff) / 1000.0f;
    }
}
void initialData_int(int *ip, int size) {
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++) {
        ip[i] = int(rand() & 0xff);
    }
}

void printMatrix(float *C, const int nx, const int ny) {
    float *ic = C;
    printf("Matrix<%d,%d>: \n", ny, nx);
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            printf("%6f", ic[j]);
        }
        ic += nx;
        printf("\n");
    }
}

void initDevice(int devNum) {
    int dev = devNum;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
}

void checkResult(float *hostRef, float *gpuRef, const int N) {
    double epsilon = 1.0E-8;
    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            printf("Results don\'t match!\n");
            printf("%f(hostRef[%d] )!= %f(gpuRef[%d])\n", hostRef[i], i, gpuRef[i], i);
            return;
        }
    }
    printf("Check result success!\n");
}

void checkResultInt(int *hostRef, int *gpuRef, const int N) {
    double epsilon = 1.0E-8;
    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            printf("Results don\'t match!\n");
            printf("%f(hostRef[%d] )!= %f(gpuRef[%d])\n", hostRef[i], i, gpuRef[i], i);
            return;
        }
    }
    printf("Check result success!\n");
}

#endif // _COMMON_H_