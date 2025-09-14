#include <stdlib.h>

#include "../include/device_memory.h"

// CSR Graph memory management
CSRGraph allocateCSRGraphOnDevice(const CSRGraph& hostGraph) {
    int *d_srcPtrs, *d_dst, *d_values;
    size_t srcPtrsSize = sizeof(int) * (hostGraph.numVertices + 1);
    size_t dstSize = sizeof(int) * hostGraph.srcPtrs[hostGraph.numVertices];

    cudaMalloc(&d_srcPtrs, srcPtrsSize);
    cudaMalloc(&d_dst, dstSize);
    cudaMalloc(&d_values, dstSize);

    cudaMemcpy(d_srcPtrs, hostGraph.srcPtrs, srcPtrsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, hostGraph.dst, dstSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, hostGraph.values, dstSize, cudaMemcpyHostToDevice);

    return {d_srcPtrs, d_dst, d_values, hostGraph.numVertices};
}

void freeCSRGraphOnDevice(CSRGraph* deviceGraph) {
    cudaFree(deviceGraph->srcPtrs);
    deviceGraph->srcPtrs = nullptr;
    cudaFree(deviceGraph->dst);
    deviceGraph->dst = nullptr;
    cudaFree(deviceGraph->values);
    deviceGraph->values = nullptr;
}

// CSC Graph memory management
CSCGraph allocateCSCGraphOnDevice(const CSCGraph& hostGraph) {
    int *d_dstPtrs, *d_src, *d_values;
    size_t dstPtrsSize = sizeof(int) * (hostGraph.numVertices + 1);
    size_t srcSize = sizeof(int) * hostGraph.dstPtrs[hostGraph.numVertices];

    cudaMalloc(&d_dstPtrs, dstPtrsSize);
    cudaMalloc(&d_src, srcSize);
    cudaMalloc(&d_values, srcSize);

    cudaMemcpy(d_dstPtrs, hostGraph.dstPtrs, dstPtrsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_src, hostGraph.src, srcSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, hostGraph.values, srcSize, cudaMemcpyHostToDevice);

    return {d_dstPtrs, d_src, d_values, hostGraph.numVertices};
}

void freeCSCGraphOnDevice(CSCGraph* deviceGraph) {
    cudaFree(deviceGraph->dstPtrs);
    deviceGraph->dstPtrs = nullptr;
    cudaFree(deviceGraph->src);
    deviceGraph->src = nullptr;
    cudaFree(deviceGraph->values);
    deviceGraph->values = nullptr;
}

// COO Graph memory management
COOGraph allocateCOOGraphOnDevice(const COOGraph& hostGraph) {
    int *d_src, *d_dst, *d_values;
    size_t edgeArraysSize = sizeof(int) * hostGraph.numEdges;

    cudaMalloc(&d_src, edgeArraysSize);
    cudaMalloc(&d_dst, edgeArraysSize);
    cudaMalloc(&d_values, edgeArraysSize);

    cudaMemcpy(d_src, hostGraph.scr, edgeArraysSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, hostGraph.dst, edgeArraysSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, hostGraph.values, edgeArraysSize, cudaMemcpyHostToDevice);

    return {d_src, d_dst, d_values, hostGraph.numEdges, hostGraph.numVertices};
}

void freeCOOGraphOnDevice(COOGraph* deviceGraph) {
    cudaFree(deviceGraph->scr);
    deviceGraph->scr = nullptr;
    cudaFree(deviceGraph->dst);
    deviceGraph->dst = nullptr;
    cudaFree(deviceGraph->values);
    deviceGraph->values = nullptr;
}

// BFS level arrays management
int* allocateAndInitLevelsOnDevice(int numVertices, int startingNode) {
    // Create host array
    int* hostLevels = (int*)malloc(sizeof(int) * numVertices);
    for (int i = 0; i < numVertices; i++) {
        hostLevels[i] = -1;
    }
    hostLevels[startingNode] = 0;

    // Allocate device memory
    int* deviceLevels;
    cudaMalloc(&deviceLevels, sizeof(int) * numVertices);
    cudaMemcpy(deviceLevels, hostLevels, sizeof(int) * numVertices, cudaMemcpyHostToDevice);

    // Free host memory
    free(hostLevels);

    return deviceLevels;
}

int* copyLevelsToHost(int* deviceLevels, int numVertices) {
    int* hostLevels = (int*)malloc(sizeof(int) * numVertices);
    cudaMemcpy(hostLevels, deviceLevels, sizeof(int) * numVertices, cudaMemcpyDeviceToHost);
    return hostLevels;
}

void freeLevelsOnDevice(int* deviceLevels) {
    cudaFree(deviceLevels);
}
