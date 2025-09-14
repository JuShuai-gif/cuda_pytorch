#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "../include/bfs_parallel.h"
#include "../include/device_memory.h"
#include "../include/graph_structures.h"

#define LOCAL_FRONTIER_CAPACITY 256

// KERNEL DEFINITIONS
__global__ void bsf_push_vertex_centric_kernel(CSRGraph graph, int* levels, int* newVertexVisitd,
                                               unsigned int currLevel) {
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex < graph.numVertices) {
        if (levels[vertex] == currLevel - 1) {
            // iterate over all the vertices at the current levels
            for (unsigned int edge = graph.srcPtrs[vertex]; edge < graph.srcPtrs[vertex + 1]; edge++) {
                unsigned int neighbour = graph.dst[edge];
                // not yet visited
                if (levels[neighbour] == -1) {
                    levels[neighbour] = currLevel;
                    *newVertexVisitd = 1;  // set the flag to 1 so we know we need to run it again
                }
            }
        }
    }
}

__global__ void bsf_pull_vertex_centric_kernel(CSCGraph graph, int* levels, int* newVertexVisitd,
                                               unsigned int currLevel) {
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;

    if (vertex < graph.numVertices) {
        if (levels[vertex] == -1) {  // Vertext not yet visited
            for (unsigned int edge = graph.dstPtrs[vertex]; edge < graph.dstPtrs[vertex + 1]; edge++) {
                unsigned int neighbour = graph.src[edge];

                if (levels[neighbour] == currLevel - 1) {
                    levels[vertex] = currLevel;
                    *newVertexVisitd = 1;
                    break;  // if any of the neighbour at the prev levels we reached our point
                }
            }
        }
    }
}

__global__ void bsf_edge_centric_kernel(COOGraph cooGraph, int* levels, int* newVertexVisitd, unsigned int currLevel) {
    unsigned int edge = blockIdx.x * blockDim.x + threadIdx.x;

    if (edge < cooGraph.numEdges) {
        unsigned int vertex = cooGraph.scr[edge];
        if (levels[vertex] == currLevel - 1) {
            unsigned int neighbour = cooGraph.dst[edge];
            if (levels[neighbour] == -1) {  // neighbour not yet visited
                levels[neighbour] = currLevel;
                *newVertexVisitd = 1;
            }
        }
    }
}

__global__ void bsf_frontier_vertex_centric_kernel(CSRGraph csrGraph, int* levels, int* prevFrontier, int* currFrontier,
                                                   int numPrevFrontier, int* numCurrentFrontier,
                                                   unsigned int currLevel) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numPrevFrontier) {
        unsigned int vertex = prevFrontier[i];
        for (unsigned int edge = csrGraph.srcPtrs[vertex]; edge < csrGraph.srcPtrs[vertex + 1]; edge++) {
            unsigned int neighbour = csrGraph.dst[edge];
            // neighbour not visited
            if (atomicCAS(&levels[neighbour], -1, currLevel) == -1) {
                unsigned int currFrontierIdx = atomicAdd(numCurrentFrontier, 1);
                currFrontier[currFrontierIdx] = neighbour;
            }
        }
    }
}

__global__ void bsf_frontier_vertex_centric_with_privatization_kernel(CSRGraph csrGraph, int* levels, int* prevFrontier,
                                                                      int* currFrontier, int numPrevFrontier,
                                                                      int* numCurrFrontier, int currLevel) {
    // Initialize privatized frontier
    __shared__ unsigned int currFrontier_s[LOCAL_FRONTIER_CAPACITY];
    __shared__ unsigned int numCurrFrontier_s;
    if (threadIdx.x == 0) {
        numCurrFrontier_s = 0;
    }
    __syncthreads();

    // Perform BFS
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPrevFrontier) {
        unsigned int vertex = prevFrontier[i];
        for (unsigned int edge = csrGraph.srcPtrs[vertex]; edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
            unsigned int neighbor = csrGraph.dst[edge];
            if (atomicCAS(&levels[neighbor], UINT_MAX, currLevel) == UINT_MAX) {
                unsigned int currFrontierIdx_s = atomicAdd(&numCurrFrontier_s, 1);
                if (currFrontierIdx_s < LOCAL_FRONTIER_CAPACITY) {
                    currFrontier_s[currFrontierIdx_s] = neighbor;
                } else {
                    numCurrFrontier_s = LOCAL_FRONTIER_CAPACITY;
                    unsigned int currFrontierIdx = atomicAdd(numCurrFrontier, 1);
                    currFrontier[currFrontierIdx] = neighbor;
                }
            }
        }
    }
    __syncthreads();

    // Allocate in global frontier
    __shared__ unsigned int currFrontierStartIdx;
    if (threadIdx.x == 0) {
        currFrontierStartIdx = atomicAdd(numCurrFrontier, numCurrFrontier_s);
    }
    __syncthreads();

    // Commit to global frontier
    for (unsigned int currFrontierIdx_s = threadIdx.x; currFrontierIdx_s < numCurrFrontier_s;
         currFrontierIdx_s += blockDim.x) {
        unsigned int currFrontierIdx = currFrontierStartIdx + currFrontierIdx_s;
        currFrontier[currFrontierIdx] = currFrontier_s[currFrontierIdx_s];
    }
}

// ORIGINAL HOST-BASED IMPLEMENTATIONS

int* bfsParallelPushVertexCentric(const CSRGraph& hostGraph, int startingNode) {
    // Initialize host levels array
    int* hostLevels = (int*)malloc(sizeof(int) * hostGraph.numVertices);
    for (int i = 0; i < hostGraph.numVertices; i++) {
        hostLevels[i] = -1;
    }
    hostLevels[startingNode] = 0;

    // Calculate size needed for graph arrays
    size_t scrPtrsSize = sizeof(int) * (hostGraph.numVertices + 1);
    size_t dstSize = sizeof(int) * hostGraph.srcPtrs[hostGraph.numVertices];
    size_t vertexSize = sizeof(int) * hostGraph.numVertices;

    // Allocate device memory
    int *d_srcPtrs, *d_dst, *d_values, *d_levels, *d_newVertexVisited;
    cudaMalloc(&d_srcPtrs, scrPtrsSize);
    cudaMalloc(&d_dst, dstSize);
    cudaMalloc(&d_values, dstSize);
    cudaMalloc(&d_levels, vertexSize);
    cudaMalloc(&d_newVertexVisited, sizeof(int));

    // Copy values to cuda
    cudaMemcpy(d_srcPtrs, hostGraph.srcPtrs, scrPtrsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, hostGraph.dst, dstSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, hostGraph.values, dstSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_levels, hostLevels, vertexSize, cudaMemcpyHostToDevice);

    CSRGraph deviceGraph = {
        .srcPtrs = d_srcPtrs, .dst = d_dst, .values = d_values, .numVertices = hostGraph.numVertices};

    int threadsPerBlock = 256;
    int blocksPerGrid = (hostGraph.numVertices + threadsPerBlock - 1) / threadsPerBlock;

    int currLevel = 1;
    int hostNewVertexVisited = 1;

    // we iterate over the levels as long as any vertex reports finding a new unvisited neighbour
    while (hostNewVertexVisited != 0) {
        hostNewVertexVisited = 0;
        cudaMemcpy(d_newVertexVisited, &hostNewVertexVisited, sizeof(int), cudaMemcpyHostToDevice);

        bsf_push_vertex_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(deviceGraph, d_levels, d_newVertexVisited,
                                                                           currLevel);

        cudaDeviceSynchronize();

        cudaMemcpy(&hostNewVertexVisited, d_newVertexVisited, sizeof(int), cudaMemcpyDeviceToHost);
        currLevel++;
    }

    cudaMemcpy(hostLevels, d_levels, vertexSize, cudaMemcpyDeviceToHost);

    cudaFree(d_srcPtrs);
    cudaFree(d_dst);
    cudaFree(d_values);
    cudaFree(d_levels);
    cudaFree(d_newVertexVisited);

    return hostLevels;
}

int* bfsParallelPullVertexCentric(const CSCGraph& hostGraph, int startingNode) {
    // Initialize host levels array
    int* hostLevels = (int*)malloc(sizeof(int) * hostGraph.numVertices);
    for (int i = 0; i < hostGraph.numVertices; i++) {
        hostLevels[i] = -1;
    }
    hostLevels[startingNode] = 0;

    // Calculate size needed for graph arrays
    size_t dstPtrsSize = sizeof(int) * (hostGraph.numVertices + 1);
    size_t srcSize =
        sizeof(int) * hostGraph.dstPtrs[hostGraph.numVertices];  // look where the last dstPtrs is poinintg - than we
                                                                 // know how many scr elements there are
    size_t vertexSize = sizeof(int) * hostGraph.numVertices;

    // Allocate device memory
    int *d_dstPtrs, *d_src, *d_values, *d_levels, *d_newVertexVisited;
    cudaMalloc(&d_dstPtrs, dstPtrsSize);
    cudaMalloc(&d_src, srcSize);
    cudaMalloc(&d_values, srcSize);
    cudaMalloc(&d_levels, vertexSize);
    cudaMalloc(&d_newVertexVisited, sizeof(int));

    // Copy values to cuda
    cudaMemcpy(d_dstPtrs, hostGraph.dstPtrs, dstPtrsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_src, hostGraph.src, srcSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, hostGraph.values, srcSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_levels, hostLevels, vertexSize, cudaMemcpyHostToDevice);

    CSCGraph deviceGraph = {
        .dstPtrs = d_dstPtrs, .src = d_src, .values = d_values, .numVertices = hostGraph.numVertices};

    int threadsPerBlock = 256;
    int blocksPerGrid = (hostGraph.numVertices + threadsPerBlock - 1) / threadsPerBlock;

    int currLevel = 1;
    int hostNewVertexVisited = 1;

    // we iterate over the levels as long as any vertex reports finding a new unvisited neighbour
    while (hostNewVertexVisited != 0) {
        hostNewVertexVisited = 0;
        cudaMemcpy(d_newVertexVisited, &hostNewVertexVisited, sizeof(int), cudaMemcpyHostToDevice);

        bsf_pull_vertex_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(deviceGraph, d_levels, d_newVertexVisited,
                                                                           currLevel);

        cudaDeviceSynchronize();

        cudaMemcpy(&hostNewVertexVisited, d_newVertexVisited, sizeof(int), cudaMemcpyDeviceToHost);
        currLevel++;
    }

    cudaMemcpy(hostLevels, d_levels, vertexSize, cudaMemcpyDeviceToHost);

    cudaFree(d_dstPtrs);
    cudaFree(d_src);
    cudaFree(d_values);
    cudaFree(d_levels);
    cudaFree(d_newVertexVisited);

    return hostLevels;
}

int* bfsParallelEdgeCentric(const COOGraph& hostGraph, int startingNode) {
    // Initialize host levels array
    int* hostLevels = (int*)malloc(sizeof(int) * hostGraph.numVertices);
    for (int i = 0; i < hostGraph.numVertices; i++) {
        hostLevels[i] = -1;
    }
    hostLevels[startingNode] = 0;

    // Calculate size needed for graph arrays
    size_t edgeArraysSize = sizeof(int) * hostGraph.numEdges;
    size_t vertexSize = sizeof(int) * hostGraph.numVertices;

    // Allocate device memory
    int *d_src, *d_dst, *d_values, *d_levels, *d_newVertexVisited;
    cudaMalloc(&d_src, edgeArraysSize);
    cudaMalloc(&d_dst, edgeArraysSize);
    cudaMalloc(&d_values, edgeArraysSize);
    cudaMalloc(&d_levels, vertexSize);
    cudaMalloc(&d_newVertexVisited, sizeof(int));

    // Copy values to cuda
    cudaMemcpy(d_src, hostGraph.scr, edgeArraysSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, hostGraph.dst, edgeArraysSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, hostGraph.values, edgeArraysSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_levels, hostLevels, vertexSize, cudaMemcpyHostToDevice);

    COOGraph deviceGraph = {.scr = d_src,
                            .dst = d_dst,
                            .values = d_values,
                            .numEdges = hostGraph.numEdges,
                            .numVertices = hostGraph.numVertices};

    int threadsPerBlock = 256;
    int blocksPerGrid = (hostGraph.numEdges + threadsPerBlock - 1) / threadsPerBlock;

    int currLevel = 1;
    int hostNewVertexVisited = 1;

    // we iterate over the levels as long as any vertex reports finding a new unvisited neighbour
    while (hostNewVertexVisited != 0) {
        hostNewVertexVisited = 0;
        cudaMemcpy(d_newVertexVisited, &hostNewVertexVisited, sizeof(int), cudaMemcpyHostToDevice);

        bsf_edge_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(deviceGraph, d_levels, d_newVertexVisited,
                                                                    currLevel);

        cudaDeviceSynchronize();

        cudaMemcpy(&hostNewVertexVisited, d_newVertexVisited, sizeof(int), cudaMemcpyDeviceToHost);
        currLevel++;
    }

    cudaMemcpy(hostLevels, d_levels, vertexSize, cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_values);
    cudaFree(d_levels);
    cudaFree(d_newVertexVisited);

    return hostLevels;
}

int* bfsParallelFrontierVertexCentric(const CSRGraph& hostGraph, int startingNode) {
    // Initialize host levels array
    int* hostLevels = (int*)malloc(sizeof(int) * hostGraph.numVertices);
    for (int i = 0; i < hostGraph.numVertices; i++) {
        hostLevels[i] = -1;
    }
    hostLevels[startingNode] = 0;

    // Initialize the first frontier hosting just the starting node
    int* hostPrevFrontier = (int*)malloc(sizeof(int) * hostGraph.numVertices);
    hostPrevFrontier[0] = startingNode;
    int hostNumPrevFrontier = 1;

    // Allocate memory for the current frontier
    int* hostCurrFrontier = (int*)malloc(sizeof(int) * hostGraph.numVertices);
    int hostNumCurrFrontier = 0;

    // Calculate size needed for graph arrays
    size_t scrPtrsSize = sizeof(int) * (hostGraph.numVertices + 1);
    size_t dstSize = sizeof(int) * hostGraph.srcPtrs[hostGraph.numVertices];
    size_t vertexSize = sizeof(int) * hostGraph.numVertices;

    // Allocate device memory
    int *d_srcPtrs, *d_dst, *d_values, *d_levels;
    int *d_prevFrontier, *d_currFrontier, *d_numCurrFrontier;

    cudaMalloc(&d_srcPtrs, scrPtrsSize);
    cudaMalloc(&d_dst, dstSize);
    cudaMalloc(&d_values, dstSize);
    cudaMalloc(&d_levels, vertexSize);
    cudaMalloc(&d_prevFrontier, vertexSize);
    cudaMalloc(&d_currFrontier, vertexSize);
    cudaMalloc(&d_numCurrFrontier, sizeof(int));

    // Copy graph structure to device
    cudaMemcpy(d_srcPtrs, hostGraph.srcPtrs, scrPtrsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, hostGraph.dst, dstSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, hostGraph.values, dstSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_levels, hostLevels, vertexSize, cudaMemcpyHostToDevice);

    CSRGraph deviceGraph = {
        .srcPtrs = d_srcPtrs, .dst = d_dst, .values = d_values, .numVertices = hostGraph.numVertices};

    int currLevel = 1;

    // Continue BFS as long as there are vertices in the frontier
    while (hostNumPrevFrontier > 0) {
        // Copy frontier data to device
        cudaMemcpy(d_prevFrontier, hostPrevFrontier, sizeof(int) * hostNumPrevFrontier, cudaMemcpyHostToDevice);

        // Reset the current frontier counter
        hostNumCurrFrontier = 0;
        cudaMemcpy(d_numCurrFrontier, &hostNumCurrFrontier, sizeof(int), cudaMemcpyHostToDevice);

        // Calculate grid dimensions based on frontier size
        int threadsPerBlock = 256;
        int blocksPerGrid = (hostNumPrevFrontier + threadsPerBlock - 1) / threadsPerBlock;

        // Launch kernel to process the frontier
        bsf_frontier_vertex_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            deviceGraph, d_levels, d_prevFrontier, d_currFrontier, hostNumPrevFrontier, d_numCurrFrontier, currLevel);

        cudaDeviceSynchronize();

        // Get the size of the new frontier
        cudaMemcpy(&hostNumCurrFrontier, d_numCurrFrontier, sizeof(int), cudaMemcpyDeviceToHost);

        // Check if the new frontier is empty
        if (hostNumCurrFrontier > 0) {
            // Copy the new frontier back to host
            cudaMemcpy(hostCurrFrontier, d_currFrontier, sizeof(int) * hostNumCurrFrontier, cudaMemcpyDeviceToHost);

            // Swap frontiers for next iteration
            int* tempFrontier = hostPrevFrontier;
            hostPrevFrontier = hostCurrFrontier;
            hostCurrFrontier = tempFrontier;

            hostNumPrevFrontier = hostNumCurrFrontier;
        } else {
            // No more vertices to explore
            hostNumPrevFrontier = 0;
        }

        currLevel++;
    }

    // Copy final levels back to host
    cudaMemcpy(hostLevels, d_levels, vertexSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_srcPtrs);
    cudaFree(d_dst);
    cudaFree(d_values);
    cudaFree(d_levels);
    cudaFree(d_prevFrontier);
    cudaFree(d_currFrontier);
    cudaFree(d_numCurrFrontier);

    // Free host frontier memory
    free(hostPrevFrontier);
    free(hostCurrFrontier);

    return hostLevels;
}

// NEW DEVICE-BASED IMPLEMENTATIONS

int* bfsParallelPushVertexCentricDevice(const CSRGraph& deviceGraph, int startingNode) {
    // Create device levels array
    int* d_levels = allocateAndInitLevelsOnDevice(deviceGraph.numVertices, startingNode);

    // Allocate new vertex visited flag
    int* d_newVertexVisited;
    cudaMalloc(&d_newVertexVisited, sizeof(int));

    int threadsPerBlock = 256;
    int blocksPerGrid = (deviceGraph.numVertices + threadsPerBlock - 1) / threadsPerBlock;

    int currLevel = 1;
    int hostNewVertexVisited = 1;

    // BFS iteration loop
    while (hostNewVertexVisited != 0) {
        hostNewVertexVisited = 0;
        cudaMemcpy(d_newVertexVisited, &hostNewVertexVisited, sizeof(int), cudaMemcpyHostToDevice);

        bsf_push_vertex_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(deviceGraph, d_levels, d_newVertexVisited,
                                                                           currLevel);

        cudaDeviceSynchronize();

        cudaMemcpy(&hostNewVertexVisited, d_newVertexVisited, sizeof(int), cudaMemcpyDeviceToHost);
        currLevel++;
    }

    // Copy results to host
    int* hostLevels = copyLevelsToHost(d_levels, deviceGraph.numVertices);

    // Free device memory
    cudaFree(d_levels);
    cudaFree(d_newVertexVisited);

    return hostLevels;
}

int* bfsParallelPullVertexCentricDevice(const CSCGraph& deviceGraph, int startingNode) {
    // Create device levels array
    int* d_levels = allocateAndInitLevelsOnDevice(deviceGraph.numVertices, startingNode);

    // Allocate new vertex visited flag
    int* d_newVertexVisited;
    cudaMalloc(&d_newVertexVisited, sizeof(int));

    int threadsPerBlock = 256;
    int blocksPerGrid = (deviceGraph.numVertices + threadsPerBlock - 1) / threadsPerBlock;

    int currLevel = 1;
    int hostNewVertexVisited = 1;

    // BFS iteration loop
    while (hostNewVertexVisited != 0) {
        hostNewVertexVisited = 0;
        cudaMemcpy(d_newVertexVisited, &hostNewVertexVisited, sizeof(int), cudaMemcpyHostToDevice);

        bsf_pull_vertex_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(deviceGraph, d_levels, d_newVertexVisited,
                                                                           currLevel);

        cudaDeviceSynchronize();

        cudaMemcpy(&hostNewVertexVisited, d_newVertexVisited, sizeof(int), cudaMemcpyDeviceToHost);
        currLevel++;
    }

    // Copy results to host
    int* hostLevels = copyLevelsToHost(d_levels, deviceGraph.numVertices);

    // Free device memory
    cudaFree(d_levels);
    cudaFree(d_newVertexVisited);

    return hostLevels;
}

int* bfsParallelEdgeCentricDevice(const COOGraph& deviceGraph, int startingNode) {
    // Create device levels array
    int* d_levels = allocateAndInitLevelsOnDevice(deviceGraph.numVertices, startingNode);

    // Allocate new vertex visited flag
    int* d_newVertexVisited;
    cudaMalloc(&d_newVertexVisited, sizeof(int));

    int threadsPerBlock = 256;
    int blocksPerGrid = (deviceGraph.numEdges + threadsPerBlock - 1) / threadsPerBlock;

    int currLevel = 1;
    int hostNewVertexVisited = 1;

    // BFS iteration loop
    while (hostNewVertexVisited != 0) {
        hostNewVertexVisited = 0;
        cudaMemcpy(d_newVertexVisited, &hostNewVertexVisited, sizeof(int), cudaMemcpyHostToDevice);

        bsf_edge_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(deviceGraph, d_levels, d_newVertexVisited,
                                                                    currLevel);

        cudaDeviceSynchronize();

        cudaMemcpy(&hostNewVertexVisited, d_newVertexVisited, sizeof(int), cudaMemcpyDeviceToHost);
        currLevel++;
    }

    // Copy results to host
    int* hostLevels = copyLevelsToHost(d_levels, deviceGraph.numVertices);

    // Free device memory
    cudaFree(d_levels);
    cudaFree(d_newVertexVisited);

    return hostLevels;
}

int* bfsParallelFrontierVertexCentricDevice(const CSRGraph& deviceGraph, int startingNode) {
    // Create device levels array
    int* d_levels = allocateAndInitLevelsOnDevice(deviceGraph.numVertices, startingNode);

    // Initialize the first frontier with the starting node
    int* hostPrevFrontier = (int*)malloc(sizeof(int) * deviceGraph.numVertices);
    hostPrevFrontier[0] = startingNode;
    int hostNumPrevFrontier = 1;

    // Allocate memory for the current frontier
    int* hostCurrFrontier = (int*)malloc(sizeof(int) * deviceGraph.numVertices);
    int hostNumCurrFrontier = 0;

    // Allocate device memory for frontiers
    int *d_prevFrontier, *d_currFrontier, *d_numCurrFrontier;
    cudaMalloc(&d_prevFrontier, sizeof(int) * deviceGraph.numVertices);
    cudaMalloc(&d_currFrontier, sizeof(int) * deviceGraph.numVertices);
    cudaMalloc(&d_numCurrFrontier, sizeof(int));

    int currLevel = 1;

    // Continue BFS as long as there are vertices in the frontier
    while (hostNumPrevFrontier > 0) {
        // Copy frontier data to device
        cudaMemcpy(d_prevFrontier, hostPrevFrontier, sizeof(int) * hostNumPrevFrontier, cudaMemcpyHostToDevice);

        // Reset the current frontier counter
        hostNumCurrFrontier = 0;
        cudaMemcpy(d_numCurrFrontier, &hostNumCurrFrontier, sizeof(int), cudaMemcpyHostToDevice);

        // Calculate grid dimensions based on frontier size
        int threadsPerBlock = 256;
        int blocksPerGrid = (hostNumPrevFrontier + threadsPerBlock - 1) / threadsPerBlock;

        // Launch kernel to process the frontier
        // bsf_frontier_vertex_centric_kernel
        bsf_frontier_vertex_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            deviceGraph, d_levels, d_prevFrontier, d_currFrontier, hostNumPrevFrontier, d_numCurrFrontier, currLevel);

        cudaDeviceSynchronize();

        // Get the size of the new frontier
        cudaMemcpy(&hostNumCurrFrontier, d_numCurrFrontier, sizeof(int), cudaMemcpyDeviceToHost);

        // Check if the new frontier is empty
        if (hostNumCurrFrontier > 0) {
            // Copy the new frontier back to host
            cudaMemcpy(hostCurrFrontier, d_currFrontier, sizeof(int) * hostNumCurrFrontier, cudaMemcpyDeviceToHost);

            // Swap frontiers for next iteration
            int* tempFrontier = hostPrevFrontier;
            hostPrevFrontier = hostCurrFrontier;
            hostCurrFrontier = tempFrontier;

            hostNumPrevFrontier = hostNumCurrFrontier;
        } else {
            // No more vertices to explore
            hostNumPrevFrontier = 0;
        }

        currLevel++;
    }

    // Copy results to host
    int* hostLevels = copyLevelsToHost(d_levels, deviceGraph.numVertices);

    // Free device memory
    cudaFree(d_levels);
    cudaFree(d_prevFrontier);
    cudaFree(d_currFrontier);
    cudaFree(d_numCurrFrontier);

    // Free host frontier memory
    free(hostPrevFrontier);
    free(hostCurrFrontier);

    return hostLevels;
}

// Direction-optimized BFS implementation that switches between push and pull approaches
int* bfsDirectionOptimized(const CSRGraph& hostCSRGraph, const CSCGraph& hostCSCGraph, int startingNode, float alpha) {
    // Allocate CSR and CSC on device
    CSRGraph deviceCSRGraph = allocateCSRGraphOnDevice(hostCSRGraph);
    CSCGraph deviceCSCGraph = allocateCSCGraphOnDevice(hostCSCGraph);

    // Call the device implementation
    int* result = bfsDirectionOptimizedDevice(deviceCSRGraph, deviceCSCGraph, startingNode, alpha);

    // Free device memory
    freeCSRGraphOnDevice(&deviceCSRGraph);
    freeCSCGraphOnDevice(&deviceCSCGraph);

    return result;
}

// Device version of the direction-optimized BFS
int* bfsDirectionOptimizedDevice(const CSRGraph& deviceCSRGraph, const CSCGraph& deviceCSCGraph, int startingNode,
                                 float alpha) {
    // Initialize host levels array
    int* hostLevels = (int*)malloc(sizeof(int) * deviceCSRGraph.numVertices);
    for (int i = 0; i < deviceCSRGraph.numVertices; i++) {
        hostLevels[i] = -1;
    }
    hostLevels[startingNode] = 0;

    // Calculate size needed for vertex array
    size_t vertexSize = sizeof(int) * deviceCSRGraph.numVertices;

    // Allocate device memory for levels and flag
    int *d_levels, *d_newVertexVisited;
    cudaMalloc(&d_levels, vertexSize);
    cudaMalloc(&d_newVertexVisited, sizeof(int));

    // Copy initial levels to device
    cudaMemcpy(d_levels, hostLevels, vertexSize, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (deviceCSRGraph.numVertices + threadsPerBlock - 1) / threadsPerBlock;

    int currLevel = 1;
    int hostNewVertexVisited = 1;

    // Track visited vertices for determining when to switch
    int totalVertices = deviceCSRGraph.numVertices;
    int visitedVertices = 1;  // Starting with just the root
    bool usingPush = true;    // Start with push strategy

    // we iterate over the levels as long as any vertex reports finding a new unvisited neighbour
    while (hostNewVertexVisited != 0) {
        hostNewVertexVisited = 0;
        cudaMemcpy(d_newVertexVisited, &hostNewVertexVisited, sizeof(int), cudaMemcpyHostToDevice);

        // Decide whether to use push or pull based on the fraction of vertices visited
        float visitedFraction = (float)visitedVertices / totalVertices;
        if (usingPush && visitedFraction > alpha) {
            // Switch to pull strategy if we've visited more than alpha fraction of vertices
            usingPush = false;
        }

        if (usingPush) {
            // Use push strategy
            bsf_push_vertex_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(deviceCSRGraph, d_levels,
                                                                               d_newVertexVisited, currLevel);
        } else {
            // Use pull strategy
            bsf_pull_vertex_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(deviceCSCGraph, d_levels,
                                                                               d_newVertexVisited, currLevel);
        }

        cudaDeviceSynchronize();
        cudaMemcpy(&hostNewVertexVisited, d_newVertexVisited, sizeof(int), cudaMemcpyDeviceToHost);

        // If new vertices were visited, update our count
        if (hostNewVertexVisited) {
            // For simplicity, we'll count the visited vertices by copying back and checking
            // In a more optimized implementation, we'd track this on the device
            cudaMemcpy(hostLevels, d_levels, vertexSize, cudaMemcpyDeviceToHost);
            int newVisitedCount = 0;
            for (int i = 0; i < totalVertices; i++) {
                if (hostLevels[i] != -1) {
                    newVisitedCount++;
                }
            }
            visitedVertices = newVisitedCount;
        }

        currLevel++;
    }

    // Copy final results back to host
    cudaMemcpy(hostLevels, d_levels, vertexSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_levels);
    cudaFree(d_newVertexVisited);

    return hostLevels;
}

__global__ void bfs_multi_level_frontier_kernel(CSRGraph graph, int* levels, int* frontier, int* frontierSize,
                                                int* nextFrontier, int* nextFrontierSize, int currLevel) {
    // Shared memory for local frontier management
    __shared__ int localFrontier[LOCAL_FRONTIER_CAPACITY];
    __shared__ int localFrontierSize;
    __shared__ int nextLocalFrontierSize;
    __shared__ bool overflowed;

    // Initialize shared variables
    if (threadIdx.x == 0) {
        localFrontierSize = *frontierSize;
        nextLocalFrontierSize = 0;
        overflowed = false;

        // Copy initial frontier to shared memory
        for (int i = 0; i < localFrontierSize && i < LOCAL_FRONTIER_CAPACITY; i++) {
            localFrontier[i] = frontier[i];
        }
    }

    __syncthreads();

    // Process current frontier vertices
    for (int i = threadIdx.x; i < localFrontierSize; i += blockDim.x) {
        int vertex = localFrontier[i];

        // Explore neighbors
        for (unsigned int edge = graph.srcPtrs[vertex]; edge < graph.srcPtrs[vertex + 1]; edge++) {
            unsigned int neighbor = graph.dst[edge];

            // If neighbor not visited, add to next frontier
            if (atomicCAS(&levels[neighbor], -1, currLevel) == -1) {
                int idx = atomicAdd(&nextLocalFrontierSize, 1);

                // Check if the next frontier exceeds capacity
                if (idx < LOCAL_FRONTIER_CAPACITY) {
                    localFrontier[idx] = neighbor;
                } else {
                    // Mark as overflowed - need to switch to multi-block kernel
                    overflowed = true;

                    // Add to global frontier directly
                    int globalIdx = atomicAdd(nextFrontierSize, 1);
                    nextFrontier[globalIdx] = neighbor;
                }
            }
        }
    }

    __syncthreads();

    // Update for next iteration or return results
    if (threadIdx.x == 0) {
        if (overflowed) {
            // Copy remaining valid local frontier entries to global
            for (int i = 0; i < nextLocalFrontierSize && i < LOCAL_FRONTIER_CAPACITY; i++) {
                int globalIdx = atomicAdd(nextFrontierSize, 1);
                nextFrontier[globalIdx] = localFrontier[i];
            }
            // Signal that we need to switch to multi-block
            *frontierSize = 0;
        } else {
            // We can continue with single-block
            *frontierSize = nextLocalFrontierSize;

            // Copy the next frontier for the host
            for (int i = 0; i < nextLocalFrontierSize; i++) {
                frontier[i] = localFrontier[i];
            }
        }
    }
}

// Device implementation of optimized frontier-based BFS
int* bfsParallelFrontierVertexCentricOptimizedDevice(const CSRGraph& deviceGraph, int startingNode) {
    // Create device levels array
    int* d_levels = allocateAndInitLevelsOnDevice(deviceGraph.numVertices, startingNode);

    // Initialize frontiers
    int* d_currentFrontier;
    int* d_nextFrontier;
    int* d_frontierSize;
    int* d_nextFrontierSize;

    cudaMalloc(&d_currentFrontier, sizeof(int) * deviceGraph.numVertices);
    cudaMalloc(&d_nextFrontier, sizeof(int) * deviceGraph.numVertices);
    cudaMalloc(&d_frontierSize, sizeof(int));
    cudaMalloc(&d_nextFrontierSize, sizeof(int));

    // Initial frontier contains just the starting node
    int hostFrontierSize = 1;
    cudaMemcpy(d_currentFrontier, &startingNode, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontierSize, &hostFrontierSize, sizeof(int), cudaMemcpyHostToDevice);

    int currentLevel = 1;
    bool useSingleBlock = true;  // Start with single-block optimization

    // Continue BFS until no more vertices in frontier
    while (hostFrontierSize > 0) {
        if (useSingleBlock && hostFrontierSize <= LOCAL_FRONTIER_CAPACITY / 2) {  // Use half capacity as safety margin
            // Reset next frontier size
            int resetSize = 0;
            cudaMemcpy(d_nextFrontierSize, &resetSize, sizeof(int), cudaMemcpyHostToDevice);

            // Process with small-frontier kernel
            bfs_multi_level_frontier_kernel<<<1, 256>>>(deviceGraph, d_levels, d_currentFrontier, d_frontierSize,
                                                        d_nextFrontier, d_nextFrontierSize, currentLevel);

            // Check if we can continue with single-block
            cudaMemcpy(&hostFrontierSize, d_frontierSize, sizeof(int), cudaMemcpyDeviceToHost);

            if (hostFrontierSize == 0) {
                // We need to switch to multi-block kernel
                int nextSize;
                cudaMemcpy(&nextSize, d_nextFrontierSize, sizeof(int), cudaMemcpyDeviceToHost);

                if (nextSize == 0) {
                    // BFS is complete
                    break;
                }

                // Swap frontiers
                int* temp = d_currentFrontier;
                d_currentFrontier = d_nextFrontier;
                d_nextFrontier = temp;

                hostFrontierSize = nextSize;
                cudaMemcpy(d_frontierSize, &hostFrontierSize, sizeof(int), cudaMemcpyHostToDevice);

                useSingleBlock = false;
            }
        } else {
            // Reset next frontier size
            int resetSize = 0;
            cudaMemcpy(d_nextFrontierSize, &resetSize, sizeof(int), cudaMemcpyHostToDevice);

            // Process with regular multi-block kernel
            int threadsPerBlock = 256;
            int blocksPerGrid = (hostFrontierSize + threadsPerBlock - 1) / threadsPerBlock;

            bsf_frontier_vertex_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                deviceGraph, d_levels, d_currentFrontier, d_nextFrontier, hostFrontierSize, d_nextFrontierSize,
                currentLevel);

            // Get next frontier size
            cudaMemcpy(&hostFrontierSize, d_nextFrontierSize, sizeof(int), cudaMemcpyDeviceToHost);

            if (hostFrontierSize == 0) {
                // BFS is complete
                break;
            }

            // Swap frontiers
            int* temp = d_currentFrontier;
            d_currentFrontier = d_nextFrontier;
            d_nextFrontier = temp;

            // Check if we can go back to single-block mode
            if (hostFrontierSize <= LOCAL_FRONTIER_CAPACITY / 2) {
                useSingleBlock = true;
            }
        }

        // Always increment level after processing a frontier
        currentLevel++;
    }

    // Copy results to host
    int* hostLevels = copyLevelsToHost(d_levels, deviceGraph.numVertices);

    // Clean up
    cudaFree(d_levels);
    cudaFree(d_currentFrontier);
    cudaFree(d_nextFrontier);
    cudaFree(d_frontierSize);
    cudaFree(d_nextFrontierSize);

    return hostLevels;
}

// Host wrapper implementation
int* bfsParallelFrontierVertexCentricOptimized(const CSRGraph& hostGraph, int startingNode) {
    // Allocate graph on device
    CSRGraph deviceGraph;

    // Allocate memory
    size_t srcPtrsSize = sizeof(int) * (hostGraph.numVertices + 1);
    size_t dstSize = sizeof(int) * hostGraph.srcPtrs[hostGraph.numVertices];

    cudaMalloc(&deviceGraph.srcPtrs, srcPtrsSize);
    cudaMalloc(&deviceGraph.dst, dstSize);
    cudaMalloc(&deviceGraph.values, dstSize);

    // Copy data to device
    cudaMemcpy(deviceGraph.srcPtrs, hostGraph.srcPtrs, srcPtrsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceGraph.dst, hostGraph.dst, dstSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceGraph.values, hostGraph.values, dstSize, cudaMemcpyHostToDevice);

    deviceGraph.numVertices = hostGraph.numVertices;

    // Run BFS
    int* result = bfsParallelFrontierVertexCentricOptimizedDevice(deviceGraph, startingNode);

    // Clean up
    cudaFree(deviceGraph.srcPtrs);
    cudaFree(deviceGraph.dst);
    cudaFree(deviceGraph.values);

    return result;
}
