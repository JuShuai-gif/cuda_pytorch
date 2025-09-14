#include <stdio.h>
#include <stdlib.h>

#include <queue>
#include <vector>

struct CSRGraph {
    int* srcPtrs;
    int* dst;
    int* values;
    int numVertices;
};

struct CSCGraph {
    int* dstPtrs;
    int* src;
    int* values;
    int numVertices;
};

struct COOGraph {
    int* scr;
    int* dst;
    int* values;
    int numEdges;
    int numVertices;
};

CSRGraph convertCOOtoCSR(const COOGraph& cooGraph) {
    CSRGraph csrGraph;
    csrGraph.numVertices = cooGraph.numVertices;
    csrGraph.srcPtrs = (int*)malloc(sizeof(int) * (cooGraph.numVertices + 1));
    csrGraph.dst = (int*)malloc(sizeof(int) * cooGraph.numEdges);
    csrGraph.values = (int*)malloc(sizeof(int) * cooGraph.numEdges);

    // Initialize srcPtrs array with zeros
    for (int i = 0; i <= cooGraph.numVertices; i++) {
        csrGraph.srcPtrs[i] = 0;
    }

    // Count occurrences of each source vertex
    for (int i = 0; i < cooGraph.numEdges; i++) {
        csrGraph.srcPtrs[cooGraph.scr[i] + 1]++;
    }

    // Cumulative sum to get row pointers
    for (int i = 1; i <= cooGraph.numVertices; i++) {
        csrGraph.srcPtrs[i] += csrGraph.srcPtrs[i - 1];
    }

    // Copy data from COO to CSR
    int* pos = (int*)malloc(sizeof(int) * cooGraph.numVertices);
    memcpy(pos, csrGraph.srcPtrs, sizeof(int) * cooGraph.numVertices);

    for (int i = 0; i < cooGraph.numEdges; i++) {
        int row = cooGraph.scr[i];
        int idx = pos[row]++;

        csrGraph.dst[idx] = cooGraph.dst[i];
        csrGraph.values[idx] = cooGraph.values[i];
    }

    free(pos);
    return csrGraph;
}

COOGraph convertCSRtoCOO(const CSRGraph& csrGraph) {
    int numVertices = csrGraph.numVertices;
    int numEdges = csrGraph.srcPtrs[numVertices];

    // Allocate memory for COO graph
    COOGraph cooGraph;
    cooGraph.numVertices = numVertices;
    cooGraph.numEdges = numEdges;
    cooGraph.scr = (int*)malloc(sizeof(int) * numEdges);
    cooGraph.dst = (int*)malloc(sizeof(int) * numEdges);
    cooGraph.values = (int*)malloc(sizeof(int) * numEdges);

    // Populate the COO arrays
    int edgeIdx = 0;
    for (int i = 0; i < numVertices; i++) {
        for (int j = csrGraph.srcPtrs[i]; j < csrGraph.srcPtrs[i + 1]; j++) {
            cooGraph.scr[edgeIdx] = i;                      // Source vertex
            cooGraph.dst[edgeIdx] = csrGraph.dst[j];        // Destination vertex
            cooGraph.values[edgeIdx] = csrGraph.values[j];  // Edge value
            edgeIdx++;
        }
    }

    return cooGraph;
}

CSCGraph convertCOOtoCSC(const COOGraph& cooGraph) {
    CSCGraph cscGraph;
    cscGraph.numVertices = cooGraph.numVertices;
    cscGraph.dstPtrs = (int*)malloc(sizeof(int) * (cooGraph.numVertices + 1));
    cscGraph.src = (int*)malloc(sizeof(int) * cooGraph.numEdges);
    cscGraph.values = (int*)malloc(sizeof(int) * cooGraph.numEdges);

    // Initialize dstPtrs array with zeros
    for (int i = 0; i <= cooGraph.numVertices; i++) {
        cscGraph.dstPtrs[i] = 0;
    }

    // Count occurrences of each destination (column)
    for (int i = 0; i < cooGraph.numEdges; i++) {
        cscGraph.dstPtrs[cooGraph.dst[i] + 1]++;
    }

    // Cumulative sum to get column pointers
    for (int i = 1; i <= cooGraph.numVertices; i++) {
        cscGraph.dstPtrs[i] += cscGraph.dstPtrs[i - 1];
    }

    // Copy data from COO to CSC
    int* pos = (int*)malloc(sizeof(int) * cooGraph.numVertices);
    memcpy(pos, cscGraph.dstPtrs, sizeof(int) * cooGraph.numVertices);

    for (int i = 0; i < cooGraph.numEdges; i++) {
        int col = cooGraph.dst[i];
        int idx = pos[col]++;

        cscGraph.src[idx] = cooGraph.scr[i];
        cscGraph.values[idx] = cooGraph.values[i];
    }

    free(pos);
    return cscGraph;
}

CSCGraph convertCSRtoCSC(const CSRGraph& csrGraph) {
    COOGraph cooGraph = convertCSRtoCOO(csrGraph);
    CSCGraph cscGraph = convertCOOtoCSC(cooGraph);

    free(cooGraph.scr);
    free(cooGraph.dst);
    free(cooGraph.values);

    return cscGraph;
}

// Generate a scale-free graph using Barabási–Albert model
COOGraph generateScaleFreeGraphCOO(int numVertices, int edgesPerNewVertex) {
    if (numVertices < 3) {
        numVertices = 3;  // Minimum vertices
    }
    if (edgesPerNewVertex >= numVertices) {
        edgesPerNewVertex = numVertices - 1;
    }
    if (edgesPerNewVertex < 1) {
        edgesPerNewVertex = 1;
    }

    // Initialize with a small fully connected network of m0 vertices
    int m0 = edgesPerNewVertex + 1;
    if (m0 >= numVertices) {
        m0 = numVertices - 1;
    }

    // Calculate maximum possible edges - be generous to avoid reallocation
    // Each new vertex adds edgesPerNewVertex*2 edges (bidirectional)
    int maxPossibleEdges = m0 * (m0 - 1) + (numVertices - m0) * edgesPerNewVertex * 2;

    // Allocate COO graph with maximum possible size
    COOGraph graph;
    graph.numVertices = numVertices;
    graph.scr = (int*)malloc(sizeof(int) * maxPossibleEdges);
    graph.dst = (int*)malloc(sizeof(int) * maxPossibleEdges);
    graph.values = (int*)malloc(sizeof(int) * maxPossibleEdges);

    // Set random seed
    srand(time(NULL));

    // Initialize edge counter
    int edgeIdx = 0;

    // Create initial complete graph
    for (int i = 0; i < m0; i++) {
        for (int j = 0; j < m0; j++) {
            if (i != j) {
                graph.scr[edgeIdx] = i;
                graph.dst[edgeIdx] = j;
                graph.values[edgeIdx] = 1;
                edgeIdx++;
            }
        }
    }

    // Track the degree of each vertex for preferential attachment
    int* degree = (int*)calloc(numVertices, sizeof(int));  // Initialize with zeros

    // Update degrees for initial vertices
    for (int i = 0; i < edgeIdx; i++) {
        degree[graph.scr[i]]++;
        degree[graph.dst[i]]++;
    }

    // Add remaining vertices using preferential attachment
    for (int newVertex = m0; newVertex < numVertices; newVertex++) {
        int edgesAdded = 0;

        // Create a probability distribution based on degrees
        int totalDegree = 0;
        for (int i = 0; i < newVertex; i++) {
            totalDegree += degree[i];
        }

        // If total degree is 0, initialize with uniform probability
        if (totalDegree == 0) {
            for (int i = 0; i < newVertex; i++) {
                degree[i] = 1;
            }
            totalDegree = newVertex;
        }

        // Keep track of connections to avoid duplicates
        bool* connected = (bool*)calloc(newVertex, sizeof(bool));

        // Add edges to existing vertices based on preferential attachment
        while (edgesAdded < edgesPerNewVertex && edgesAdded < newVertex) {
            // Choose target based on degree
            int target = -1;
            int randomValue = rand() % totalDegree;
            int cumulativeProbability = 0;

            for (int i = 0; i < newVertex; i++) {
                if (connected[i]) {
                    continue;  // Skip already connected vertices
                }

                cumulativeProbability += degree[i];
                if (randomValue < cumulativeProbability) {
                    target = i;
                    break;
                }
            }

            // If no target found using probability, pick randomly from unconnected
            if (target == -1) {
                // Find unconnected vertices
                std::vector<int> unconnected;
                for (int i = 0; i < newVertex; i++) {
                    if (!connected[i]) {
                        unconnected.push_back(i);
                    }
                }

                if (!unconnected.empty()) {
                    target = unconnected[rand() % unconnected.size()];
                } else {
                    break;  // No more vertices to connect to
                }
            }

            // Add edge if valid target found
            if (target != -1 && !connected[target]) {
                // Safety check for array bounds
                if (edgeIdx + 2 > maxPossibleEdges) {
                    printf("Error: Edge index exceeds maximum possible edges.\n");
                    break;
                }

                // Add edge
                graph.scr[edgeIdx] = newVertex;
                graph.dst[edgeIdx] = target;
                graph.values[edgeIdx] = 1;
                edgeIdx++;

                // Add reverse edge (undirected graph)
                graph.scr[edgeIdx] = target;
                graph.dst[edgeIdx] = newVertex;
                graph.values[edgeIdx] = 1;
                edgeIdx++;

                // Update degrees and mark as connected
                degree[newVertex]++;
                degree[target]++;
                connected[target] = true;

                edgesAdded++;
            }
        }

        free(connected);
    }

    // Set final edge count
    graph.numEdges = edgeIdx;

    free(degree);
    return graph;
}

// Generate a small-world graph using Watts-Strogatz model
COOGraph generateSmallWorldGraphCOO(int numVertices, int k, float rewireProbability) {
    // k is mean degree (must be even)
    if (k % 2 != 0) {
        k--;
    }
    if (k >= numVertices) {
        k = numVertices - 1;
    }
    if (k < 2) {
        k = 2;
    }

    // Total edges in the graph
    int totalEdges = numVertices * k / 2;  // Undirected edges

    // Allocate COO graph
    COOGraph graph;
    graph.numVertices = numVertices;
    graph.numEdges = totalEdges;
    graph.scr = (int*)malloc(sizeof(int) * totalEdges * 2);  // *2 for directed edges
    graph.dst = (int*)malloc(sizeof(int) * totalEdges * 2);
    graph.values = (int*)malloc(sizeof(int) * totalEdges * 2);

    // Set random seed
    srand(time(NULL));

    // Create initial ring lattice (store as COO)
    int edgeIdx = 0;
    for (int i = 0; i < numVertices; i++) {
        for (int j = 1; j <= k / 2; j++) {
            int neighbor = (i + j) % numVertices;

            // Add edge
            graph.scr[edgeIdx] = i;
            graph.dst[edgeIdx] = neighbor;
            graph.values[edgeIdx] = 1;
            edgeIdx++;

            // Add reverse edge (undirected graph)
            graph.scr[edgeIdx] = neighbor;
            graph.dst[edgeIdx] = i;
            graph.values[edgeIdx] = 1;
            edgeIdx++;
        }
    }

    // Create a copy of the original edges for rewiring
    int* originalDst = (int*)malloc(sizeof(int) * totalEdges * 2);
    memcpy(originalDst, graph.dst, sizeof(int) * totalEdges * 2);

    // Track connections to avoid duplicates during rewiring
    bool** connections = (bool**)malloc(sizeof(bool*) * numVertices);
    for (int i = 0; i < numVertices; i++) {
        connections[i] = (bool*)calloc(numVertices, sizeof(bool));
    }

    // Initialize connection matrix
    for (int i = 0; i < edgeIdx; i++) {
        int src = graph.scr[i];
        int dst = graph.dst[i];
        connections[src][dst] = true;
    }

    // Rewire edges with probability p (only forward edges to avoid inconsistency)
    for (int i = 0; i < edgeIdx; i += 2) {
        float random = static_cast<float>(rand()) / RAND_MAX;

        if (random < rewireProbability) {
            int src = graph.scr[i];
            int oldDst = graph.dst[i];

            // Try to find a new target that isn't already connected
            int attempts = 0;
            int newDst;
            bool validTarget = false;

            while (!validTarget && attempts < 50) {
                newDst = rand() % numVertices;

                // Avoid self-loops and existing connections
                if (newDst != src && !connections[src][newDst]) {
                    validTarget = true;
                }

                attempts++;
            }

            // If found a valid new target, rewire the edge
            if (validTarget) {
                // Remove old connection
                connections[src][oldDst] = false;
                connections[oldDst][src] = false;

                // Add new connection
                connections[src][newDst] = true;
                connections[newDst][src] = true;

                // Update edge in COO
                graph.dst[i] = newDst;

                // Update reverse edge
                graph.scr[i + 1] = newDst;
                graph.dst[i + 1] = src;
            }
        }
    }

    // Free memory
    for (int i = 0; i < numVertices; i++) {
        free(connections[i]);
    }
    free(connections);
    free(originalDst);

    // Set final edge count
    graph.numEdges = edgeIdx;

    return graph;
}

// BFS returning a pointer to the list of levels for all vertices
int* bfs(const CSRGraph& graph, int startingNode) {
    int* levels = (int*)malloc(sizeof(int) * graph.numVertices);
    std::vector<bool> visited(graph.numVertices, false);

    // set the default level to -1 meaning it is not yet visited
    for (int i = 0; i < graph.numVertices; i++) {
        levels[i] = -1;
    }

    std::queue<int> queue;

    levels[startingNode] = 0;
    visited[startingNode] = true;
    queue.push(startingNode);

    while (!queue.empty()) {
        int vertex = queue.front();
        queue.pop();

        for (int edge = graph.srcPtrs[vertex]; edge < graph.srcPtrs[vertex + 1]; edge++) {
            int neighbour = graph.dst[edge];
            if (!visited[neighbour]) {
                levels[neighbour] = levels[vertex] + 1;
                visited[neighbour] = true;
                queue.push(neighbour);
            }
        }
    }
    return levels;
}

__global__ void bsf_push_vertex_centric_kernel(CSRGraph graph, int* levels, int* newVertexVisitd,
                                               unsigned int currLevel) {
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex < graph.numVertices) {
        if (levels[vertex] == currLevel - 1) {
            // iterate over all the vertices at the current level
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
                    break;  // if any of the neighbour at the prev level we reached our point
                }
            }
        }
    }
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

    int threadsPerBlock = 256;
    int blocksPerGrid = (hostGraph.numVertices + threadsPerBlock - 1) / threadsPerBlock;

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

bool compareBFSResults(int* sequentialLevels, int* parallelLevels, int numVertices, bool printDetails = false) {
    bool resultsMatch = true;

    if (printDetails) {
        printf("Comparing BFS results:\n");
        printf("%-10s %-15s %-15s %-10s\n", "Vertex", "Sequential", "Parallel", "Match");
        printf("------------------------------------------------\n");
    }

    for (int i = 0; i < numVertices; i++) {
        bool vertexMatch = (sequentialLevels[i] == parallelLevels[i]);

        if (!vertexMatch) {
            resultsMatch = false;

            // Always print mismatches even if printDetails is false
            if (!printDetails) {
                printf("Mismatch at vertex %d: Sequential=%d, Parallel=%d\n", i, sequentialLevels[i],
                       parallelLevels[i]);
            }
        }

        if (printDetails) {
            printf("%-10d %-15d %-15d %-10s\n", i, sequentialLevels[i], parallelLevels[i], vertexMatch ? "✓" : "✗");
        }
    }

    if (printDetails) {
        printf("------------------------------------------------\n");
        printf("Overall result: %s\n", resultsMatch ? "Both implementations match!" : "Implementations differ!");
    } else if (resultsMatch) {
        printf("All BFS results match between sequential and parallel implementations.\n");
    }

    return resultsMatch;
}

int main() {
    // Generate a scale-free graph with 1000 vertices and 3 edges per new vertex
    COOGraph scaleFreeCOO = generateScaleFreeGraphCOO(1000, 3);
    CSRGraph scaleFreeCSR = convertCOOtoCSR(scaleFreeCOO);

    // Generate a small-world graph with 1000 vertices, 4 neighbors, 0.1 rewiring probability
    COOGraph smallWorldCOO = generateSmallWorldGraphCOO(1000, 4, 0.1);
    CSRGraph smallWorldCSR = convertCOOtoCSR(smallWorldCOO);

    // Run BFS on both graph types
    printf("Running BFS on Scale-Free Graph:\n");
    int* seqScaleFree = bfs(scaleFreeCSR, 0);
    int* parScaleFree = bfsParallelEdgeCentric(scaleFreeCOO, 0);
    compareBFSResults(seqScaleFree, parScaleFree, scaleFreeCSR.numVertices, false);

    printf("\nRunning BFS on Small-World Graph:\n");
    int* seqSmallWorld = bfs(smallWorldCSR, 0);
    int* parSmallWorld = bfsParallelEdgeCentric(smallWorldCOO, 0);
    compareBFSResults(seqSmallWorld, parSmallWorld, smallWorldCSR.numVertices, false);

    // Clean up
    free(seqScaleFree);
    free(parScaleFree);
    free(seqSmallWorld);
    free(parSmallWorld);

    // Free graph memory
    free(scaleFreeCOO.scr);
    free(scaleFreeCOO.dst);
    free(scaleFreeCOO.values);
    free(scaleFreeCSR.srcPtrs);
    free(scaleFreeCSR.dst);
    free(scaleFreeCSR.values);

    free(smallWorldCOO.scr);
    free(smallWorldCOO.dst);
    free(smallWorldCOO.values);
    free(smallWorldCSR.srcPtrs);
    free(smallWorldCSR.dst);
    free(smallWorldCSR.values);

    return 0;
}
