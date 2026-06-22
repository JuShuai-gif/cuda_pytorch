#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <vector>

#include "../include/graph_generators.h"
#include "../include/graph_structures.h"

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
