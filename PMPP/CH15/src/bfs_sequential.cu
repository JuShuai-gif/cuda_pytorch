#include <stdio.h>
#include <stdlib.h>

#include <queue>
#include <vector>

#include "../include/bfs_sequential.h"
#include "../include/graph_structures.h"

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
