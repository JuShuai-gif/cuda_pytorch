#include <stdio.h>

#include "../include/utils.h"

bool compareBFSResults(int* sequentialLevels, int* parallelLevels, int numVertices, bool printDetails) {
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
        // printf("All BFS results match between sequential and parallel implementations.\n");
    }

    return resultsMatch;
}
