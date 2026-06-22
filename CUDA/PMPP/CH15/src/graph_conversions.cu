#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/graph_conversions.h"
#include "../include/graph_structures.h"

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
