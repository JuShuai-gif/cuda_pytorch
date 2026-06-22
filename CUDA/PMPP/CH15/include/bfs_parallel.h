#ifndef CHAPTER_15_CODE_INCLUDE_BFS_PARALLEL_H_
#define CHAPTER_15_CODE_INCLUDE_BFS_PARALLEL_H_
#include "graph_structures.h"
// Functions taking graph on the host
int* bfsParallelPushVertexCentric(const CSRGraph& hostGraph, int startingNode);
int* bfsParallelPullVertexCentric(const CSCGraph& hostGraph, int startingNode);
int* bfsParallelEdgeCentric(const COOGraph& hostGraph, int startingNode);
int* bfsParallelFrontierVertexCentric(const CSRGraph& hostGraph, int startingNode);
int* bfsParallelFrontierVertexCentricOptimized(const CSRGraph& hostGraph, int startingNode);
int* bfsDirectionOptimized(const CSRGraph& hostCSRGraph, const CSCGraph& hostCSCGraph, int startingNode,
                           float alpha = 0.1);
// Functions taking graph already on the device
int* bfsParallelPushVertexCentricDevice(const CSRGraph& deviceGraph, int startingNode);
int* bfsParallelPullVertexCentricDevice(const CSCGraph& deviceGraph, int startingNode);
int* bfsParallelEdgeCentricDevice(const COOGraph& deviceGraph, int startingNode);
int* bfsParallelFrontierVertexCentricDevice(const CSRGraph& deviceGraph, int startingNode);
int* bfsParallelFrontierVertexCentricOptimizedDevice(const CSRGraph& deviceGraph, int startingNode);
int* bfsDirectionOptimizedDevice(const CSRGraph& deviceCSRGraph, const CSCGraph& deviceCSCGraph, int startingNode,
                                 float alpha = 0.1);
#endif  // CHAPTER_15_CODE_INCLUDE_BFS_PARALLEL_H_
