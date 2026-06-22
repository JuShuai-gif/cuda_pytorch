#ifndef CHAPTER_15_CODE_INCLUDE_DEVICE_MEMORY_H_
#define CHAPTER_15_CODE_INCLUDE_DEVICE_MEMORY_H_
#include <cuda_runtime.h>

#include "graph_structures.h"
// CSR Graph memory management
CSRGraph allocateCSRGraphOnDevice(const CSRGraph& hostGraph);
void freeCSRGraphOnDevice(CSRGraph* deviceGraph);
// CSC Graph memory management
CSCGraph allocateCSCGraphOnDevice(const CSCGraph& hostGraph);
void freeCSCGraphOnDevice(CSCGraph* deviceGraph);
// COO Graph memory management
COOGraph allocateCOOGraphOnDevice(const COOGraph& hostGraph);
void freeCOOGraphOnDevice(COOGraph* deviceGraph);
// BFS level arrays management
int* allocateAndInitLevelsOnDevice(int numVertices, int startingNode);
int* copyLevelsToHost(int* deviceLevels, int numVertices);
void freeLevelsOnDevice(int* deviceLevels);
#endif  // CHAPTER_15_CODE_INCLUDE_DEVICE_MEMORY_H_
