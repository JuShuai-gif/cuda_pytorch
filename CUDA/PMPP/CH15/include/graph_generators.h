#ifndef CHAPTER_15_CODE_INCLUDE_GRAPH_GENERATORS_H_
#define CHAPTER_15_CODE_INCLUDE_GRAPH_GENERATORS_H_
#include "graph_structures.h"
COOGraph generateScaleFreeGraphCOO(int numVertices, int edgesPerNewVertex);
COOGraph generateSmallWorldGraphCOO(int numVertices, int k, float rewireProbability);
#endif  // CHAPTER_15_CODE_INCLUDE_GRAPH_GENERATORS_H_
