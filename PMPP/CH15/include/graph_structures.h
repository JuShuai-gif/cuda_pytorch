#ifndef CHAPTER_15_CODE_INCLUDE_GRAPH_STRUCTURES_H_
#define CHAPTER_15_CODE_INCLUDE_GRAPH_STRUCTURES_H_
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
#endif  // CHAPTER_15_CODE_INCLUDE_GRAPH_STRUCTURES_H_
