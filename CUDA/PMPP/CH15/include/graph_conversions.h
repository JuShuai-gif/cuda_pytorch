#ifndef CHAPTER_15_CODE_INCLUDE_GRAPH_CONVERSIONS_H_
#define CHAPTER_15_CODE_INCLUDE_GRAPH_CONVERSIONS_H_
#include "graph_structures.h"
CSRGraph convertCOOtoCSR(const COOGraph& cooGraph);
COOGraph convertCSRtoCOO(const CSRGraph& csrGraph);
CSCGraph convertCOOtoCSC(const COOGraph& cooGraph);
CSCGraph convertCSRtoCSC(const CSRGraph& csrGraph);
#endif  // CHAPTER_15_CODE_INCLUDE_GRAPH_CONVERSIONS_H_
