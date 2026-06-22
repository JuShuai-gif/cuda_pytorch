// Chapter: 不同C++结构的效率
// Example 7.6. Set flush-to-zero and denormals-are-zero mode (SSE2):

#include <xmmintrin.h>
_mm_setcsr(_mm_getcsr() | 0x8040);
