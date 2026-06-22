// Chapter: 不同C++结构的效率
// Example 7.5. Set flush-to-zero mode (SSE):

#include <xmmintrin.h>
_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
