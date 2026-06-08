// Chapter: 具体的优化主题
// Example 14.21. // Only for SSE2 or x64

#include <emmintrin.h>
static inline int lrintf (float const x)
{
    return _mm_cvtss_si32(_mm_load_ss(&x));
}
static inline int lrint (double const x)
{
    return _mm_cvtsd_si32(_mm_load_sd(&x));
}
