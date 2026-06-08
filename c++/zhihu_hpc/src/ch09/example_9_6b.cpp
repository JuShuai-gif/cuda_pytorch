// Chapter: 优化内存访问
// Example 9.6b.
#include "xmmintrin.h" // header for intrinsic functions
// This function stores a double without loading a cache line:
static inline void StoreNTD(double * dest, double const & source)
{
    _mm_stream_pi((__m64*)dest, *(__m64*)&source); // MOVNTQ
    _mm_empty(); // EMMS
}
const int SIZE = 512; // number of rows and columns in matrix
// function to transpose and copy matrix
void TransposeCopy(double a[SIZE][SIZE], double b[SIZE][SIZE])
{
    int r, c;
    for (r = 0; r < SIZE; r++)
    {
        for (c = 0; c < SIZE; c++)
        {
            StoreNTD(&a[c][r], b[r][c]);
        }
    }
}
