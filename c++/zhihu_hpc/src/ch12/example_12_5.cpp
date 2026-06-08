// Chapter: 使用向量操作
// Example 12.5. Aligned arrays

// Define macro for aligning data
#ifdef _MSC_VER // If Microsoft compiler
#define Alignd(X) __declspec(align(16)) X
#else // Gnu compiler, etc.
#define Alignd(X) X __attribute__((aligned(16)))
#endif
const int size = 256; // Array size
Alignd ( short int aa[size] ); // Make three aligned arrays
Alignd ( short int bb[size] );
Alignd ( short int cc[size] );
// Function to load aligned integer vector from array
static inline __m128i LoadVectorA(void const * p)
{
    return _mm_load_si128((__m128i const*)p);
}
// Function to store aligned integer vector into array
static inline void StoreVectorA(void * d, __m128i const & x)
{
    _mm_store_si128((__m128i *)d, x);
}
