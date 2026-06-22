// Chapter: 使用向量操作
// Example 12.4b. Vectorized with SSE2

#include <emmintrin.h> // Define SSE2 intrinsic functions
// Function to load unaligned integer vector from array
static inline __m128i LoadVector(void const * p)
{
    return _mm_loadu_si128((__m128i const*)p);
}
// Function to store unaligned integer vector into array
static inline void StoreVector(void * d, __m128i const & x)
{
    _mm_storeu_si128((__m128i *)d, x);
}
// Branch/loop function vectorized:
void SelectAddMul(short int aa[], short int bb[], short int cc[])、
{
    // Make a vector of (0,0,0,0,0,0,0,0)
    __m128i zero = _mm_set1_epi16(0);
    // Make a vector of (2,2,2,2,2,2,2,2)
    __m128i two = _mm_set1_epi16(2);
    // Roll out loop by eight to fit the eight-element vectors:
    for (int i = 0; i < 256; i += 8)
    {
        // Load eight consecutive elements from bb into vector b:
        __m128i b = LoadVector(bb + i);
        // Load eight consecutive elements from cc into vector c:
        __m128i c = LoadVector(cc + i);
        // Add 2 to each element in vector c
        __m128i c2 = _mm_add_epi16(c, two);
        // Multiply b and c
        __m128i bc = _mm_mullo_epi16 (b, c);
        // Compare each element in b to 0 and generate a bit-mask:
        __m128i mask = _mm_cmpgt_epi16(b, zero);
        // AND each element in vector c2 with the bit-mask:
        c2 = _mm_and_si128(c2, mask);
        // AND each element in vector bc with the inverted bit-mask:
        bc = _mm_andnot_si128(mask, bc);
        // OR the results of the two AND operations:
        __m128i a = _mm_or_si128(c2, bc);
        // Store the result vector in eight consecutive elements in aa:
        StoreVector(aa + i, a);
    }
}
