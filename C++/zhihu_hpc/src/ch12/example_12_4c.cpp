// Chapter: 使用向量操作
// Example 12.4c. Same example, vectorized with SSE4.1

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
void SelectAddMul(short int aa[], short int bb[], short int cc[])
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
        // Use mask to choose between c2 and bc for each element
        __m128i a = _mm_blendv_epi8(bc, c2, mask);
        // Store the result vector in eight consecutive elements in aa:
        StoreVector(aa + i, a);
    }
}
