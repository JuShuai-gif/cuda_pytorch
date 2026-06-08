// Chapter: 使用向量操作
// Example 12.4d. Same example, using Intel vector classes

#include <dvec.h> // Define vector classes
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
    Is16vec8 zero(0,0,0,0,0,0,0,0);
    // Make a vector of (2,2,2,2,2,2,2,2)
    Is16vec8 two(2,2,2,2,2,2,2,2);
    // Roll out loop by eight to fit the eight-element vectors:
    for (int i = 0; i < 256; i += 8)
    {
        // Load eight consecutive elements from bb into vector b:
        Is16vec8 b = LoadVector(bb + i);
        // Load eight consecutive elements from cc into vector c:
        Is16vec8 c = LoadVector(cc + i);
        // result = b > 0 ? c + 2 : b * c;
        Is16vec8 a = select_gt(b, zero, c + two, b * c);
        // Store the result vector in eight consecutive elements in aa:
        StoreVector(aa + i, a);
    }
}
