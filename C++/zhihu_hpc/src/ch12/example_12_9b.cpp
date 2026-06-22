// Chapter: 使用向量操作
// Example 12.9b. Taylor series, vectorized
#include <dvec.h> // Define vector classes (Intel)
#include <pmmintrin.h> // SSE3 required
// This function adds the elements of a vector, uses SSE3.
// (This is faster than the function add_horizontal)
static inline float add_elements(__m128 const & x)
{
    __m128 s;
    s = _mm_hadd_ps(x, x);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}
float Exp(float x)
{
    // Approximate exp(x) for small x
    __declspec(align(16)) // align table by 16
    const float coef[16] = { // table of 1/n!
    1., 1./2., 1./6., 1./24., 1./120., 1./720., 1./5040.,
    1./40320., 1./362880., 1./3628800., 1./39916800.,
    1./4.790016E8, 1./6.22702E9, 1./8.71782E10,
    1./1.30767E12, 1./2.09227E13};
    float x2 = x * x; // x^2
    float x4 = x2 * x2; // x^4
    // Define vectors of four floats
    F32vec4 xxn(x4, x2*x, x2, x); // x^1, x^2, x^3, x^4
    F32vec4 xx4(x4); // x^4
    F32vec4 s(0.f, 0.f, 0.f, 1.f); // initialize sum
    for (int i = 0; i < 16; i += 4)
    {
        // Loop by 4
        s += xxn * _mm_load_ps(coef+i); // s += x^n/n!
        xxn *= xx4; // next four x^n
    }
    return add_elements(s); // add the four sums
}
