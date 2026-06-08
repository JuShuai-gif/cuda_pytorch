// Chapter: 元编程
// Example 15.1d. Integer power using template metaprogramming

// Template for pow(x,N) where N is a positive integer constant.
// General case, N is not a power of 2:
template <bool IsPowerOf2, int N>
class powN
{
public:
    static double p(double x) {
    // Remove right-most 1-bit in binary representation of N:
    #define N1 (N & (N-1))
    return powN<(N1&(N1-1))==0,N1>::p(x) * powN<true,N-N1>::p(x);
    #undef N1
    }
};

// Partial template specialization for N a power of 2
template <int N>
class powN<true,N>
{
public:
    static double p(double x)
    {
        return powN<true,N/2>::p(x) * powN<true,N/2>::p(x);
    }
};

// Full template specialization for N = 1. This ends the recursion
template<>
class powN<true,1>
{
public:
    static double p(double x)
    {
        return x;
    }
};

// Full template specialization for N = 0
// This is used only for avoiding infinite loop if powN is
// erroneously called with IsPowerOf2 = false where it should be true.
template<>
class powN<true,0>
{
public:
    static double p(double x)
    {
        return 1.0;
    }
};

// Function template for x to the power of N
template <int N>
static inline double IntegerPower (double x)
{
    // (N & N-1)==0 if N is a power of 2
    return powN<(N & N-1)==0,N>::p(x);
}

// Use template to get x to the power of 10
double xpow10(double x)
{
    return IntegerPower<10>(x);
}
