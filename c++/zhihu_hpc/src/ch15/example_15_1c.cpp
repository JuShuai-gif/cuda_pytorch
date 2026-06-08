// Chapter: 元编程
// Example 15.1c. Calculate integer power, loop unrolled

double xpow10(double x)
{
    double x2 = x *x; // x^2
    double x4 = x2*x2; // x^4
    double x8 = x4*x4; // x^8
    double x10 = x8*x2; // x^10
    return x10; // return x^10
}
