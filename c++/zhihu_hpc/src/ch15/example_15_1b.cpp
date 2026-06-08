// Chapter: 元编程
// Example 15.1b. Calculate integer power using loop

double ipow (double x, unsigned int n)
{
    double y = 1.0; // used for multiplication
    while (n != 0)
    {
        // loop for each bit in nn
        if (n & 1)
            y *= x; // multiply if bit = 1
        x *= x; // square x
        n >>= 1; // get next bit of n
    }
    return y; // return y = pow(x,n)
}
double xpow10(double x)
{
    return ipow(x,10); // ipow faster than pow
}
