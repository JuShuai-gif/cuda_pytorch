// Chapter: 使用向量操作
// Example 12.9a. Taylor series
float Exp(float x)
{
    // Approximate exp(x) for small x
    float xn = x; // x^n
    float sum = 1.f; // sum, initialize to x^0/0!
    float nfac = 1.f; // n factorial
    for (int n = 1; n <= 16; n++)
    {
        sum += xn / nfac;
        xn *= x;
        nfac *= n+1;
    }
return sum;
}
