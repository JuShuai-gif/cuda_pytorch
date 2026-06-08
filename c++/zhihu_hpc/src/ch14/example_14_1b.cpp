// Chapter: 具体的优化主题
// Example 14.1b

int factorial (int n)
{
    // n!
    // Table of factorials:
    const int FactorialTable[13] = {1, 1, 2, 6, 24, 120, 720,
        5040, 40320, 362880, 3628800, 39916800, 479001600};
    if ((unsigned int)n < 13)
    {
        // Bounds checking (see page 137)
        return FactorialTable[n]; // Table lookup
    }
    else
    {
        return 0; // return 0 if out of range
    }
}
