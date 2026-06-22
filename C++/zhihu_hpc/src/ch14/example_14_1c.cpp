// Chapter: 具体的优化主题
// Example 14.1c

void CriticalInnerFunction ()
{
    // Table of factorials:
    const int FactorialTable[13] = {1, 1, 2, 6, 24, 120, 720,
        5040, 40320, 362880, 3628800, 39916800, 479001600};
    ...
    int i, a, b;
    // Critical innermost loop:
    for (i = 0; i < 1000; i++)
    {
        ...
        a = FactorialTable[b];
        ...
    }
}
