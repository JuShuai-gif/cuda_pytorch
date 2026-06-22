// Chapter: 具体的优化主题
// Example 14.1a

int factorial (int n)
{
    // n!
    int i, f = 1;
    for (i = 2; i <= n; i++)
        f *= i;
    return f;
}
