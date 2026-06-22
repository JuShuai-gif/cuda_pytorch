// Chapter: 不同C++结构的效率
// Example 7.32b

double x, n, factorial = 1.0; int i;
for (i = (int)n - 2, x = 2.0; i >= 0; i--, x++)
    factorial *= x;
