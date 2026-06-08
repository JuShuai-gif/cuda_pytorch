// Chapter: 编译器中的优化
// Example 8.13a

int i, a[100], b;
for (i = 0; i < 100; i++)
{
    a[i] = b * b + 1;
}
