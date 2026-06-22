// Chapter: 编译器中的优化
// Example 8.13b

int i, a[100], b, temp;
temp = b * b + 1;
for (i = 0; i < 100; i++)
{
    a[i] = temp;
}
