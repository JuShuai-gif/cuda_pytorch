// Chapter: 编译器中的优化
// Example 8.14b

int i, a[100], temp;
temp = 3;
for (i = 0; i < 100; i++)
{
    a[i] = temp;
    temp += 9;
}
