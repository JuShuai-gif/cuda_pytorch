// Chapter: 编译器中的优化
// Example 8.26a

void Func(int a[], int & r)
{
    int i;
    for (i = 0; i < 100; i++)
    {
        a[i] = r + i/2;
    }
}
