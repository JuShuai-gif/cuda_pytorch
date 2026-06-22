// Chapter: 编译器中的优化
// Example 8.21

void Func1 (int a[], int * p)
{
    int i;
    for (i = 0; i < 100; i++)
    {
        a[i] = *p + 2;
    }
}

void Func2()
{
    int list[100];
    Func1(list, &list[8]);
}
