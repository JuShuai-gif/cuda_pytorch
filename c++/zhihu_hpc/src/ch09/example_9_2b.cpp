// Chapter: 优化内存访问
// Example 9.2b

void F3(bool y)
{
    union
    {
        int a[1000];
        float b[1000];
    };
    if (y)
    {
        F1(a);
    }
    else
    {
        F2(b);
    }
}
