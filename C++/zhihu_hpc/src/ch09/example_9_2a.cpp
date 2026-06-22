// Chapter: 优化内存访问
// Example 9.2a

void F1(int x[]);
void F2(float x[]);
void F3(bool y)
{
    if (y)
    {
        int a[1000];
        F1(a);
    }
    else
    {
        float b[1000];
        F2(b);
    }
}
