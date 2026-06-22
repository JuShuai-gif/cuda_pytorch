// Chapter: 编译器中的优化
// Example 8.7

int SomeFunction (int a, int x[])
{
    int b, c;
    x[0] = a;
    b = a + 1;
    x[1] = b;
    c = b + 1;
    return c;
}
