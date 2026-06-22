// Chapter: 编译器中的优化
// Example 8.11a

int SomeFunction (int a, bool b)
{
    if (b)
    {
        a = a * 2;
    }
    else
    {
        a = a * 3;
    }
    if (b)
    {
        return a + 1;
    }
    else
    {
        return a - 1;
    }
}
