// Chapter: 编译器中的优化
// Example 8.11b

int SomeFunction (int a, bool b)
{
    if (b)
    {
        a = a * 2;
        return a + 1;
    }
    else
    {
        a = a * 3;
        return a - 1;
    }
}
