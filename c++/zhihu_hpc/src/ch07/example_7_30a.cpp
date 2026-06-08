// Chapter: 不同C++结构的效率
// Example 7.30a

int i;
for (i = 0; i < 20; i++)
{
    if (i % 2 == 0)
    {
        FuncA(i);
    }
    else
    {
        FuncB(i);
    }
    FuncC(i);
}
