// Chapter: 不同C++结构的效率
// Example 7.30b

int i;
for (i = 0; i < 20; i += 2)
{
    FuncA(i);
    FuncC(i);
    FuncB(i+1);
    FuncC(i+1);
}
