// Chapter: 不同C++结构的效率
// Example 7.12

void FuncA (int * p)
{
    *p = *p + 2;
}
void FuncB (int & r)
{
    r = r + 2;
}
