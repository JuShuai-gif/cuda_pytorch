// Chapter: 不同C++结构的效率
// Example 7.35. Tail call

void function2(int x);
void function1(int y)
{
    ...
    function2(y+1);
}
