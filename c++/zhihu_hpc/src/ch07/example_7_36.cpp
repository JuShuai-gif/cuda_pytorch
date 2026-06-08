// Chapter: 不同C++结构的效率
// Example 7.36. Tail call with return value

int function2(int x);
int function1(int y)
{
    ...
    return function2(y+1);
}
