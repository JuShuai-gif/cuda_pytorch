// Chapter: 编译器中的优化
// Example 8.20
//module1.cpp

int Func1(int x)
{
    return x*x + 1;
}

//module2.cpp
int Func2()
{
    int a = Func1(2);
    ...
}
