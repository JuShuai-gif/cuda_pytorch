// Chapter: 不同C++结构的效率
// Example 7.46

int Multiply (int x, int m)
{
    return x * m;
}

template <int m>
int MultiplyBy (int x)
{
    return x * m;
}

int a, b;
a = Multiply(10,8);
b = MultiplyBy<8>(10);
