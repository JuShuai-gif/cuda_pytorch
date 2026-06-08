// Chapter: 编译器中的优化
// Example 8.8b

double x, y; bool b;
if (b)
{
    y = sin(x);
}
else
{
    y = cos(x);
}
z = y + 1.;
