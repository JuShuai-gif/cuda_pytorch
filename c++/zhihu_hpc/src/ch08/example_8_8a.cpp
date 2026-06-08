// Chapter: 编译器中的优化
// Example 8.8a

double x, y, z; bool b;
if (b)
{
    y = sin(x);
    z = y + 1.;
}
else
{
    y = cos(x);
    z = y + 1.;
}
