// Chapter: 具体的优化主题
// Example 14.23b

union
{
    double d;
    int i[2];
} u;
if (u.i[1] < 0)
{
    // test sign bit
    // u.d is negative or -0
}
