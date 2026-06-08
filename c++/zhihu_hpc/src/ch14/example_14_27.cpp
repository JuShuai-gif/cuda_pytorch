// Chapter: 具体的优化主题
// Example 14.27

union
{
    float f;
    int i;
} u, v;
if (u.i > v.i)
{
    // u.f > v.f if both positive
}
