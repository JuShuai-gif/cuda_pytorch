// Chapter: 具体的优化主题
// Example 14.28

union
{
    float f;
    unsigned int i;
} u, v;
if (u.i * 2 > v.i * 2)
{
    // abs(u.f) > abs(v.f)
}
