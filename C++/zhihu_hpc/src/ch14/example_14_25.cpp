// Chapter: 具体的优化主题
// Example 14.25

union
{
    float f;
    int i;
} u;
if (u.i & 0x7FFFFFFF)
{
    // test bits 0 - 30
    // f is nonzero
}
else
{
    // f is zero
}
