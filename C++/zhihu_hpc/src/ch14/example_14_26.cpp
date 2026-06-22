// Chapter: 具体的优化主题
// Example 14.26

union
{
    float f;
    int i;
} u;
int n;
if (u.i & 0x7FFFFFFF)
{
    // check if nonzero
    u.i += n << 23; // add n to exponent
}
