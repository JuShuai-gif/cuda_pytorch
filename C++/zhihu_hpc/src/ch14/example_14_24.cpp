// Chapter: 具体的优化主题
// Example 14.24

union
{
    float f;
    int i;
} u;
u.i &= 0x7FFFFFFF; // set sign bit to zero
