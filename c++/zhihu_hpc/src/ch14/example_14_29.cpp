// Chapter: 具体的优化主题
// Example 14.29

union
{
    float f;
    int i;
} u;
int n;
u.i = (n & 0x7FFFFF) | 0x3F800000; // Now 1.0 <= u.f < 2.0
