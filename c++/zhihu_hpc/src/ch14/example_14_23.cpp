// Chapter: 具体的优化主题
// Example 14.23

union
{
    float f;
    int i;
} u;
u.i ^= 0x80000000; // flip sign bit of u.f
