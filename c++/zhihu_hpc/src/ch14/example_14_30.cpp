// Chapter: 具体的优化主题
// Example 14.30

const int size = 100;
// Array of 100 doubles:
union {double d; unsigned int u[2]} a[size];
unsigned int absvalue, largest_abs = 0;
int i, largest_index = 0;
for (i = 0; i < size; i++)
{
    // Get upper 32 bits of a[i] and shift out sign bit:
    absvalue = a[i].u[1] * 2;
    // Find numerically largest element (approximately):
    if (absvalue > largest_abs)
    {
        largest_abs = absvalue;
        largest_index = i;
    }
}
