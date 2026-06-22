// Chapter: 编译器中的优化
// Example 8.3a

float parabola (float x)
{
    return x * x + 1.0f;
}
float a, b;
a = parabola (2.0f);
b = a + 1.0f;
