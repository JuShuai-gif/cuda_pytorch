// Chapter: 使用向量操作
// Example 12.6. Function with vector parameters

Vec4f polynomial (Vec4f const & x)
{
    // polynomial(x) = 2.5*x^2 - 8*x + 2
    return (2.5f * x - 8.0f) * x + 2.0f;
}
