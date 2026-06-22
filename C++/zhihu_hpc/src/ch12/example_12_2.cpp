// Chapter: 使用向量操作
// Example 12.2

__declspec(align(16)) // Make all instances of S1 aligned

struct S1
{ // Structure of 4 floats
    float a, b, c, d;
};
void Func()
{
    S1 x, y;
    ...
    x.a = y.a + 1.;
    x.b = y.b + 2.;
    x.c = y.c + 3.;
    x.d = y.d + 4.;
};
