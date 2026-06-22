// Chapter: 不同C++结构的效率
// Example 7.44a

struct Bitfield
{
    int a:4;
    int b:2;
    int c:2;
};
Bitfield x;
int A, B, C;
x.a = A;
x.b = B;
x.c = C;
