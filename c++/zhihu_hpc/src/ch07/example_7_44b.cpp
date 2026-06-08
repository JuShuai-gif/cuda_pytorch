// Chapter: 不同C++结构的效率
// Example 7.44b

union Bitfield
{
struct
{
    int a:4;
    int b:2;
    int c:2;
};
char abc;
};
Bitfield x;
int A, B, C;
x.abc = A | (B << 4) | (C << 6);
