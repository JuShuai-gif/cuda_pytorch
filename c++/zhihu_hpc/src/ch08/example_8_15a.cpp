// Chapter: 编译器中的优化
// Example 8.15a

struct S1 {double a; double b;};
S1 list[100]; int i;
for (i = 0; i < 100; i++)
{
    list[i].a = 1.0;
    list[i].b = 2.0;
}
