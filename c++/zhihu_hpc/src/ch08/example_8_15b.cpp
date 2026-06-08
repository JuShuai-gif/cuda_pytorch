// Chapter: 编译器中的优化
// Example 8.15b

struct S1 {double a; double b;};
S1 list[100], *temp;
for (temp = &list[0]; temp < &list[100]; temp++)
{
    temp->a = 1.0;
    temp->b = 2.0;
}
