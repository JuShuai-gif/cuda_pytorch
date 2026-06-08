// Chapter: 编译器中的优化
// Example 8.5a

void Plus2 (int * p)
{
    *p = *p + 2;
}
int a;
Plus2 (&a);
