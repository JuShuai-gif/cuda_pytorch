// Chapter: 具体的优化主题
// Example 14.9

struct S1
{
    int a;
    int b;
    int c;
    int UnusedFiller;
};
int order(int x);
const int size = 100;
S1 list[size]; int i, j;
...
for (i = 0; i < size; i++)
{
    j = order(i);
    list[j].a = list[j].b + list[j].c;
}
