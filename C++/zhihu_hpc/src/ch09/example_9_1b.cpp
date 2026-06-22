// Chapter: 优化内存访问
// Example 9.1b

int Func(int);
const int size = 1024;
struct Sab {int a; int b;};
Sab ab[size];
int i;
...
for (i = 0; i < size; i++)
{
    ab[i].b = Func(ab[i].a);
}
