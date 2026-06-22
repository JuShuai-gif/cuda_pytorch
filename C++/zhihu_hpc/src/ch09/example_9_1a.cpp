// Chapter: 优化内存访问
// Example 9.1a

int Func(int);
const int size = 1024;
int a[size], b[size], i;
...
for (i = 0; i < size; i++)
{
    b[i] = Func(a[i]);
}
