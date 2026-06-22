// Chapter: 使用向量操作
// Example 12.1a. Automatic vectorization

const int size = 1024;
int a[size], b[size];
// ...
for (int i = 0; i < size; i++)
{
    a[i] = b[i] + 2;
}
