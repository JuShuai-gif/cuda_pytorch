// Chapter: 乱序执行
// Example 11.3

const int size = 100; int i;
float a[size], b[size], c[size];
float register temp;
for (i = 0; i < size; i++)
{
    temp = a[i] + b[i];
    c[i] = temp * temp;
}
