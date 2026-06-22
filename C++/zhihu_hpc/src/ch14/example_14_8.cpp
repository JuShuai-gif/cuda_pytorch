// Chapter: 具体的优化主题
// Example 14.8

const int rows = 10, columns = 8;
float matrix[rows][columns];
int i, j;
int order(int x);
...
for (i = 0; i < rows; i++)
{
    j = order(i);
    matrix[j][0] = i;
}
