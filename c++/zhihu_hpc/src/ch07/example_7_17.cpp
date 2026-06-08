// Chapter: 不同C++结构的效率
// Example 7.17

const int rows = 20, columns = 50;
float matrix[rows][columns];
int i, j; float x;
for (i = 0; i < rows; i++)
    for (j = 0; j < columns; j++)
        matrix[i][j] += x;
