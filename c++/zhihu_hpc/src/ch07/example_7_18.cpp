// Chapter: 不同C++结构的效率
// Example 7.18

int FuncRow(int); int FuncCol(int);
const int rows = 20, columns = 32;
float matrix[rows][columns];
int i; float x;
for (i = 0; i < 100; i++)
    matrix[FuncRow(i)][FuncCol(i)] += x;
