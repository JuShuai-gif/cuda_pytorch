// Chapter: 优化内存访问
// Example 9.6a
const int SIZE = 512; // number of rows and columns in matrix
// function to transpose and copy matrix
void TransposeCopy(double a[SIZE][SIZE], double b[SIZE][SIZE])
{
    int r, c;
    for (r = 0; r < SIZE; r++)
    {
        for (c = 0; c < SIZE; c++)
        {
            a[c][r] = b[r][c];
        }
    }
}
