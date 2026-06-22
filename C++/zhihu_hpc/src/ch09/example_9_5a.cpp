// Chapter: 优化内存访问
// Example 9.5a
const int SIZE = 64;// number of rows/columns in matrix
void transpose(double a[SIZE][SIZE])// function to transpose matrix
{
    // define a macro to swap two array elements:
    #define swapd(x,y) {temp=x; x=y; y=temp;}
    int r, c; double temp;
    for (r = 1; r < SIZE; r++)
    { // loop through rows
        for (c = 0; c < r; c++)
        {
            // loop columns below diagonal
            swapd(a[r][c], a[c][r]); // swap elements
        }
    }
}

void test ()
{
    __declspec(__align(64))     // align by cache line size
    double matrix[SIZE][SIZE]; // define matrix
    transpose(matrix);         // call transpose function
}
