// Chapter: 优化内存访问
// Example 9.5b

void transpose(double a[SIZE][SIZE])
{
    // Define macro to swap two elements:
    #define swapd(x,y) {temp=x; x=y; y=temp;}
    // Check if level-2 cache contentions will occur:
    if (SIZE > 256 && SIZE % 128 == 0)
    {
        // Cache contentions expected. Use square blocking:
        int r1, r2, c1, c2; double temp;
        // Define size of squares:
        const int TILESIZE = 8; // SIZE must be divisible by TILESIZE
        // Loop r1 and c1 for all squares:
        for (r1 = 0; r1 < SIZE; r1 += TILESIZE)
        {
            for (c1 = 0; c1 < r1; c1 += TILESIZE)
            {
            // Loop r2 and c2 for elements inside sqaure:
                for (r2 = r1; r2 < r1+TILESIZE; r2++)
                {
                    for (c2 = c1; c2 < c1+TILESIZE; c2++)
                    {
                        swapd(a[r2][c2],a[c2][r2]);
                    }
                }
            }
           // At the diagonal there is only half a square.
           // This triangle is handled separately:
            for (r2 = r1+1; r2 < r1+TILESIZE; r2++)
            {
                for (c2 = r1; c2 < r2; c2++)
                {
                    swapd(a[r2][c2],a[c2][r2]);
                }
            }
        }
    }
    else
    {
        // No cache contentions. Use simple method.
        // This is the code from example 9.5a:
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
}
