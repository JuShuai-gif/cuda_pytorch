// Chapter: 编译器中的优化
// Example 8.23a. Loop to make table of polynomial

const double A = 1.1, B = 2.2, C = 3.3;  // Polynomial coefficients
double Table[100];                       // Table
int x; // Loop counter
for (x = 0; x < 100; x++)
{
    Table[x] = A*x*x + B*x + C;          // Calculate polynomial
}
