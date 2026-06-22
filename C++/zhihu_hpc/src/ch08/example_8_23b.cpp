// Chapter: 编译器中的优化
// Example 8.23b. Calculate polynomial with induction variables

const double A = 1.1, B = 2.2, C = 3.3; // Polynomial coefficients
double Table[100];                      // Table
int x;                                  // Loop counter
const double A2 = A + A;                // = 2*A
double Y = C;                           // = A*x*x + B*x + C
double Z = A + B;                       // = Delta Y
for (x = 0; x < 100; x++)
{
    Table[x] = Y;                       // Store result
    Y += Z;                             // Update induction variable Y
    Z += A2;                            // Update induction variable Z
}
