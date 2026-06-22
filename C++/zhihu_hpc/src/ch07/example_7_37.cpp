// Chapter: 不同C++结构的效率
// Example 7.37. Factorial as recursive function

unsigned long int factorial(unsigned int n)
{
    if (n < 2) return 1;
        return n * factorial(n-1);
}
