// Chapter: 不同C++结构的效率
// Example 7.38. Factorial function as loop

unsigned long int factorial(unsigned int n)
{
    unsigned long int product = 1;
    while (n > 1)
    {
        product *= n;
        n--;
    }
    return product;
}
