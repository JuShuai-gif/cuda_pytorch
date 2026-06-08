// Chapter: 编译器中的优化
// Example 8.22

#ifdef __GNUC__
#define pure_function __attribute__((const))
#else
#define pure_function
#endif

double Func1(double) pure_function ;
double Func2(double x)
{
    return Func1(x) * Func1(x) + 1.;
}
