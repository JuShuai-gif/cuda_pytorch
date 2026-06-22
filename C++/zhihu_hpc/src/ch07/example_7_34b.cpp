// Chapter: 不同C++结构的效率
// Example 7.34b. Replace macro by template

template <typename T>
static inline T max(T const & a, T const & b)
{
    return a > b ? a : b;
}
