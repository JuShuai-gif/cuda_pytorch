// Chapter: 不同C++结构的效率
// Example 7.28

class c1
{
    const int x; // constant data
public:
    c1() : x(0) {}; // constructor initializes x to 0
    void xplus2()
    {
        // this function can modify x
        *const_cast<int*>(&x) += 2;
    } // add 2 to x
};
