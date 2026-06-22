// Chapter: 不同C++结构的效率
// Example 7.43

#include <iostream>

union {
    float f;
    int i;
} x;
x.f = 2.0f;
x.i |= 0x80000000; // set sign bit of f
cout << x.f;       // will give -2.0
