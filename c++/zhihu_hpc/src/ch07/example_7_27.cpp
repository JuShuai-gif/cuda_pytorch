// Chapter: 不同C++结构的效率
// Example 7.27

float x;
*(int*)&x |= 0x80000000; // Set sign bit of x
