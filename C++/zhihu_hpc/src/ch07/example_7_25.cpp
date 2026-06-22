// Chapter: 不同C++结构的效率
// Example 7.25

unsigned int u; double d;
d = (double)(signed int)u; // Faster, but risk of overflow
