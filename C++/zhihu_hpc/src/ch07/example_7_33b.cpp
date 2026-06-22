// Chapter: 不同C++结构的效率
// Example 7.33b
// Needs: wrapped in a function to compile standalone

#include <cstring>

const int size = 1000;
float a[size], b[size];
// set a to zero
memset(a, 0, sizeof(a));
// copy a to b
memcpy(b, a, sizeof(b));
