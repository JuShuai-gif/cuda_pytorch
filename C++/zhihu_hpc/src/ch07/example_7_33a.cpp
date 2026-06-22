// Chapter: 不同C++结构的效率
// Example 7.33a

const int size = 1000; int i;
float a[size], b[size];
// set a to zero
for (i = 0; i < size; i++)
    a[i] = 0.0;
// copy a to b
for (i = 0; i < size; i++)
    b[i] = a[i];
