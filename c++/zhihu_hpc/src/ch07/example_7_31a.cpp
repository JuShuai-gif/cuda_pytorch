// Chapter: 不同C++结构的效率
// Example 7.31a

char string[100], *p = string;
while (*p != 0)
    *(p++) |= 0x20;
