// Chapter: 不同C++结构的效率
// Example 7.31b

char string[100], *p = string; int i, StringLength;
for (i = StringLength; i > 0; i--)
    *(p++) |= 0x20;
