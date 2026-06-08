// Chapter: 编译器中的优化
// Example 8.24. Integer constant

const int ArraySize = 1000;
int List[ArraySize];
...
for (int i = 0; i < ArraySize; i++)
    List[i]++;
