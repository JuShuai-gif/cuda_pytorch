// Chapter: 乱序执行
// Example 11.2a

const int size = 100;
float list[size], sum = 0; int i;
for (i = 0; i < size; i++)
    sum += list[i];
