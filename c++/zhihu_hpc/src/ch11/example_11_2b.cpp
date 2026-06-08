// Chapter: 乱序执行
// Example 11.2b

const int size = 100;
float list[size], sum1 = 0, sum2 = 0; int i;
for (i = 0; i < size; i += 2)
{
    sum1 += list[i];
    sum2 += list[i+1];
}
sum1 += sum2;
