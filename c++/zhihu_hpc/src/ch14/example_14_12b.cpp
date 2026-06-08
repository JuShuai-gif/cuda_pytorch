// Chapter: 具体的优化主题
// Example 14.12b

int list[300];
int i, i_div_3;
for (i = i_div_3 = 0; i < 300; i += 3, i_div_3++)
{
    list[i] += i_div_3;
    list[i+1] += i_div_3;
    list[i+2] += i_div_3;
}
