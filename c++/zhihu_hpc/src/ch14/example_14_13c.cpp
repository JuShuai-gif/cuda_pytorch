// Chapter: 具体的优化主题
// Example 14.13c

int list[301];
int i;
for (i = 0; i < 301; i += 3)
{
    list[i] = 0;
    list[i+1] = 1;
    list[i+2] = 2;
}
list[300] = 0;
