// Chapter: 使用向量操作
// Example 12.8b. Sum of a list, rolled out by 4
float a[100];
float s0 = 0, s1 = 0, s2 = 0, s3 = 0, sum;
for (int i = 0; i < 100; i += 4)
{
    s0 += a[i];
    s1 += a[i+1];
    s2 += a[i+2];
    s3 += a[i+3];
}
sum = (s0+s1)+(s2+s3);
