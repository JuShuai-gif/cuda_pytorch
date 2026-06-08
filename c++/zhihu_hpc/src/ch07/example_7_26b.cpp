// Chapter: 不同C++结构的效率
// Example 7.26b

float a[100]; int i; float i2;
for (i = 0, i2 = 0; i < 100; i++, i2 += 2.0f)
    a[i] = i2;
