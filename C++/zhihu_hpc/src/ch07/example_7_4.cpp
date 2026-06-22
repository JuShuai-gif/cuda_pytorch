// Chapter: 不同C++结构的效率
// Example 7.4. Signed and unsigned integers

int a, b;
double c;
b = (unsigned int)a / 10; // Convert to unsigned for fast division
c = a * 2.5; // Use signed when converting to double
