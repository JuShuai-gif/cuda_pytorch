// Chapter: 具体的优化主题
// Example 14.11

int a, b, c;
a = b % c; // This is slow
a = b % 10; // Modulo by a constant is faster
a = (unsigned int)b % 10; // Still faster if unsigned
a = b % 16; // Faster if divisor is a power of 2
a = (unsigned int)b % 16; // Still faster if unsigned
