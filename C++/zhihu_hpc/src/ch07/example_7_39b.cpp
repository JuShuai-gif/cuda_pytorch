// Chapter: 不同C++结构的效率
// Example 7.39b

struct S1
{
    double b;     // 8 bytes. first byte at 0, last byte at 7
    int d;        // 4 bytes. first byte at 8, last byte at 11
    short int a;  // 2 bytes. first byte at 12, last byte at 13
                  // 2 unused bytes
};
S1 ArrayOfStructures[100];
