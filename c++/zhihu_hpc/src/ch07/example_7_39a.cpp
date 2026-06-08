// Chapter: 不同C++结构的效率
// Example 7.39a

struct S1
{
    short int a;         // 2 bytes. first byte at 0, last byte at 1
                         // 6 unused bytes
    double b;            // 8 bytes. first byte at 8, last byte at 15
    int d;               // 4 bytes. first byte at 16, last byte at 19
                         // 4 unused bytes
};
S1 ArrayOfStructures[100];
