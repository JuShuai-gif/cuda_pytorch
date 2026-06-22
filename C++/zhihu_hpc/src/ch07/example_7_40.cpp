// Chapter: 不同C++结构的效率
// Example 7.40

class S2
{
public:
    int a[100]; // 400 bytes. first byte at 0, last byte at 399
    int b; // 4 bytes. first byte at 400, last byte at 403
    int ReadB() {return b;}
};
