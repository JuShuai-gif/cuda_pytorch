// Chapter: 使用向量操作
// Example 12.4e. Same example, using VCL

#include "vectorclass.h" // Define vector classes
void SelectAddMul(short int aa[], short int bb[], short int cc[])
{
    // Define vector objects
    Vec16s a, b, c;
    // Roll out loop by eight to fit the eight-element vectors:
    for (int i = 0; i < 256; i += 16)
    {
        // Load eight consecutive elements from bb into vector b:
        b.load(bb+i);
        // Load eight consecutive elements from cc into vector c:
        c.load(cc+i);
        // result = b > 0 ? c + 2 : b * c;
        a = select(b > 0, c + 2, b * c);
        // Store the result vector in eight consecutive elements in aa:
        a.store(aa+i);
    }
}
