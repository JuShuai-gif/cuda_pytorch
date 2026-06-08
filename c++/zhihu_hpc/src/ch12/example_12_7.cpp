// Chapter: 使用向量操作
// Example 12.7. Vector class code with automatic CPU dispatching
#include "vectorclass.h" // vector class library
#include <stdio.h> // define fprintf
// define function type
typedef void FuncType(short int aa[], short int bb[], short int cc[]);
// function prototypes for each version
FuncType SelectAddMul, SelectAddMul_SSE2, SelectAddMul_SSE41,
SelectAddMul_AVX2, SelectAddMul_dispatch;
// Define function name depending on instruction set
#if INSTRSET == 2 // SSE2
    #define FUNCNAME SelectAddMul_SSE2
#elif INSTRSET == 5 // SSE4.1
    #define FUNCNAME SelectAddMul_SSE41
#elif INSTRSET == 8 // AVX2
    #define FUNCNAME SelectAddMul_AVX2
#endif
// specific version of the function. Compile once for each version
void FUNCNAME(short int aa[], short int bb[], short int cc[])
{
    Vec16s a, b, c; // Define biggest possible vector objects
    // Roll out loop by 16 to fit the biggest vectors:
    for (int i = 0; i < 256; i += 16)
    {
        b.load(bb+i);
        c.load(cc+i);
        a = select(b > 0, c + 2, b * c);
        a.store(aa+i);
    }
}
#if INSTRSET == 2
// make dispatcher in only the lowest of the compiled versions
#include "instrset_detect.cpp" // instrset_detect function
// Function pointer initially points to the dispatcher.
// After first call it points to the selected version
FuncType * SelectAddMul_pointer = &SelectAddMul_dispatch;
// Dispatcher
void SelectAddMul_dispatch(short int aa[], short int bb[], short int cc[])
{
    // Detect supported instruction set
    int iset = instrset_detect();
    // Set function pointer
    if (iset >= 8)
        SelectAddMul_pointer = &SelectAddMul_AVX2;
    else if (iset >= 5)
        SelectAddMul_pointer = &SelectAddMul_SSE41;
    else if (iset >= 2)
        SelectAddMul_pointer = &SelectAddMul_SSE2;
    else
    {
        // Error: lowest instruction set not supported
        fprintf(stderr, "\nError: Instruction set SSE2 not supported");
        return;
    }
    // continue in dispatched version
    return (*SelectAddMul_pointer)(aa, bb, cc);
}
// Entry to dispatched function call
inline void SelectAddMul(short int aa[], short int bb[], short int cc[])
{
    // go to dispatched version
    return (*SelectAddMul_pointer)(aa, bb, cc);
}
#endif // INSTRSET == 2
