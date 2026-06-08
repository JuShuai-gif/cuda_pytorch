// Chapter: 为不同指令集生成多个版本的关键代码
// Example 13.2. CPU dispatching in Gnu compiler
// Same as example 13.1, Requires binutils version 2.20 or later

// Header file for InstructionSet()
#include "asmlib.h"

// Lowest version
int CriticalFunction_386(int parm1, int parm2) {...}

// SSE2 version
int CriticalFunction_SSE2(int parm1, int parm2) {...}

// AVX version
int CriticalFunction_AVX(int parm1, int parm2) {...}

// Prototype for the common entry point
extern "C" int CriticalFunction ();
__asm__ (".type CriticalFunction, @gnu_indirect_function");

// Make the dispatcher function.
typeof(CriticalFunction) * CriticalFunctionDispatch(void)
    __asm__ ("CriticalFunction");
typeof(CriticalFunction) * CriticalFunctionDispatch(void)
{
    // Returns a pointer to the desired function version
    // Get supported instruction set, using asmlib library
    int level = InstructionSet();
    // Set pointer to the appropriate version (May use a table
    // of function pointers if there are many branches):
    if (level >= 11)
    {
        // AVX supported
        return &CriticalFunction_AVX;
    }
    if (level >= 4)
    {
        // SSE2 supported
        return &CriticalFunction_SSE2;
    }
    // Default version
    return &CriticalFunction_386;
}

int main()
{
    int a, b, c;
    ...
    // Call critical function
    a = CriticalFunction(b, c);
    ...
    return 0;
}
