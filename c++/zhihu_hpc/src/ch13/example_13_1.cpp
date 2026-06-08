// Chapter: 为不同指令集生成多个版本的关键代码
// Example 13.1
// CPU dispatching on first call

// Header file for InstructionSet()
#include "asmlib.h"

// Define function type with desired paramet
typedef int CriticalFunctionType(int parm1, int parm2);

// Function prototype
CriticalFunctionType CriticalFunction_Dispatch;

// Function pointer serves as entry point.
// After first call it will point to the appropriate function version
CriticalFunctionType * CriticalFunction = &CriticalFunction_Dispatch;

// Lowest version
int CriticalFunction_386(int parm1, int parm2) {...}

// SSE2 version
int CriticalFunction_SSE2(int parm1, int parm2) {...}

// AVX version
int CriticalFunction_AVX(int parm1, int parm2) {...}

// Dispatcher. Will be called only first time
int CriticalFunction_Dispatch(int parm1, int parm2)
{
    // Get supported instruction set, using asmlib library
    int level = InstructionSet();
    // Set pointer to the appropriate version (May use a table
    // of function pointers if there are many branches):
    if (level >= 11)
    {
        // AVX supported
        CriticalFunction = &CriticalFunction_AVX;
    }
    else if (level >= 4)
    {
        // SSE2 supported
        CriticalFunction = &CriticalFunction_SSE2;
    }
    else
    {
        // Generic version
        CriticalFunction = &CriticalFunction_386;
    }
    // Now call the chosen version
    return (*CriticalFunction)(parm1, parm2);
}

int main()
{
    int a, b, c;
    ...
    // Call critical function through function pointer
    a = (*CriticalFunction)(b, c);
    ...
    return 0;
}
