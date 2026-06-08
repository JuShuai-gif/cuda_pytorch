// Chapter: 不同C++结构的效率
// Example 7.49

// Portability note: This example is specific to Microsoft compilers.
// It will look different in other compilers.
#include <excpt.h>
#include <float.h>
#include <math.h>
#define EXCEPTION_FLT_OVERFLOW 0xC0000091L

void MathLoop()
{
    const int arraysize = 1000; unsigned int dummy;
    double a[arraysize], b[arraysize], c[arraysize];
    // Enable exception for floating point overflow:
    _controlfp_s(&dummy, 0, _EM_OVERFLOW);
    //_controlfp(0, _EM_OVERFLOW); // if above line doesn't work
    int i = 0; // Initialize loop counter outside both loops
    // The purpose of the while loop is to resume after exceptions:
    while (i < arraysize)
    {
        // Catch exceptions in this block:
        __try
        {
            // Main loop for calculations:
            for ( ; i < arraysize; i++)
            {
                // Overflow may occur in multiplication here:
                a[i] = log (b[i] * c[i]);
            }
        }
        // Catch floating point overflow but no other exceptions:
        __except (GetExceptionCode() == EXCEPTION_FLT_OVERFLOW
            ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH)
        {
            // Floating point overflow has occurred.
            // Reset floating point status:
            _fpreset();
            _controlfp_s(&dummy, 0, _EM_OVERFLOW);
            // _controlfp(0, _EM_OVERFLOW); // if above doesn't work
            // Re-do the calculation in a way that avoids overflow:
            a[i] = log(b[i]) + log(c[i]);
            // Increment loop counter and go back into the for-loop:
            i++;
        }
    }
}
