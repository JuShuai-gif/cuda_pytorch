// Chapter: 测试速度
// Example 16.2

#include <stdio.h>
#include <asmlib.h> // Use ReadTSC() from library asmlib..
                    // or from example 16.1
void CriticalFunction(); // This is the function we want to measure
...
const int NumberOfTests = 10; // Number of times to test
int i; long long time1;
long long timediff[NumberOfTests]; // Time difference for each test
for (i = 0; i < NumberOfTests; i++)
{
    // Repeat NumberOfTests times
    time1 = ReadTSC(); // Time before test
    CriticalFunction(); // Critical function to test
    timediff[i] = ReadTSC() - time1; // (time after) - (time before)
}
printf("\nResults:"); // Print heading
for (i = 0; i < NumberOfTests; i++)
{
    // Loop to print out results
    printf("\n%2i %10I64i", i, timediff[i]);
}
