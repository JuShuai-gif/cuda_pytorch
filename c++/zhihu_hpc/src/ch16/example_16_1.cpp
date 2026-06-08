// Chapter: 测试速度
// Example 16.1

#include <intrin.h> // Or #include <ia32intrin.h> etc.
long long ReadTSC()
{
    // Returns time stamp counter
    int dummy[4]; // For unused returns
    volatile int DontSkip; // Volatile to prevent optimizing
    long long clock; // Time
    __cpuid(dummy, 0); // Serialize
    DontSkip = dummy[0]; // Prevent optimizing away cpuid
    clock = __rdtsc(); // Read time
    return clock;
}
