// Chapter: 具体的优化主题
// Example 14.19

static inline int lrint (double const x)    { // Round to nearest integer
int n;
#if defined(__unix__) || defined(__GNUC__)
//  32位 Linux, Gnu/AT&T syntax:
__asm ("fldl %1 \n fistpl %0 " : "=m"(n) : "m"(x) : "memory" );
#else
//  32位 Windows, Intel/MASM syntax:
__asm fld qword ptr x;
__asm fistp dword ptr n;
#endif
return n;
}
