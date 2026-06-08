// Chapter: 具体的优化主题
// Example 14.3b
int n;
char const * const Greek[4] = {
    "Alpha", "Beta", "Gamma", "Delta"
    };
if ((unsigned int)n < 4)
{
    // Check that index is not out of range
    printf(Greek[n]);
}
