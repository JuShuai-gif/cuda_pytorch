// Chapter: 使用向量操作
// Example 12.1b. Vectorization with alignment problem

void AddTwo(int * __restrict aa, int * __restrict bb)
{
    for (int i = 0; i < size; i++)
    {
        aa[i] = bb[i] + 2;
    }
}
