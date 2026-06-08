// Chapter: 编译器中的优化
// Example 8.26b

void Func(int a[], int & r)
{
    int i;
    int Induction = r;
    for (i = 0; i < 100; i += 2)
    {
        a[i] = Induction;
        a[i+1] = Induction;
        Induction++;
    }
}
