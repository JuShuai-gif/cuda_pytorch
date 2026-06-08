// Chapter: 不同C++结构的效率
// Example 7.48

class C1
{
public:
    ...
    ~C1();
};

void F1()
{
    C1 x;
    ...
}
void F0()
{
    try
    {
        F1();
    }
    catch (...)
    {
        ...
    }
}
