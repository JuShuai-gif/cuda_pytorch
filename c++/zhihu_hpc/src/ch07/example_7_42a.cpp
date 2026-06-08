// Chapter: 不同C++结构的效率
// Example 7.42a. Multiple inheritance

class B1; class B2;
class D : public B1, public B2
{
public:
    int c;
};
