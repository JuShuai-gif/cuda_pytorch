// Chapter: 编译器中的优化
// Example 8.19. Devirtualization

class C0
{
public:
    virtual void f();
};
class C1 : public C0
{
public:
    virtual void f();
};
void g()
{
    C1 obj1;
    C0 * p = & obj1;
    p->f(); // Virtual call to C1::f
}
