// Chapter: 不同C++结构的效率
// Example 7.47a. Runtime polymorphism with virtual functions

#include <iostream>

class CHello {
public:
    void NotPolymorphic(); // Non-polymorphic functions go here
    virtual void Disp();   // Virtual function
    void Hello() {
        cout << "Hello ";
        Disp(); // Call to virtual function
    }
};
class C1 : public CHello {
public:
    virtual void Disp() {
        cout << 1;
    }
};
class C2 : public CHello {
public:
    virtual void Disp() {
        cout << 2;
    }
};
void test() {
    C1 Object1;
    C2 Object2;
    CHello *p;
    p = &Object1;
    p->NotPolymorphic(); // Called directly
    p->Hello();          // Writes "Hello 1"
    p = &Object2;
    p->Hello(); // Writes "Hello 2"
}
