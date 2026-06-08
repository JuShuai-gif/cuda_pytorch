// Chapter: 不同C++结构的效率
// Example 7.47b. Compile-time polymorphism with templates

#include <iostream>

// Place non-polymorphic functions in the grandparent class:
class CGrandParent {
public:
    void NotPolymorphic();
};
// Any function that needs to call a polymorphic function goes in the
// parent class. The child class is given as a template parameter:
template <typename MyChild>
class CParent : public CGrandParent {
public:
    void Hello() {
        cout << "Hello ";
        // call polymorphic child function:
        (static_cast<MyChild *>(this))->Disp();
    }
};
// The child classes implement the functions that have multiple
// versions:
class CChild1 : public CParent<CChild1> {
public:
    void Disp() {
        cout << 1;
    }
};
class CChild2 : public CParent<CChild2> {
public:
    void Disp() {
        cout << 2;
    }
};
void test() {
    CChild1 Object1;
    CChild2 Object2;
    CChild1 *p1;
    p1 = &Object1;
    p1->Hello(); // Writes "Hello 1"
    CChild2 *p2;
    p2 = &Object2;
    p2->Hello(); // Writes "Hello 2"
}
