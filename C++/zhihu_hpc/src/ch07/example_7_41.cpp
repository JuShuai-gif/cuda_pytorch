// Chapter: 不同C++结构的效率
// Example 7.41
class S3
{
public:
    int a;
    int b;
    int Sum1() {return a + b;}
};
int Sum2(S3 * p) {return p->a + p->b;}
int Sum3(S3 & r) {return r.a + r.b;}
