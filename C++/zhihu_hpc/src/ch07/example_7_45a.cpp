// Chapter: 不同C++结构的效率
// Example 7.45a

class vector
{ // 2-dimensional vector
public:
    float x, y; // x,y coordinates
    vector() {} // default constructor
    vector(float a, float b)
    {
        x = a;
        y = b;
    } // constructor
    vector operator + (vector const & a)
    { // sum operator
        return vector(x + a.x, y + a.y);
    } // add elements
};
vector a, b, c, d;
a = b + c + d; // makes intermediate object for (b + c)
