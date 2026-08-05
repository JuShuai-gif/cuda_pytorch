// Template basics: functions, classes, non-type parameters, static_assert.
//
// The book (PDF p.210-214): templates generate regular C++ code for each
// instantiation. Non-type (integral) template parameters make the compiler
// generate a distinct function per value, enabling static_assert at compile
// time.

#include <cstdio>

namespace {

// Templated function: one function per type (PDF p.211).
template <typename T>
T pow_n(const T& v, int n) {
    T product = T{1};
    for (int i = 0; i < n; ++i) {
        product *= v;
    }
    return product;
}

// Non-type template parameter: one function per N (PDF p.212).
template <typename T, int N>
T const_pow_n(const T& v) {
    static_assert(N >= 0, "N must be non-negative");
    T product = T{1};
    for (int i = 0; i < N; ++i) {
        product *= v;
    }
    return product;
}

// Templated class (PDF p.211).
template <typename T>
class Rectangle {
public:
    Rectangle(T x, T y, T w, T h) : x_(x), y_(y), w_(w), h_(h) {}
    T area() const { return w_ * h_; }
    T width() const { return w_; }
    T height() const { return h_; }

private:
    T x_, y_, w_, h_;
};

template <typename T>
bool is_square(const Rectangle<T>& r) {
    return r.width() == r.height();
}

}  // namespace

int main() {
    std::printf("== templates_basics ==\n");

    // pow_n with float and int -> two generated functions.
    const float x = pow_n(2.0F, 3);
    const int y = pow_n(3, 3);
    std::printf("pow_n(2,3)=%g pow_n(3,3)=%d\n", x, y);

    // Non-type parameter: const_pow_n<N> generates one function per N.
    std::printf("const_pow_n<2>(4)=%g const_pow_n<3>(4)=%g\n",
                const_pow_n<float, 2>(4.0F), const_pow_n<float, 3>(4.0F));

    // Templated class.
    const Rectangle<float> rectf{2.0F, 2.0F, 4.0F, 4.0F};
    const Rectangle<int> recti{-2, -2, 4, 4};
    std::printf("rectf area=%g square=%d\n", rectf.area(),
                is_square(rectf));
    std::printf("recti area=%d square=%d\n", recti.area(), is_square(recti));

    // static_assert on template parameter value.
    const auto ok = const_pow_n<int, 5>(2);
    (void)ok;
    // const_pow_n<int, -1>(2) would NOT compile.

    return 0;
}
