// Chapter: 不同C++结构的效率
// Example 7.1
float SomeFunction (int x) {
    static float list[] = {1.1, 0.3, -2.0, 4.4, 2.5};
    return list[x];
}
