// Chapter: 不同C++结构的效率
// Example 7.15b
// Needs: SafeArray from example_7_15a

#include <iostream>

SafeArray<float, 100> list; // Make array of 100 floats
for (int i = 0; i < list.Size(); i++) {
    // Loop through array
    cout << list[i] << endl; // Output array element
}
