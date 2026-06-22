// Chapter: 不同C++结构的效率
// Example 7.15a. Array with bounds checking

#include <cstring>

template <typename T, unsigned int N>
class SafeArray {
protected:
    T a[N]; // Array with N elements of type T39
public:
    SafeArray() {
        // Constructor
        memset(a, 0, sizeof(a)); // Initialize to zero
    }
    int Size() {
        // Return the size of the array
        return N;
    }
    T &operator[](unsigned int i) {
        // Safe [] array index operator
        if (i >= N) {
            // Index out of range. The next line provokes an error.
            // You may insert any other error reporting here:
            return *(T *)0; // Return a null reference to provoke error
        }
        // No error
        return a[i]; // Return reference to a[i]
    }
};
