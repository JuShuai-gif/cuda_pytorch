// Chapter: 具体的优化主题
// Example 14.4a

const int size = 16; int i;
float list[size];
...
if (i < 0 || i >= size)
{
    cout << "Error: Index out of range";
}
else
{
    list[i] += 1.0f;
}
