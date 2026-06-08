// Chapter: 不同C++结构的效率
// Example 7.3. Explain volatile

volatile int seconds; // incremented every second by another thread
void DelayFiveSeconds()
{
    seconds = 0;
    while (seconds < 5)
    {
        // do nothing while seconds count to 5
    }
}
