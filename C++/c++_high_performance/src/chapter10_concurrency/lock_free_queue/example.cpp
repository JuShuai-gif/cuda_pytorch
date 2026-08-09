// Single-producer / single-consumer lock-free queue (PDF p.309-311).

#include <cstdio>
#include <cstddef>

#include "lock_free_queue.hpp"

int main() {
    std::printf("== lock_free_queue ==\n");

    chp10::LockFreeQueue<int, 4> q;
    q.push(1);
    q.push(2);
    q.push(3);
    std::printf("size=%zu front=%d\n", q.size(), q.front());
    q.pop();
    std::printf("after pop: size=%zu front=%d\n", q.size(), q.front());

    // Ring wrap-around: push enough to wrap, but stay within capacity.
    q.push(4);
    q.push(5);
    std::printf("wrapped: size=%zu front=%d\n", q.size(), q.front());

    return 0;
}
