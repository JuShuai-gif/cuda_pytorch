// Stack vs heap: growth direction, addresses, and size limits.
//
// The book (PDF p.177-181) explains that the stack grows downwards on most
// platforms (towards lower addresses), is per-thread, never fragments, and
// has a fixed maximum size (~8 MB by default on Linux). The heap is shared,
// grows upwards, and can fragment.

#include <cstddef>
#include <cstdio>
#include <memory>
#include <new>

namespace {

void func2() {
    int i = 0;
    std::printf("func2() stack addr: %p\n", static_cast<void*>(&i));
}

void func1() {
    int i = 0;
    std::printf("func1() stack addr: %p\n", static_cast<void*>(&i));
    func2();
}

}  // namespace

int main() {
    std::printf("== stack_vs_heap ==\n");

    // Stack growth direction: later function calls get lower addresses.
    int main_i = 0;
    std::printf("main() stack addr: %p\n", static_cast<void*>(&main_i));
    func1();
    func2();

    // Heap growth direction: consecutive allocations usually go up.
    int* h1 = new int{};
    int* h2 = new int{};
    std::printf("heap h1: %p\n", static_cast<void*>(h1));
    std::printf("heap h2: %p\n", static_cast<void*>(h2));
    delete h1;
    delete h2;

    // Stack default size (reported by ulimit).
    std::printf("\n(default stack size via `ulimit -s` is printed by the shell)\n");

    // Stack objects vs heap objects: allocation speed differs by orders of
    // magnitude; we just observe addresses here, timing is in the benchmark
    // if present.
    std::printf("stack local: %p\n", static_cast<void*>(&main_i));
    std::unique_ptr<int> heap_obj(new int{});
    std::printf("heap object: %p\n", static_cast<void*>(heap_obj.get()));
    return 0;
}
