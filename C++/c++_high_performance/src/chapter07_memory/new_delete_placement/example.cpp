// new/delete, placement new, and <memory> helpers.
//
// The book (PDF p.182-185): a new expression = allocate + construct; delete =
// destruct + deallocate. Placement new separates allocation from construction.
// C++17 <memory> provides uninitialized_* / destroy_at for the same job.

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

namespace {

struct User {
    explicit User(const char* n) : name(n) {}
    void print_name() const { std::printf("user: %s\n", name.c_str()); }
    std::string name;
};

}  // namespace

int main() {
    std::printf("== new_delete_placement ==\n");

    // 1. Regular new/delete: allocate + construct / destruct + deallocate.
    auto* user = new User{"John"};
    user->print_name();
    delete user;

    // 2. Placement new: construct an object in pre-allocated memory.
    void* memory = std::malloc(sizeof(User));
    if (memory == nullptr) {
        return 1;
    }
    auto* u2 = new (memory) User("Jane");
    u2->print_name();
    u2->~User();        // explicit destructor call (only valid after placement new)
    std::free(memory);

    // 3. C++17 <memory> helpers: uninitialized_fill_n + destroy_at.
    void* memory2 = std::malloc(sizeof(User));
    if (memory2 == nullptr) {
        return 1;
    }
    auto* u3 = static_cast<User*>(memory2);
    std::uninitialized_fill_n(u3, 1, User{"Doe"});
    u3->print_name();
    std::destroy_at(u3);
    std::free(memory2);

    std::printf("placement new and <memory> helpers done\n");
    return 0;
}
