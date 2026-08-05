// Amortized time complexity of vector::push_back.
//
// The book (PDF p.92-94) explains that push_back is O(1) amortized: the
// internal array grows exponentially (here doubling), so the expensive
// reallocation happens only log2(n) times over n insertions.
//
// We log capacity and the number of element moves to verify the geometric
// growth and that total moves are O(n).

#include <cstddef>
#include <cstdio>
#include <vector>

namespace {

struct Tracked {
    explicit Tracked(int v = 0) : value(v) {}
    Tracked(const Tracked& other) : value(other.value) { ++copies; }
    Tracked(Tracked&& other) noexcept : value(other.value) { ++moves; }
    Tracked& operator=(const Tracked&) = delete;
    Tracked& operator=(Tracked&&) noexcept {
        value = 0;
        ++moves;
        return *this;
    }
    int value = 0;
    static std::size_t copies;
    static std::size_t moves;
};

std::size_t Tracked::copies = 0;
std::size_t Tracked::moves = 0;

}  // namespace

int main() {
    std::printf("== vector_growth ==\n");

    std::vector<Tracked> v;
    std::size_t last_capacity = 0;
    std::size_t reallocation_count = 0;

    // Insert 1024 elements; print capacity changes.
    for (std::size_t i = 0; i < 1024; ++i) {
        v.emplace_back(static_cast<int>(i));
        if (v.capacity() != last_capacity) {
            if (v.capacity() != 0) {
                std::printf("after %4zu inserts: capacity = %5zu "
                            "(moves so far: %zu)\n",
                            i + 1, v.capacity(), Tracked::moves);
            }
            last_capacity = v.capacity();
            ++reallocation_count;
        }
    }
    std::printf("size=%zu capacity=%zu reallocations=%zu\n", v.size(),
                v.capacity(), reallocation_count);
    std::printf("total moves (elements copied on reallocation): %zu\n",
                Tracked::moves);
    std::printf("moves / size = %.3f  (the amortized per-insert cost)\n",
                static_cast<double>(Tracked::moves) /
                    static_cast<double>(v.size()));

    return 0;
}
