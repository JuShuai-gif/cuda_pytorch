// Object pool demo: a small "game loop" allocating/destroying particles.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "object_pool.hpp"

namespace {

struct Particle {
    float x = 0.0F, y = 0.0F, vx = 0.0F, vy = 0.0F;
    int alive = 0;
};

}  // namespace

int main() {
    std::printf("== object_pool ==\n");

    // A pool large enough for the maximum number of simultaneous particles.
    chp::ObjectPool pool(sizeof(Particle), 10'000);

    std::vector<chp::Pooled<Particle>> particles;

    // Simulate 3 frames of a particle system: spawn then kill half.
    for (int frame = 0; frame < 3; ++frame) {
        // Spawn 3000 particles each frame.
        for (int i = 0; i < 3000; ++i) {
            particles.emplace_back(pool);
            auto& p = *particles.back();
            p.x = static_cast<float>(i);
            p.y = 0.0F;
            p.vx = static_cast<float>((i % 7) - 3);
            p.vy = 1.0F;
        }
        // Kill the first half of the oldest particles.
        const auto kill_count = particles.size() / 2;
        particles.erase(particles.begin(),
                        particles.begin() + static_cast<std::ptrdiff_t>(kill_count));
        std::printf("frame %d: %zu particles alive, pool in_use=%zu/%zu\n",
                    frame, particles.size(), pool.in_use(), pool.capacity());
    }

    // Pooled objects live on the pool; destroying them returns blocks.
    particles.clear();
    std::printf("after clear: pool in_use=%zu free=%zu\n",
                pool.in_use(), pool.free_count());

    return 0;
}
