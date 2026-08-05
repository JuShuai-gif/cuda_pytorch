// Experiment 11: AoS vs SoA.
//
// Particle data benchmark. Compares:
//  - AoS touching all fields
//  - AoS touching only x,y (wasted cache line bytes)
//  - SoA touching x,y
// Reports time and effective bandwidth (only the used bytes counted).
//
// Reference: PDF 6.2.1 (data layout, splitting large structures).

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#include "benchmark.h"

static constexpr int N = 1 << 21;   // 2M particles
static constexpr int kRounds = 5;

struct alignas(16) ParticleAoS {
    float x, y, z, velocity, mass;
};

int main() {
    std::printf("Experiment 11: AoS vs SoA (%d particles)\n", N);

    std::vector<ParticleAoS> aos(N);
    std::vector<float> sx(N), sy(N);
    for (int i = 0; i < N; ++i) {
        aos[i].x = aos[i].y = aos[i].z = aos[i].velocity = aos[i].mass = 1.0f;
        sx[i] = sy[i] = 1.0f;
    }

    auto aos_all = [&] {
        float s = 0;
        for (auto& p : aos) s += p.x + p.y + p.z + p.velocity + p.mass;
        bm::do_not_optimize(s);
    };
    auto aos_xy = [&] {
        float s = 0;
        for (auto& p : aos) s += p.x + p.y;
        bm::do_not_optimize(s);
    };
    auto soa_xy = [&] {
        float s = 0;
        for (int i = 0; i < N; ++i) s += sx[i] + sy[i];
        bm::do_not_optimize(s);
    };

    struct Mode { const char* name; std::function<void()> fn; size_t used_bytes; };
    Mode modes[] = {
        {"aos_all_fields", aos_all, N * sizeof(ParticleAoS)},
        {"aos_only_xy", aos_xy, N * 8u},
        {"soa_only_xy", soa_xy, N * 8u},
    };

    std::printf("%-18s %-12s %-14s %-14s\n", "mode", "time_ms", "GB/s(used)",
                "checksum");
    for (auto& m : modes) {
        m.fn();
        auto res = bm::time_rounds(kRounds, m.fn);
        double gbps = (double)m.used_bytes / (res.median_ms * 1e6 / 1e9) / 1e9;
        std::printf("%-18s %-12.3f %-14.3f %-14.0f\n", m.name, res.median_ms,
                    gbps, (double)m.used_bytes);
    }
    return 0;
}
