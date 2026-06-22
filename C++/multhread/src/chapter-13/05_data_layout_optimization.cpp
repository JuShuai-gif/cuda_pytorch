// 05_data_layout_optimization.cpp — 数据布局优化综合演示
// 演示: SoA vs AoS、冷热数据分离、struct padding 最佳实践

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

// ================================================================
// AoS (Array of Structures) vs SoA (Structure of Arrays)
// ================================================================

// AoS: 数据以对象为中心排列
struct ParticleAoS {
    float x, y, z;      // 位置
    float vx, vy, vz;   // 速度
    float mass;         // 质量
    int type;           // 类型
    char name[32];      // 名称 (冷数据)
};
// sizeof: 4*7 + 4 + 4 + 32 + padding ≈ 68 字节

// SoA: 每个字段独立成数组
struct ParticleSystemSoA {
    std::vector<float> x, y, z;
    std::vector<float> vx, vy, vz;
    std::vector<float> mass;
    std::vector<int> type;
    // name 等冷数据分离或按需访问
};

// ================================================================
// 基准测试: AoS vs SoA 的遍历性能
// ================================================================
void demo_aos_vs_soa() {
    std::cout << "=== AoS vs SoA 性能对比 ===\n";

    const size_t kCount = 1'000'000;

    // 构造 AoS 数据
    std::vector<ParticleAoS> aos(kCount);
    for (size_t i = 0; i < kCount; ++i) {
        aos[i].x = static_cast<float>(rand()) / RAND_MAX;
        aos[i].y = static_cast<float>(rand()) / RAND_MAX;
        aos[i].z = static_cast<float>(rand()) / RAND_MAX;
        aos[i].vx = static_cast<float>(rand()) / RAND_MAX;
        aos[i].vy = static_cast<float>(rand()) / RAND_MAX;
        aos[i].vz = static_cast<float>(rand()) / RAND_MAX;
    }

    // 构造 SoA 数据
    ParticleSystemSoA soa;
    soa.x.resize(kCount);
    soa.y.resize(kCount);
    soa.z.resize(kCount);
    soa.vx.resize(kCount);
    soa.vy.resize(kCount);
    soa.vz.resize(kCount);
    for (size_t i = 0; i < kCount; ++i) {
        soa.x[i] = aos[i].x;
        soa.y[i] = aos[i].y;
        soa.z[i] = aos[i].z;
        soa.vx[i] = aos[i].vx;
        soa.vy[i] = aos[i].vy;
        soa.vz[i] = aos[i].vz;
    }

    const int kRounds = 20;

    // 测试 AoS: 只更新位置 (但需要加载整个结构体)
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < kRounds; ++r) {
            for (size_t i = 0; i < kCount; ++i) {
                aos[i].x += aos[i].vx;
                aos[i].y += aos[i].vy;
                aos[i].z += aos[i].vz;
            }
        }
        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);
        std::cout << "  AoS (加载整个结构): " << elapsed.count()
                  << " ms\n";
    }

    // 测试 SoA: 只访问需要的字段
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < kRounds; ++r) {
            for (size_t i = 0; i < kCount; ++i) {
                soa.x[i] += soa.vx[i];
                soa.y[i] += soa.vy[i];
                soa.z[i] += soa.vz[i];
            }
        }
        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);
        std::cout << "  SoA (只加载需要的): " << elapsed.count()
                  << " ms\n";
    }

    std::cout << "  结论: 只访问部分字段时，SoA 避免加载无关数据到缓存\n";
    std::cout << "  sizeof(ParticleAoS) = " << sizeof(ParticleAoS)
              << " 字节\n";
}

// ================================================================
// 冷热数据分离
// ================================================================
struct HotData {
    alignas(64) std::atomic<int> active_count{0};
    alignas(64) std::atomic<long long> last_update{0};
};

struct ColdData {
    std::string description;
    std::string config_json;
    std::vector<int> history;
};

struct OptimizedObject {
    HotData hot;   // 频繁访问，独占 cache line(s)
    ColdData cold; // 偶尔访问，不污染 hot 数据的缓存
};

void demo_hot_cold_splitting() {
    std::cout << "\n=== 冷热数据分离 ===\n";
    std::cout << "  sizeof(HotData) = " << sizeof(HotData) << "\n";
    std::cout << "  sizeof(ColdData) = " << sizeof(ColdData) << "\n";
    std::cout << "  sizeof(OptimizedObject) = "
              << sizeof(OptimizedObject) << "\n";
    std::cout << "  原则: 频繁访问的字段放前面(cache line 开头)，\n";
    std::cout << "        冷数据放后面或不常访问的位置\n";
}

// ================================================================
// Padding 最佳实践
// ================================================================
void demo_padding_best_practices() {
    std::cout << "\n=== Padding 最佳实践 ===\n";

    // 实践 1: 用 alignas 确保对齐
    struct alignas(64) CacheLinePadded {
        int critical_data;
    };
    std::cout << "  alignas(64) struct size: "
              << sizeof(CacheLinePadded) << " (应为 64)\n";

    // 实践 2: 手动 padding（跨编译器兼容）
    struct ManualPad {
        int value;
        char padding[60];
    };
    static_assert(sizeof(ManualPad) == 64, "Manual padding failed");
    std::cout << "  Manual padding size: " << sizeof(ManualPad)
              << " (应为 64)\n";

    // 实践 3: C++17 标准常量
#ifdef __cpp_lib_hardware_interference_size
    std::cout << "  destructive_interference_size: "
              << std::hardware_destructive_interference_size << "\n";
    std::cout << "  constructive_interference_size:  "
              << std::hardware_constructive_interference_size << "\n";
#else
    std::cout << "  destructive_interference_size: 不可用 (< GCC 12)\n";
    std::cout << "  constructive_interference_size:  不可用\n";
#endif

    std::cout << "\n  优化检查清单:\n";
    std::cout << "  [ ] 热点字段对齐到 cache line\n";
    std::cout << "  [ ] 不同线程写入的字段在不同 cache line\n";
    std::cout << "  [ ] 只读共享数据可以紧凑排列\n";
    std::cout << "  [ ] 跨 NUMA 的数据访问最小化\n";
}

int main() {
    demo_aos_vs_soa();
    demo_hot_cold_splitting();
    demo_padding_best_practices();

    return 0;
}
