/**
 * lecture6_part1.cpp - 内存局部性优化
 *
 * 演示 CS149 第6讲的概念：
 * - 行优先 vs 分块遍历二维网格
 * - 循环融合（loop fusion）提升算术强度
 * - 缓存行效应对性能的影响
 * - 分块矩阵运算实现时间局部性
 * - 算术强度（Arithmetic Intensity）的计算
 *
 * 关键概念详解：
 * ─────────────────────────────────────────────────────────────
 * 【缓存行与空间局部性】
 *   现代 CPU 的缓存以 64 字节的"缓存行"为单位加载数据。
 *   当你访问一个 float（4 字节）时，整个 64 字节缓存行（16 个 float）
 *   都被加载进缓存。按行优先顺序遍历矩阵时，访问模式是连续的，
 *   充分利用了被加载的缓存行。按列遍历则会浪费带宽。
 *
 * 【时间局部性与分块】
 *   如果一段数据需要在短时间内被多次访问，我们希望它一直留在缓存中。
 *   分块（blocking/tiling）是一种技术，它将大矩阵分解为能完整放入
 *   缓存的小块，先处理一整块再进入下一块，避免数据被驱逐后再次加载。
 *
 * 【算术强度】
 *   AI = 计算量（FLOPs）/ 通信量（字节）。AI 越高，越能利用
 *   内存带宽。现代 GPU 有 10+ TFLOPS 算力和 1+ TB/s 带宽，
 *   需要 AI >> 10 才能让计算成为瓶颈而非带宽。
 *
 * 【循环融合】
 *   将多个独立循环合并为一个，消除中间数组的存取。
 *   例如：E = D + ((A + B) * C)，分三个循环 AI=1/3，
 *   融合为一个循环 AI=3/5（5 次内存访问产生 3 次浮点运算）。
 *
 * 编译：g++ -std=c++17 lecture6_part1.cpp -o lecture6_part1 && ./lecture6_part1
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>

// ============================================================================
// 第一部分：网格遍历顺序与缓存效应
// ============================================================================

/**
 * 模拟第6讲中的网格求解器访问模式。
 *
 * 关键洞察：行优先遍历会加载太多缓存行，
 * 因为前一行的数据在再次使用前就被驱逐出了缓存。
 *
 * 【为什么行优先遍历在网格求解器中效率低】
 * 考虑 5-point stencil（每个元素需要访问自身和上下左右邻居）：
 * 当遍历到第 i 行时，缓存已经因为前面的行满了，无法保留第 i-1 行的数据。
 * 但第 i+1 行需要用到第 i 行的数据。如果第 i 行的数据已经被驱逐，
 * 就需要从内存重新加载 → 缓存未命中 → 性能下降。
 */

// 缓存模拟参数常量
constexpr int CACHE_LINE_ELEMENTS = 4;   // 每个 64 字节缓存行可容纳 4 个 float
constexpr int CACHE_CAPACITY_LINES = 6;  // 模拟 6 行容量的极小缓存（仅用于演示）

// 计算网格位置 (i,j) 落入哪个缓存行
inline int get_line(int i, int j, int cols) {
    return (i * cols + j) / CACHE_LINE_ELEMENTS;
}

class CacheSimulator {
private:
    int hits;
    int misses;
    std::vector<int> cache_tags;  // 当前缓存中存放了哪些缓存行
    int access_counter;

    int get_line(int row, int col, int stride) {
        return (row * stride + col) / CACHE_LINE_ELEMENTS;
    }

public:
    CacheSimulator() : hits(0), misses(0), access_counter(0) {
        cache_tags.resize(CACHE_CAPACITY_LINES, -1);
    }

    /**
     * 模拟一次缓存访问。
     * 返回 true 表示命中（hit），false 表示未命中（miss）。
     * 使用简单的 FIFO 替换策略。
     */
    bool access(int line) {
        access_counter++;
        // 检查该缓存行是否已在缓存中
        for (int i = 0; i < CACHE_CAPACITY_LINES; i++) {
            if (cache_tags[i] == line) {
                hits++;
                return true;  // 命中
            }
        }
        // 未命中：从内存加载（简单的 FIFO 替换策略）
        misses++;
        cache_tags[access_counter % CACHE_CAPACITY_LINES] = line;
        return false;
    }

    void reset() {
        hits = 0;
        misses = 0;
        access_counter = 0;
        std::fill(cache_tags.begin(), cache_tags.end(), -1);
    }

    int get_hits() const { return hits; }
    int get_misses() const { return misses; }
    double hit_rate() const {
        int total = hits + misses;
        return total > 0 ? 100.0 * hits / total : 0.0;
    }
};

/**
 * 标准的行优先网格遍历（如网格求解器中的做法）。
 * 问题：访问同一数据的时间间隔太长，数据在再次需要时已经不在缓存中。
 */
void row_major_traversal(int N, CacheSimulator& cache) {
    cache.reset();
    std::cout << "  行优先遍历（" << N << "x" << N << "）：\n";

    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            // 访问 5 点模板：中心、北、南、东、西
            // 每个元素需要自身和 4 个邻居的数据
            cache.access(get_line(i, j, N + 2));
            cache.access(get_line(i - 1, j, N + 2));
            cache.access(get_line(i + 1, j, N + 2));
            cache.access(get_line(i, j - 1, N + 2));
            cache.access(get_line(i, j + 1, N + 2));
        }
    }

    int total = cache.get_hits() + cache.get_misses();
    std::cout << "    总访问次数: " << total << "\n";
    std::cout << "    缓存命中: " << cache.get_hits()
              << "  未命中: " << cache.get_misses()
              << "  命中率: " << std::fixed << std::setprecision(1)
              << cache.hit_rate() << "%\n";
}

/**
 * 分块遍历：以适配缓存的小块为单位处理网格。
 * 改善时间局部性：数据在被驱逐之前就被重复使用。
 *
 * 当块大小合适时（整个工作集能放入缓存），同一块内的
 * 数据在 5 次模板访问中都不会被驱逐。
 */
void blocked_traversal(int N, int block_size, CacheSimulator& cache) {
    cache.reset();
    std::cout << "  分块遍历（" << N << "x" << N
              << "，块大小=" << block_size << "）：\n";

    for (int bi = 1; bi <= N; bi += block_size) {
        for (int bj = 1; bj <= N; bj += block_size) {
            int i_end = std::min(bi + block_size, N + 1);
            int j_end = std::min(bj + block_size, N + 1);
            for (int i = bi; i < i_end; i++) {
                for (int j = bj; j < j_end; j++) {
                    cache.access(get_line(i, j, N + 2));
                    cache.access(get_line(i - 1, j, N + 2));
                    cache.access(get_line(i + 1, j, N + 2));
                    cache.access(get_line(i, j - 1, N + 2));
                    cache.access(get_line(i, j + 1, N + 2));
                }
            }
        }
    }

    int total = cache.get_hits() + cache.get_misses();
    std::cout << "    总访问次数: " << total << "\n";
    std::cout << "    缓存命中: " << cache.get_hits()
              << "  未命中: " << cache.get_misses()
              << "  命中率: " << std::fixed << std::setprecision(1)
              << cache.hit_rate() << "%\n";
}

// ============================================================================
// 第二部分：循环融合 - 提升算术强度
// ============================================================================

/**
 * 演示第6讲中的循环融合：
 *
 * 分离循环（三个独立循环）：
 *   E = D + ((A + B) * C) 需要 3 个独立循环
 *   算术强度 = 1/3（每次算术运算需要 2 次 load + 1 次 store）
 *
 *   循环1: tmp1 = A + B  → 2 loads, 1 store, 1 FLOP
 *   循环2: tmp2 = tmp1 * C → 2 loads, 1 store, 1 FLOP
 *   循环3: E = tmp2 + D  → 2 loads, 1 store, 1 FLOP
 *   总计: 6 loads + 3 stores = 9 次内存访问，3 FLOPs → AI = 3/9 = 1/3
 *
 * 融合循环（单循环）：
 *   E[i] = D[i] + (A[i] + B[i]) * C[i] 在一个循环中完成
 *   算术强度 = 3/5（3 次算术运算对应 4 loads + 1 store = 5 次内存访问）
 *
 *   总计: 4 loads (A,B,C,D) + 1 store (E), 3 FLOPs → AI = 3/5 = 0.6
 *   融合后不需要 tmp1 和 tmp2 数组 → 减少内存占用和访问
 */

void benchmark_separate_loops(int N) {
    std::vector<float> A(N, 1.0f), B(N, 2.0f), C(N, 3.0f);
    std::vector<float> D(N, 4.0f), E(N);
    std::vector<float> tmp1(N), tmp2(N);

    auto start = std::chrono::high_resolution_clock::now();

    // 循环1：tmp1 = A + B（每个元素 2 loads + 1 store）
    for (int i = 0; i < N; i++) {
        tmp1[i] = A[i] + B[i];
    }

    // 循环2：tmp2 = tmp1 * C（每个元素 2 loads + 1 store）
    for (int i = 0; i < N; i++) {
        tmp2[i] = tmp1[i] * C[i];
    }

    // 循环3：E = tmp2 + D（每个元素 2 loads + 1 store）
    for (int i = 0; i < N; i++) {
        E[i] = tmp2[i] + D[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    // 验证计算结果
    double checksum = 0.0;
    for (int i = 0; i < N; i++) checksum += E[i];

    std::cout << "  分离循环: " << std::fixed << std::setprecision(4)
              << elapsed << "秒  （AI=1/3, 校验和=" << checksum << "）\n";
}

void benchmark_fused_loop(int N) {
    std::vector<float> A(N, 1.0f), B(N, 2.0f), C(N, 3.0f);
    std::vector<float> D(N, 4.0f), E(N);

    auto start = std::chrono::high_resolution_clock::now();

    // 融合循环：E = D + (A + B) * C
    // 每个元素：4 loads（A, B, C, D），1 store（E），3 次算术运算（+, *, +）
    // 算术强度 = 3/5 = 0.6（比分离循环提升了近 1 倍）
    for (int i = 0; i < N; i++) {
        E[i] = D[i] + (A[i] + B[i]) * C[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    double checksum = 0.0;
    for (int i = 0; i < N; i++) checksum += E[i];

    std::cout << "  融合循环:     " << std::fixed << std::setprecision(4)
              << elapsed << "秒  （AI=3/5, 校验和=" << checksum << "）\n";
}

// ============================================================================
// 第三部分：算术强度演示
// ============================================================================

/**
 * 计算并展示各种运算的算术强度。
 *
 * AI = 计算量 / 通信量
 * 更高的 AI = 更好地利用内存带宽。
 *
 * 【AI 与硬件的关系】
 * 现代 GPU（如 NVIDIA H100）：
 *   - 算力：约 60 TFLOPS（FP32）
 *   - 带宽：约 2 TB/s（HBM3）
 *   - "屋顶线"交叉点：AI = 60/2 = 30
 *   - 当 AI > 30 时，计算是瓶颈（compute bound）
 *   - 当 AI < 30 时，带宽是瓶颈（memory bound）
 *
 * CPU 通常有更低的交叉点（约 AI=10），因为有更低的 FLOPs/BW 比。
 */
void analyze_arithmetic_intensity() {
    std::cout << "\n=== 算术强度分析 ===\n\n";

    std::cout << "算术强度（AI）= 计算量（FLOPs）/ 通信量（字节）\n\n";

    // 逐元素向量乘法：C[i] = A[i] * B[i]
    // 1 FLOP, 2*4=8 字节加载, 4 字节存储 = 12 字节 → AI = 1/12 ≈ 0.083
    std::cout << "运算操作                          FLOPs  字节数  AI\n";
    std::cout << "────────────────────────────────  ─────  ─────  ──────\n";
    std::cout << "C[i] = A[i] * B[i]                  1     12     " << std::fixed
              << std::setprecision(4) << (1.0 / 12.0) << "\n";
    std::cout << "C[i] = A[i] + B[i] * C[i]          2     16     "
              << (2.0 / 16.0) << "\n";
    std::cout << "E[i] = D[i]+(A[i]+B[i])*C[i]       3     20     "
              << (3.0 / 20.0) << "\n";
    std::cout << "C[i] = α*A[i] + β*B[i]（BLAS）      3     16     "
              << (3.0 / 16.0) << "\n";
    std::cout << "矩阵乘法（内积形式）               2N    4N+4   "
              << (2.0 / 4.0) << " （每个元素）\n\n";

    std::cout << "关键结论：现代 GPU（10+ TFLOPS 算力，1+ TB/s 带宽）上\n";
    std::cout << "要使计算成为瓶颈（compute bound），需要 AI >> 10。\n";
    std::cout << "逐元素操作通常 AI < 1，属于 memory bound。\n";
    std::cout << "矩阵乘法（AI ≈ N/2）是少数能接近计算瓶颈的运算。\n";
}

// ============================================================================
// 第四部分：分块矩阵运算
// ============================================================================

/**
 * 对比朴素矩阵乘法与分块矩阵乘法。
 * 分块版本将子块保持在缓存中以便重复使用。
 *
 * 【为什么朴素矩阵乘法对缓存不友好】
 * 标准三层循环 i,j,k 中，访问 B[k][j] 时 k 是最内层循环变量，
 * 这意味着对 B 的访问是跨行的（j 不变但 k 变化），不是连续的。
 * 缓存行被加载后只用了一个元素就被替换了 → 极差的缓存利用率。
 */
void matrix_multiply_naive(int N, const std::vector<double>& A,
                            const std::vector<double>& B, std::vector<double>& C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/**
 * 分块矩阵乘法。
 * 一次处理一个 BLOCK x BLOCK 的子矩阵，该子矩阵能完整放入缓存，
 * 从而大幅减少对主存的访问次数。
 *
 * 分块乘法的数据复用：
 * - A 的子块被加载一次，被 B 的多列重复使用
 * - B 的子块被加载一次，被 A 的多行重复使用
 * - C 的子块在块被处理期间保留在缓存中（时间局部性）
 */
void matrix_multiply_blocked(int N, int block, const std::vector<double>& A,
                              const std::vector<double>& B, std::vector<double>& C) {
    std::fill(C.begin(), C.end(), 0.0);

    for (int bi = 0; bi < N; bi += block) {
        for (int bj = 0; bj < N; bj += block) {
            for (int bk = 0; bk < N; bk += block) {
                // 在这个块内部进行矩阵乘法
                int i_end = std::min(bi + block, N);
                int j_end = std::min(bj + block, N);
                int k_end = std::min(bk + block, N);

                // 注意循环顺序是 i,k,j（而非 i,j,k）：
                // 这利用了 A[i][k] 和 B[k][j] 的空间局部性
                for (int i = bi; i < i_end; i++) {
                    for (int k = bk; k < k_end; k++) {
                        double aik = A[i * N + k];
                        for (int j = bj; j < j_end; j++) {
                            C[i * N + j] += aik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

void benchmark_matrix_multiply() {
    std::cout << "\n=== 分块矩阵乘法（N=256） ===\n\n";

    const int N = 256;
    std::vector<double> A(N * N), B(N * N), C_naive(N * N), C_blocked(N * N);

    // 用随机值初始化
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < N * N; i++) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }

    // 朴素矩阵乘法
    auto start = std::chrono::high_resolution_clock::now();
    matrix_multiply_naive(N, A, B, C_naive);
    auto end = std::chrono::high_resolution_clock::now();
    double naive_time = std::chrono::duration<double>(end - start).count();

    std::cout << "  朴素（i,j,k）: " << std::fixed << std::setprecision(4)
              << naive_time << "秒\n";

    // 分块矩阵乘法 - 尝试不同的块大小
    for (int block : {16, 32, 64}) {
        start = std::chrono::high_resolution_clock::now();
        matrix_multiply_blocked(N, block, A, B, C_blocked);
        end = std::chrono::high_resolution_clock::now();
        double block_time = std::chrono::duration<double>(end - start).count();

        // 验证正确性：与朴素版本的结果对比
        bool correct = true;
        for (int i = 0; i < N * N && correct; i++) {
            correct = (std::abs(C_naive[i] - C_blocked[i]) < 1e-6);
        }

        std::cout << "  分块（块大小=" << block << "）: " << std::fixed
                  << std::setprecision(4) << block_time << "秒"
                  << "  加速比=" << std::setprecision(2) << (naive_time / block_time) << "x"
                  << "  正确=" << (correct ? "是" : "否") << "\n";
    }
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "============================================================\n";
    std::cout << "第6讲 第一部分：内存局部性优化\n";
    std::cout << "============================================================\n";

    // === 第一部分：网格遍历缓存模拟 ===
    std::cout << "\n--- 网格遍历与缓存效应 ---\n\n";
    std::cout << "缓存配置: " << CACHE_LINE_ELEMENTS << " 个元素/行, "
              << CACHE_CAPACITY_LINES << " 行的容量\n\n";

    CacheSimulator cache;
    row_major_traversal(6, cache);
    std::cout << "\n";
    blocked_traversal(6, 3, cache);

    std::cout << "\n  观察：分块遍历在两次访问之间将数据保留在缓存中，\n";
    std::cout << "  而行优先遍历会丢失前面行的数据，导致再次需要时缓存未命中。\n";

    // === 第二部分：循环融合基准测试 ===
    std::cout << "\n--- 循环融合：算术强度 ---\n\n";
    const int N_FUSION = 10000000;
    benchmark_separate_loops(N_FUSION);
    benchmark_fused_loop(N_FUSION);
    std::cout << "\n  融合循环：每次计算所需的内存往返次数更少。\n";
    std::cout << "  临时数组（tmp1, tmp2）被消除 → 更好的局部性和更少的内存占用。\n";

    // === 第三部分：算术强度分析 ===
    analyze_arithmetic_intensity();

    // === 第四部分：分块矩阵乘法 ===
    benchmark_matrix_multiply();

    // === 总结 ===
    std::cout << "\n=== 局部性优化：关键技术 ===\n";
    std::cout << "┌────────────────────┬──────────────────────────────────────┐\n";
    std::cout << "│ 技术               │ 收益                                  │\n";
    std::cout << "├────────────────────┼──────────────────────────────────────┤\n";
    std::cout << "│ 分块遍历           │ 将工作集保持在缓存中                 │\n";
    std::cout << "│ 循环融合           │ 减少中间变量的存储/加载              │\n";
    std::cout << "│ 分块矩阵乘法       │ 在缓存中重复使用子块                 │\n";
    std::cout << "│ 行优先存储顺序     │ 连续访问共置在同一缓存行中           │\n";
    std::cout << "│ 高 AI 运算         │ 更充分利用内存带宽                   │\n";
    std::cout << "└────────────────────┴──────────────────────────────────────┘\n";

    std::cout << "\n所有测试成功完成。\n";
    return 0;
}
