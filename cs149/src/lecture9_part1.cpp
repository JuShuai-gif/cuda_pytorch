// lecture9_part1.cpp
// Stanford CS149, 第9讲：高效评估深度神经网络（DNN）
// 第一部分：矩阵乘法 — 朴素、分块、以及分层分块实现
//
// 本文件实现了多种逐步优化的稠密矩阵乘法（GEMM）变体：
//   1. 朴素三重循环（算术强度低，带宽受限）
//   2. 单级分块（缓存优化，提升数据复用）
//   3. 分层分块（同时利用L1和L2缓存层级）
//   4. 有利于SIMD向量化的转置预处理版本（2种变体）
//   5. 寄存器分块/微内核方案（将小块数据保持在寄存器中）
//
// 核心概念说明：
//   - 算术强度（Arithmetic Intensity, AI）：FLOPs / 访存字节数。AI越高，越接近计算受限。
//   - 分块（Blocking/Tiling）：将大矩阵划分为能放入缓存的小块，
//     使得小块内的数据在被替换出缓存之前能被多次复用，
//     从而大幅提高算术强度。
//   - L1缓存（~32KB）：CPU中最快的缓存层级，延迟最低。
//   - L2缓存（~256KB）：速度略慢于L1但容量更大。
//   - 寄存器分块：最内层循环将数据保存在CPU寄存器中（编译器-O2优化），
//     避免额外的load/store指令开销。
//   - SIMD向量化：预转置矩阵B使内层循环访存变为连续步长（步长=1），
//     有利于SIMD指令（如AVX-512）同时对多个数据进行运算。
//
// 所有变体计算的都是 C = A * B（即 C += A * B 的累加形式）
//
// 编译命令：g++ -std=c++17 -O2 -pthread lecture9_part1.cpp -o lecture9_part1
// 运行命令：./lecture9_part1

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <cassert>

// ============================================================================
// 矩阵辅助类 —— 封装矩阵数据的存储与基本操作
// 采用行优先（row-major）布局：data[r * cols + c]
// ============================================================================

class Matrix {
public:
    std::vector<float> data;  // 一维扁平化存储矩阵元素
    size_t rows, cols;        // 行数、列数

    Matrix(size_t r, size_t c) : rows(r), cols(c), data(r * c, 0.0f) {}

    // 按行优先布局访问元素
    float& at(size_t r, size_t c) { return data[r * cols + c]; }
    float  at(size_t r, size_t c) const { return data[r * cols + c]; }

    // 用指定值填充整个矩阵
    void fill(float val) {
        std::fill(data.begin(), data.end(), val);
    }

    // 用确定性伪随机值填充矩阵（便于验证正确性）
    void randomize(float scale = 1.0f) {
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = static_cast<float>(i % 100) * scale;
        }
    }

    // 比较两个矩阵是否近似相等（用于验证不同实现的正确性）
    bool equals(const Matrix& other, float tolerance = 0.01f) const {
        if (rows != other.rows || cols != other.cols) return false;
        for (size_t i = 0; i < data.size(); i++) {
            if (std::abs(data[i] - other.data[i]) > tolerance) return false;
        }
        return true;
    }
};

// 打印矩阵的子区域（默认显示 6x6），用于可视化验证
void printMatrix(const std::string& name, const Matrix& m,
                 size_t maxRows = 6, size_t maxCols = 6)
{
    std::cout << name << " (" << m.rows << "x" << m.cols << "):\n";
    for (size_t r = 0; r < std::min(m.rows, maxRows); r++) {
        std::cout << "  ";
        for (size_t c = 0; c < std::min(m.cols, maxCols); c++) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(1)
                      << m.at(r, c);
        }
        if (m.cols > maxCols) std::cout << " ...";
        std::cout << "\n";
    }
    if (m.rows > maxRows) std::cout << "  ...\n";
}

// ============================================================================
// 计时工具 —— 测量函数的执行时间（毫秒精度）
// ============================================================================

template<typename Func>
double timeIt(Func f, const std::string& label) {
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  " << label << ": " << std::fixed << std::setprecision(2)
              << ms << " 毫秒\n";
    return ms;
}

// ============================================================================
// 版本1：朴素矩阵乘法（Naive GEMM）
// 计算公式：C[j][i] += A[j][k] * B[k][i]
//
// 存在的问题分析：
//   - 无时间局部性（Temporal Locality）：
//     内层循环每次迭代都需要从缓存中重新加载A和B的元素，
//     A和B的元素在缓存中被反复加载，没有被有效复用。
//   - 算术强度（AI）低：
//     AI = 1次浮点运算 / (2次加载 + 0次存储到缓存)
//        ≈ 每次内层循环迭代约0.5 FLOPs/Byte
//   - 结果：此实现通常是带宽受限（bandwidth-bound）的，
//     即性能瓶颈在于内存带宽而非计算能力。
// ============================================================================

void gemmNaive(const Matrix& A, const Matrix& B, Matrix& C) {
    assert(A.cols == B.rows);
    assert(C.rows == A.rows && C.cols == B.cols);

    // 遍历C的每个元素，计算其值为A对应行与B对应列的点积
    for (size_t j = 0; j < A.rows; j++) {
        for (size_t i = 0; i < B.cols; i++) {
            float sum = C.at(j, i);
            for (size_t k = 0; k < A.cols; k++) {
                sum += A.at(j, k) * B.at(k, i);
            }
            C.at(j, i) = sum;
        }
    }
}

// ============================================================================
// 版本2：单级分块（Blocked GEMM）
// 思路：将A和B划分为多个小块（blocks/tiles），在块内完成部分C的计算。
// 关键优势：当块大小足够小时，A和B的子块能同时驻留在缓存中，
// 从而在每个块的计算期间充分利用缓存内的数据复用。
//
// 分块大小选择原则：
//   - 目标：将 3 * BLOCKSIZE^2 * sizeof(float) 放入L1缓存
//   - L1缓存大小：~32KB → BLOCKSIZE^2 * 4字节 * 3 ≈ 32KB → BLOCKSIZE ≈ 52
//   - 实际常用值：L1用32-64，L2用128-256
//
// 算术强度提升原理：
//   - 在块内计算中，A的每个元素被复用 BS_I 次（对应该块内C的不同列），
//     B的每个元素被复用 BS_J 次（对应该块内C的不同行）
//   - 因此整体算术强度 O(BS)，显著高于朴素版本
// ============================================================================

void gemmBlocked(const Matrix& A, const Matrix& B, Matrix& C,
                 size_t BS_J, size_t BS_I, size_t BS_K)
{
    assert(A.cols == B.rows);
    assert(C.rows == A.rows && C.cols == B.cols);

    // 矩阵维度命名：C 是 M×N，A 是 M×K，B 是 K×N
    size_t M = A.rows;
    size_t N = B.cols;
    size_t K = A.cols;

    // 外层三层循环：遍历所有分块
    for (size_t jb = 0; jb < M; jb += BS_J) {
        size_t jEnd = std::min(jb + BS_J, M);
        for (size_t ib = 0; ib < N; ib += BS_I) {
            size_t iEnd = std::min(ib + BS_I, N);
            for (size_t kb = 0; kb < K; kb += BS_K) {
                size_t kEnd = std::min(kb + BS_K, K);

                // 内层：计算C块 C[jb:jEnd][ib:iEnd] 的部分结果
                // 注意此处第三个维度（K方向）也是分块的，
                // 所以C中的每个元素会在多个kb块中被累加更新
                for (size_t j = jb; j < jEnd; j++) {
                    for (size_t i = ib; i < iEnd; i++) {
                        float sum = 0.0f;
                        for (size_t k = kb; k < kEnd; k++) {
                            sum += A.at(j, k) * B.at(k, i);
                        }
                        C.at(j, i) += sum;
                    }
                }
            }
        }
    }
}

// ============================================================================
// 版本3：分层分块（Hierarchical Blocking, L1 + L2 双缓存层级利用）
// 思想：现代CPU有多个缓存层级（L1、L2、L3），各层容量和速度不同。
// 分层分块在外层使用较大的L2块，在内层嵌套更小的L1块。
//
// 典型尺寸选择：
//   - L2缓存 ~256KB → L2分块大小 ≈ 128
//   - L1缓存 ~32KB  → L1分块大小 ≈ 32
//
// 优势：
//   - L2块足够大，可以在L2缓存中容纳大块A和B的子矩阵
//   - L1块嵌套在L2块内部，保证最内层计算的数据始终留在最快的L1缓存中
//   - 实际上形成了一个6重循环嵌套，有效利用了缓存层级结构
//
// 与版本2的区别：版本2只有一层分块，而版本3有两层嵌套分块，
// 分别对应L2和L1缓存的容量约束。
// ============================================================================

void gemmHierarchical(const Matrix& A, const Matrix& B, Matrix& C,
                      size_t L2_J, size_t L2_I, size_t L2_K,
                      size_t L1_J, size_t L1_I, size_t L1_K)
{
    assert(A.cols == B.rows);
    assert(C.rows == A.rows && C.cols == B.cols);

    size_t M = A.rows;
    size_t N = B.cols;
    size_t K = A.cols;

    // L2级别分块（外层循环）—— 数据块大小适配L2缓存
    for (size_t jb2 = 0; jb2 < M; jb2 += L2_J) {
        size_t jEnd2 = std::min(jb2 + L2_J, M);
        for (size_t ib2 = 0; ib2 < N; ib2 += L2_I) {
            size_t iEnd2 = std::min(ib2 + L2_I, N);
            for (size_t kb2 = 0; kb2 < K; kb2 += L2_K) {
                size_t kEnd2 = std::min(kb2 + L2_K, K);

                // L1级别分块（嵌套在L2块内部）—— 数据块大小适配L1缓存
                for (size_t jb1 = jb2; jb1 < jEnd2; jb1 += L1_J) {
                    size_t jEnd1 = std::min(jb1 + L1_J, jEnd2);
                    for (size_t ib1 = ib2; ib1 < iEnd2; ib1 += L1_I) {
                        size_t iEnd1 = std::min(ib1 + L1_I, iEnd2);
                        for (size_t kb1 = kb2; kb1 < kEnd2; kb1 += L1_K) {
                            size_t kEnd1 = std::min(kb1 + L1_K, kEnd2);

                            // 最内层计算核：在L1块内执行乘加运算
                            for (size_t j = jb1; j < jEnd1; j++) {
                                for (size_t i = ib1; i < iEnd1; i++) {
                                    float sum = 0.0f;
                                    for (size_t k = kb1; k < kEnd1; k++) {
                                        sum += A.at(j, k) * B.at(k, i);
                                    }
                                    C.at(j, i) += sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// 版本4：分块 + 预转置B矩阵（改善访存模式，促进SIMD向量化）
//
// 关键优化思路：
//   - 将B转置为BT（BT[k][i] = B[k][i]，也可以从B[i][k]角度理解），
//     使得内层循环中对BT的访问是沿着i方向连续的行优先步长访问
//   - 这样A的某行和BT对应的某行都是连续存储的（步长=1），
//     对CPU的预取器和SIMD向量化十分友好
//   - 预转置的一次性开销 O(N*K) 可以被后续大量计算所摊薄（amortize）
//
// 本节课中讨论的场景：当i维度较小且需要SIMD向量化连续数据时使用此方法。
// 注意：transpose本身带来了额外存储开销和一次全矩阵遍历。
// ============================================================================

void gemmBlockedTranspose(const Matrix& A, const Matrix& B, Matrix& C,
                          size_t BS_J, size_t BS_K)
{
    assert(A.cols == B.rows);
    assert(C.rows == A.rows && C.cols == B.cols);

    size_t M = A.rows;
    size_t N = B.cols;
    size_t K = A.cols;
    size_t BS_I = N;  // i维度的全部范围（未分块）

    // 预转置 B → BT：BT[k][i] = B[k][i]
    // BT的维度为 K × N
    Matrix BT(K, N);
    for (size_t i = 0; i < N; i++) {
        for (size_t k = 0; k < K; k++) {
            BT.at(k, i) = B.at(k, i);
        }
    }

    for (size_t jb = 0; jb < M; jb += BS_J) {
        size_t jEnd = std::min(jb + BS_J, M);
        for (size_t kb = 0; kb < K; kb += BS_K) {
            size_t kEnd = std::min(kb + BS_K, K);

            for (size_t j = 0; j < jEnd; j++) {
                for (size_t i = 0; i < N; i++) {
                    float sum = 0.0f;
                    // 点积：A的某行 · BT的某行（两者都是连续存储的）
                    // 这对于SIMD向量化极其有利：可以一次加载多个元素
                    for (size_t k = kb; k < kEnd; k++) {
                        sum += A.at(j, k) * BT.at(k, i);
                    }
                    C.at(j, i) += sum;
                }
            }
        }
    }
}

// ============================================================================
// 版本5：微内核/寄存器分块（Register Blocking / Micro-Kernel）
//
// 概念说明：
//   在高性能GEMM库（如BLIS、cuBLAS）中，最内层循环操作的是一个足够小的
//   子块，其大小刚好能放入CPU的寄存器文件中（例如4x4的C子块，加上4xK的A
//   行块和Kx4的B列块）。这被称为"微内核"（micro-kernel）。
//
// 微内核的优势：
//   - 将C中的元素只加载一次、只存储一次，中间状态全部保留在寄存器中
//   - 最大限度地减少load/store指令数量
//   - 编译器在-O2及以上优化等级下，可能将小块局部变量分配给寄存器
//
// 本实现模拟了一个 4x4 微内核（MR=4, NR=4），配合KC维度的分块。
// 参数说明：
//   - MR (Micro-Rows)：C块的行数
//   - NR (Micro-Cols)：C块的列数
//   - KC (K-Common)：沿K维度的分块大小，决定了A和B的某个维度在寄存器中的驻留
// ============================================================================

void gemmMicroKernel(const Matrix& A, const Matrix& B, Matrix& C,
                     size_t MR, size_t NR, size_t KC)
{
    assert(A.cols == B.rows);
    assert(C.rows == A.rows && C.cols == B.cols);

    size_t M = A.rows;
    size_t N = B.cols;
    size_t K = A.cols;

    // 外部分块循环——为微内核提供合适的输入块
    for (size_t jb = 0; jb < M; jb += MR) {
        size_t jEnd = std::min(jb + MR, M);
        for (size_t ib = 0; ib < N; ib += NR) {
            size_t iEnd = std::min(ib + NR, N);
            for (size_t kb = 0; kb < K; kb += KC) {
                size_t kEnd = std::min(kb + KC, K);

                // 微内核：计算 MR × NR 大小的C子块
                // 使用 MR × KC 的A子块 和 KC × NR 的B子块
                // 这些子块的数据，编译器（在-O2下）会尽量分配在寄存器中
                for (size_t j = jb; j < jEnd; j++) {
                    for (size_t i = ib; i < iEnd; i++) {
                        float c_accum = C.at(j, i);  // C元素仅加载一次
                        for (size_t k = kb; k < kEnd; k++) {
                            c_accum += A.at(j, k) * B.at(k, i);
                        }
                        C.at(j, i) = c_accum;  // C元素在计算结束后才存储一次
                    }
                }
            }
        }
    }
}

// ============================================================================
// 并行化GEMM —— 对j循环（C的行维度）进行多线程并行化
//
// 并行化策略：
//   - C的不同行之间没有数据依赖（read-write冲突），因此可以安全并行
//   - 每个线程负责处理C的一段连续行
//   - 使用std::thread进行OS级线程创建和join
//
// 适用于多核CPU场景；在GPU上，这种并行度可以进一步扩展到数千个线程。
// ============================================================================

void gemmParallel(const Matrix& A, const Matrix& B, Matrix& C,
                  size_t numThreads)
{
    size_t M = A.rows;
    size_t N = B.cols;
    size_t K = A.cols;

    // 使用带分块的版本，在外层j循环上并行化
    const size_t BS = 64;

    std::vector<std::thread> workers;
    for (size_t t = 0; t < numThreads; t++) {
        workers.emplace_back([&A, &B, &C, M, N, K, BS, t, numThreads]() {
            // 计算当前线程负责的j块范围
            size_t chunkSize = ((M + BS - 1) / BS + numThreads - 1) / numThreads;
            size_t jbStart = t * chunkSize * BS;
            size_t jbEnd   = std::min((t + 1) * chunkSize * BS, M);

            for (size_t jb = jbStart; jb < jbEnd; jb += BS) {
                size_t jEnd = std::min(jb + BS, M);
                for (size_t ib = 0; ib < N; ib += BS) {
                    size_t iEnd = std::min(ib + BS, N);
                    for (size_t kb = 0; kb < K; kb += BS) {
                        size_t kEnd = std::min(kb + BS, K);
                        for (size_t j = jb; j < jEnd; j++) {
                            for (size_t i = ib; i < iEnd; i++) {
                                float sum = 0.0f;
                                for (size_t k = kb; k < kEnd; k++) {
                                    sum += A.at(j, k) * B.at(k, i);
                                }
                                C.at(j, i) += sum;
                            }
                        }
                    }
                }
            }
        });
    }
    for (auto& w : workers) w.join();
}

// ============================================================================
// 主函数 —— 正确性验证 + 性能对比 + 算术强度分析
// ============================================================================

int main()
{
    std::cout << "==================================================\n";
    std::cout << "第9讲 第一部分：矩阵乘法优化方案对比\n";
    std::cout << "==================================================\n\n";

    // ---- 小矩阵正确性验证测试 ----
    {
        std::cout << "--- 正确性验证（小矩阵） ---\n";

        size_t M = 8, K = 6, N = 8;
        Matrix A(M, K);  A.randomize();
        Matrix B(K, N);  B.randomize();
        Matrix C1(M, N); C1.fill(0);
        Matrix C2(M, N); C2.fill(0);
        Matrix C3(M, N); C3.fill(0);
        Matrix C4(M, N); C4.fill(0);
        Matrix C5(M, N); C5.fill(0);

        // 使用朴素版本计算参考结果
        gemmNaive(A, B, C1);

        // 逐一对比验证各个优化版本的正确性
        gemmBlocked(A, B, C2, 4, 4, 4);
        std::cout << "  分块版本：       " << (C2.equals(C1) ? "通过" : "未通过") << "\n";

        gemmHierarchical(A, B, C3, 6, 6, 6, 3, 3, 3);
        std::cout << "  分层分块版本：   " << (C3.equals(C1) ? "通过" : "未通过") << "\n";

        gemmBlockedTranspose(A, B, C4, 4, 4);
        std::cout << "  分块+转置版本：  " << (C4.equals(C1) ? "通过" : "未通过") << "\n";

        gemmMicroKernel(A, B, C5, 4, 4, 4);
        std::cout << "  微内核版本：     " << (C5.equals(C1) ? "通过" : "未通过") << "\n";

        printMatrix("C（计算结果）", C1);
    }

    // ---- 性能对比（中等规模矩阵 256×256） ----
    {
        std::cout << "\n--- 性能对比（256×256 矩阵） ---\n";

        size_t M = 256, K = 256, N = 256;
        Matrix A(M, K);  A.randomize();
        Matrix B(K, N);  B.randomize();

        // 朴素版本
        {
            Matrix C(M, N); C.fill(0);
            timeIt([&]() { gemmNaive(A, B, C); }, "朴素（Naive）");
        }

        // 单级分块（L1适配大小）
        {
            Matrix C(M, N); C.fill(0);
            timeIt([&]() { gemmBlocked(A, B, C, 32, 32, 32); }, "分块（32×32×32）");
        }

        // 分层分块（同时利用L1和L2缓存）
        {
            Matrix C(M, N); C.fill(0);
            timeIt([&]() { gemmHierarchical(A, B, C, 128, 128, 128, 32, 32, 32); },
                   "分层分块（L2:128, L1:32）");
        }

        // 分块 + 预转置
        {
            Matrix C(M, N); C.fill(0);
            timeIt([&]() { gemmBlockedTranspose(A, B, C, 32, 32); },
                   "分块+转置（32×32）");
        }

        // 微内核（寄存器分块）
        {
            Matrix C(M, N); C.fill(0);
            timeIt([&]() { gemmMicroKernel(A, B, C, 4, 4, 64); },
                   "微内核（4×4×64）");
        }

        // 并行版本（4线程）
        {
            Matrix C(M, N); C.fill(0);
            timeIt([&]() { gemmParallel(A, B, C, 4); }, "并行分块（4线程）");
        }
    }

    // ---- 算术强度分析 ----
    {
        std::cout << "\n--- 算术强度分析 ---\n";
        std::cout << "对于 M=N=K=256 的矩阵乘法，各版本的内存访问情况：\n";
        std::cout << "  朴素版本：A的每个元素被加载 N 次，\n";
        std::cout << "          B的每个元素被加载 M 次。\n";
        std::cout << "          算术强度 AI ≈ O(1) → 在GPU上带宽受限\n\n";
        std::cout << "  分块版本（BS=32）：A的32×32子块只加载一次，\n";
        std::cout << "                   被复用32次（对应C的32列）。\n";
        std::cout << "                   算术强度 AI ≈ BS → 可能实现计算受限\n";
    }

    // ---- 大规模矩阵测试（1024×1024） ----
    {
        std::cout << "\n--- 大规模矩阵测试（1024×1024） ---\n";

        size_t M = 1024, K = 1024, N = 1024;
        Matrix A(M, K);  A.randomize(0.001f);
        Matrix B(K, N);  B.randomize(0.001f);

        // 大矩阵只运行优化版本（朴素版本会非常慢）
        {
            Matrix C(M, N); C.fill(0);
            timeIt([&]() { gemmBlocked(A, B, C, 64, 64, 64); },
                   "分块（64×64×64）");
        }
        {
            Matrix C(M, N); C.fill(0);
            timeIt([&]() { gemmHierarchical(A, B, C, 256, 256, 256, 32, 32, 32); },
                   "分层分块（L2:256, L1:32）");
        }
        {
            Matrix C(M, N); C.fill(0);
            timeIt([&]() { gemmParallel(A, B, C, 8); },
                   "并行分块（8线程）");
        }

        // Roofline分析：1024^3 ≈ 10亿次浮点运算
        double gflops = 2.0 * M * N * K / 1e9;  // 每次乘加（mul-add）= 2次浮点运算
        std::cout << "  总浮点运算量：~" << std::fixed << std::setprecision(1)
                  << gflops << " GFLOPs\n";
    }

    std::cout << "\n==================================================\n";
    std::cout << "本示例演示的核心概念：\n";
    std::cout << "  - 朴素GEMM：算术强度 AI = O(1)，带宽受限\n";
    std::cout << "  - 分块（Blocking）：通过复用缓存中的数据提高算术强度\n";
    std::cout << "  - 分层分块（Hierarchical）：同时利用L1 + L2缓存层级结构\n";
    std::cout << "  - 预转置（Pre-transpose）：改善访存模式，有利于SIMD向量化\n";
    std::cout << "  - 微内核（Micro-kernel）：寄存器级别的分块，减少load/store\n";
    std::cout << "  - 并行化（Parallelization）：对外层j循环（C的行）进行多线程并行\n";
    std::cout << "==================================================\n";

    return 0;
}
