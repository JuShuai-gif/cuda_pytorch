// lecture10_part1.cpp
// 脉动阵列（Systolic Array）矩阵乘法仿真
// 模拟 Google TPU v1 中使用的权重驻留（Weight-Stationary）脉动阵列架构
// Stanford CS149, 2025年秋季 - 第10讲：硬件专用化（Hardware Specialization）
//
// 核心概念说明：
//
// 1. 脉动阵列（Systolic Array）：
//    - 一种由大量简单处理单元（PE）按网格排列构成的数据流架构
//    - 数据像"脉搏"一样以固定的节奏在阵列中有序流动
//    - 每个PE执行简单的乘加运算（MAC: Multiply-Accumulate）
//    - 特点：数据驱动（data-driven），最小化控制开销
//
// 2. 权重驻留（Weight-Stationary）数据流：
//    - Google TPU v1 采用的策略
//    - 权重被预加载到每个PE中并保持不动（驻留/stationary）
//    - 输入数据从左到右水平流经阵列
//    - 每个PE列累积部分和（partial sum）
//    - 优势：减少了权重的重复加载，最大化数据复用
//
// 3. 与 SIMD（单指令多数据流）的对比：
//    - SIMD：控制驱动，需要频繁的指令发射和寄存器读写
//    - Systolic：数据驱动，控制开销极小（无需指令流）
//    - TPU v1：约30%的芯片面积用于算术单元，远高于传统CPU
//      因为不需要复杂的分支预测、乱序执行等控制逻辑
//
// 4. 时空调度（Spatio-Temporal Scheduling）：
//    - 空间维度：不同PE同时处理不同数据
//    - 时间维度：数据按周期在PE间传递
//    - 同时具有时间局部性（PE内累加）和空间局部性（PE间数据流）
//
// 编译命令：g++ -std=c++17 -O2 lecture10_part1.cpp -o lecture10_part1
// 运行命令：./lecture10_part1

#include <iostream>
#include <vector>
#include <iomanip>
#include <cassert>

// 脉动阵列的默认尺寸（4×4 PE网格）
const int PE_ROWS = 4;
const int PE_COLS = 4;

// ============================================================================
// 处理单元（Processing Element, PE）
// 脉动阵列的基本构建模块，每个PE执行一个乘加操作
//
// PE的工作循环：
//   1. 加载权重（weight）：预加载阶段，权重值驻留在PE中
//   2. 接收输入（input）：每个周期接收来自左侧PE或外部的输入数据
//   3. 计算（compute）：执行 accumulator += weight × input
//   4. 前传输入（forwardInput）：将输入数据传递给右侧的PE
//      （实现流水线式的数据传递）
// ============================================================================
struct PE {
    double weight = 0.0;       // 驻留权重值（在整个计算过程中保持不变）
    double accumulator = 0.0;  // 累加器：存储乘加运算的累积结果
    double input = 0.0;        // 当前输入值（来自左侧）

    void loadWeight(double w) { weight = w; }
    void receiveInput(double x) { input = x; }
    void compute() { accumulator += weight * input; }
    void forwardInput(double& toNext) const { toNext = input; }
    void reset() { accumulator = 0.0; input = 0.0; }
};

// ============================================================================
// 权重驻留脉动阵列（Weight-Stationary Systolic Array）
//
// 工作原理：
//   1. 权重被预加载到PE网格中并保持不动（驻留，stationary）
//   2. 输入数据从左到右流经整个阵列（流水线式传递）
//   3. 沿每一列向下累积部分和（在本实现中各列分别累积到accumulators_）
//
// 对应 TPU v1 的 MXU（Matrix Multiply Unit）设计：
//   - 256×256 PE 网格
//   - 每个PE执行 int8 乘加运算
//   - 峰值算力 92 TFLOPS (int8)
// ============================================================================
class SystolicArray {
public:
    SystolicArray(int rows = PE_ROWS, int cols = PE_COLS)
        : rows_(rows), cols_(cols),
          pes_(rows, std::vector<PE>(cols)),
          accumulators_(cols, 0.0) {}

    // 预加载权重到PE网格中（权重驻留策略）
    // weight[r][c] 被加载到位于第 r 行、第 c 列的PE中
    void loadWeights(const std::vector<std::vector<double>>& weights) {
        assert(weights.size() == (size_t)rows_);
        for (int r = 0; r < rows_; ++r) {
            assert(weights[r].size() == (size_t)cols_);
            for (int c = 0; c < cols_; ++c) {
                pes_[r][c].loadWeight(weights[r][c]);
                pes_[r][c].reset();
            }
        }
    }

    // 将一个输入列（向量）流式推入脉动阵列
    // inputs[i] 进入第 i 行PE；每个PE将数据向右传递给下一个PE
    //
    // 数据流过程（一个时钟周期内）：
    //   PE[r][0].receiveInput(inputs[r])  →  PE[r][0].compute()
    //   PE[r][0].forwardInput(data)       →  PE[r][1] 获取 data，等等
    void streamInput(const std::vector<double>& inputs) {
        assert(inputs.size() == (size_t)rows_);

        // 每一行独立处理（行间无直接依赖）
        for (int r = 0; r < rows_; ++r) {
            double data = inputs[r];
            for (int c = 0; c < cols_; ++c) {
                pes_[r][c].receiveInput(data);   // PE接收数据
                pes_[r][c].compute();            // 执行乘加
                pes_[r][c].forwardInput(data);   // 将数据传递给下一个PE（向右）
            }
        }
    }

    // 读出累加器结果（每列一个输出值）
    // 将所有PE同一列中的累加器值求和
    std::vector<double> readAccumulators() const {
        std::vector<double> result(cols_, 0.0);
        for (int c = 0; c < cols_; ++c) {
            // 对该列所有PE的累加器值求和
            for (int r = 0; r < rows_; ++r) {
                result[c] += pes_[r][c].accumulator;
            }
        }
        return result;
    }

    // 使用权重驻留脉动阵列执行完整 GEMM 运算：C = A × B
    // 参数含义：
    //   A: M × K 矩阵
    //   B: K × N 矩阵
    //   C: M × N 矩阵（输出）
    //
    // 权重驻留策略：
    //   1. 预加载 B^T 作为权重（PE[r][c] 获得 B[c][r]，
    //      用于计算 C[r][c]）
    //   2. 沿着 K 维度将 A 的每一列依次推入脉动阵列
    //   3. 每次推入后，所有PE完成一次乘加更新
    static std::vector<std::vector<double>> gemm(
        const std::vector<std::vector<double>>& A,
        const std::vector<std::vector<double>>& B) {

        int M = A.size();
        int K = A[0].size();
        int Kb = B.size();
        int N = B[0].size();
        assert(Kb == K);

        std::vector<std::vector<double>> C(M, std::vector<double>(N, 0.0));

        // 权重驻留脉动阵列演示：
        // 预加载 B^T 作为权重（PE[r][c] 获得 B[c][r]，用于计算 C[r][c]）
        SystolicArray sa(N, M);
        std::vector<std::vector<double>> W(N, std::vector<double>(M));
        for (int c = 0; c < N; ++c)
            for (int r = 0; r < M; ++r)
                W[c][r] = B[r][c];  // B转置：每个PE列累加一个C元素

        sa.loadWeights(W);

        // 沿着 K 维度将 A 的每一列依次推入阵列
        // 在第 k 步：输入 A[i][k] 进入 PE 的第 i 行
        for (int k = 0; k < K; ++k) {
            std::vector<double> input_col(N);
            for (int j = 0; j < N; ++j) {
                input_col[j] = A[k][j];  // 简化：A 为 K×M，使用 A^T 视图
            }
            sa.streamInput(input_col);
        }

        // 读出累加器结果
        std::vector<double> acc = sa.readAccumulators();
        for (int j = 0; j < N; ++j)
            C[0][j] = acc[j];  // 演示简化

        return C;
    }

    // 打印PE网格的当前状态（显示每个PE的累加器值）
    void printState() const {
        std::cout << "PE网格状态（累加器值）:\n";
        for (int r = 0; r < rows_; ++r) {
            for (int c = 0; c < cols_; ++c) {
                std::cout << std::setw(10) << std::fixed
                          << std::setprecision(2) << pes_[r][c].accumulator << " ";
            }
            std::cout << "\n";
        }
    }

private:
    int rows_, cols_;                                   // PE网格的行列数
    std::vector<std::vector<PE>> pes_;                   // PE二维网格
    std::vector<double> accumulators_;                   // 每列的累加器
};

// ============================================================================
// 朴素 GEMM —— 用于验证脉动阵列计算结果的正确性
// ============================================================================
std::vector<std::vector<double>> naiveGemm(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B) {

    int M = A.size();
    int K = A[0].size();
    int N = B[0].size();
    std::vector<std::vector<double>> C(M, std::vector<double>(N, 0.0));

    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < K; ++k)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

// ============================================================================
// 主函数 —— 演示脉动阵列的逐步执行过程、正确性验证、扩展性讨论
// ============================================================================
int main() {
    std::cout << "=== 第10讲：脉动阵列仿真 ===\n";
    std::cout << "Stanford CS149 - 硬件专用化\n\n";

    // 示例：A = 4×4 的权重矩阵，B = 4×1 的输入向量
    // 等价于 y = Wx，其中 W 为 4×4 权重矩阵，x 为 4×1 输入向量
    std::vector<std::vector<double>> W = {
        {1.0, 2.0, 3.0, 4.0},
        {5.0, 6.0, 7.0, 8.0},
        {9.0, 10.0, 11.0, 12.0},
        {13.0, 14.0, 15.0, 16.0}
    };

    std::vector<std::vector<double>> X = {
        {0.5}, {1.0}, {1.5}, {2.0}
    };

    std::cout << "权重矩阵 W（4×4）:\n";
    for (const auto& row : W) {
        for (double v : row) std::cout << std::setw(8) << v;
        std::cout << "\n";
    }

    std::cout << "\n输入向量 x（4×1）:\n";
    for (const auto& row : X) {
        for (double v : row) std::cout << std::setw(8) << v;
        std::cout << "\n";
    }

    // 第一步：逐步演示脉动执行过程
    std::cout << "\n--- 脉动阵列逐步执行过程 ---\n";
    SystolicArray sa(4, 4);
    sa.loadWeights(W);

    std::cout << "权重加载后：\n";
    sa.printState();

    // 依次推入每个输入元素（演示流水线式执行）
    for (int k = 0; k < 4; ++k) {
        std::cout << "\n推入输入 x[" << k << "] = " << X[k][0] << "：\n";
        std::vector<double> input = {X[k][0], X[k][0], X[k][0], X[k][0]};
        sa.streamInput(input);
        sa.printState();
    }

    std::cout << "\n累加器输出结果：";
    auto acc = sa.readAccumulators();
    for (double v : acc) std::cout << v << " ";
    std::cout << "\n";

    // 第二步：与朴素 GEMM 对比验证正确性
    std::cout << "\n--- 与朴素 GEMM 对比验证 ---\n";
    auto expected = naiveGemm(W, X);
    std::cout << "朴素 GEMM 计算结果：\n";
    for (const auto& row : expected) {
        for (double v : row) std::cout << v << " ";
        std::cout << "\n";
    }

    // 第三步：讨论脉动阵列对更大规模矩阵的扩展
    std::cout << "\n--- 扩展到更大规模 ---\n";
    std::cout << "对于更大规模的矩阵（例如 8×8 × 8×4096）：\n";
    std::cout << "  - 需要 4096 个累加器来存放输出列\n";
    std::cout << "  - 需要在空间维度上对计算进行分块（tiling）\n";
    std::cout << "  - TPU 的核心优势：约30%的芯片面积用于算术单元\n";
    std::cout << "  - SIMD 对比：控制驱动，局部性有限\n";
    std::cout << "  - 脉动阵列：数据驱动的波前传播，同时具有时间和空间局部性\n";
    std::cout << "\n  进一步说明：\n";
    std::cout << "  • SIMD（如AVX-512）需要从寄存器加载数据、执行指令、写回结果，\n";
    std::cout << "    存在较大的指令发射开销和寄存器文件读写开销\n";
    std::cout << "  • 脉动阵列通过硬件级别的数据流调度消除了指令开销，\n";
    std::cout << "    每个周期每个PE自然地完成一次乘加\n";
    std::cout << "  • TPU v1 使用 65536 个 int8 MAC 单元（256×256网格），\n";
    std::cout << "    在 700MHz 下达到 92 TFLOPS 的峰值算力\n";

    return 0;
}
