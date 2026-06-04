/*
 * preprocessing_with_libraries.cpp
 * 第2章：C++中的数据准备与预处理
 *
 * PDF 参考：第51-56页，"基于库的预处理"
 *
 * PDF 强调，相同的预处理技术（缺失值填充、编码、缩放、降维、时间序列操作）
 * 可以使用经过实战检验的 C++ 库更简洁高效地实现。
 * 本文件使用 Armadillo 演示基于库的方法。
 *
 * PDF 中提到的库（第51-52页）：
 *   - dlib：机器学习和数据处理工具
 *   - Eigen：用于变换和嵌入的线性代数
 *   - mlpack：可扩展的机器学习和预处理
 *   - Armadillo：用于数据处理的线性代数
 *   - GPU 后端：cuBLAS/cuFFT/cuML（可选的硬件加速）
 *
 * 本演示使用 Armadillo 的原因：
 *   1. 系统中已安装（libarmadillo-dev）
 *   2. 提供类似 MATLAB 的语法：简洁、可读的矩阵操作
 *   3. 包装了 LAPACK/BLAS 以提供高性能线性代数
 *   4. PDF 第55页展示了 Armadillo 的具体示例（arma::conv 等）
 *
 * 依赖：sudo apt install libarmadillo-dev
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include <armadillo>

// ----------------------------------------------------------------
// Armadillo 为预处理任务提供了简洁的语法，
// 这些任务如果用原始 C++ 循环则需要很多行代码。
// ----------------------------------------------------------------

// --- 滚动窗口均值（PDF 第55页） ---
// arma::conv() 执行一维卷积——远比手动循环简单。
// "same" 模式：输出与输入大小相同，边缘进行填充。
arma::vec rollingMeanArma(const arma::vec &data, int window) {
    arma::vec kernel(window);
    kernel.fill(1.0 / window); // 均匀权重 = 移动平均
    return arma::conv(data, kernel, "same");
}

// --- 指数平滑（PDF 第55页） ---
// Smoothed[t] = alpha * data[t] + (1-alpha) * smoothed[t-1]
// Armadillo 通过逐元素操作使每一步成为一行代码。
arma::vec expSmoothArma(const arma::vec &data, double alpha) {
    arma::vec smoothed(data.n_elem);
    smoothed(0) = data(0);
    for (size_t i = 1; i < data.n_elem; ++i) {
        smoothed(i) = alpha * data(i) + (1.0 - alpha) * smoothed(i - 1);
    }
    return smoothed;
}

// --- Z-Score 标准化（PDF 第42页，库版本） ---
// Armadillo 向量化：内置 mean()、stddev()。
// 无需手动循环——整个操作只需2行。
arma::vec zScoreArma(const arma::vec &data) {
    double mu = arma::mean(data);
    double sigma = arma::stddev(data);
    return (data - mu) / sigma;
}

// --- 基于 SVD 的 PCA（PDF 第52-54页，库版本） ---
// PDF 中提到，对于 PCA，mlpack/Eigen 的 SVD 方法比协方差 EVD
// 在数值上更稳定。Armadillo 提供了 princomp()。
// 返回：{变换后的数据、特征值、特征向量}
struct PCAResult {
    arma::mat scores;      // 变换后的数据 (n x k)
    arma::vec eigenvalues; // 每个主成分解释的方差
    arma::mat loadings;    // 特征向量 (d x k)
};

PCAResult pcaArma(const arma::mat &data, int numComponents) {
    PCAResult result;
    arma::princomp(result.scores, result.loadings, result.eigenvalues, data);
    // 只保留前 numComponents 个
    result.scores = result.scores.cols(0, numComponents - 1);
    result.eigenvalues = result.eigenvalues.rows(0, numComponents - 1);
    result.loadings = result.loadings.cols(0, numComponents - 1);
    return result;
}

// --- Min-Max 缩放（库版本） ---
// Armadillo 的 min()/max() 是向量化的。
arma::vec minMaxArma(const arma::vec &data, double a, double b) {
    double minVal = arma::min(data);
    double maxVal = arma::max(data);
    return a + (data - minVal) / (maxVal - minVal) * (b - a);
}

int main() {
    std::cout << "=== Library-Powered Preprocessing (Armadillo) ===\n";
    std::cout << "PDF pages 51-56: Same techniques, concise code, better numerics\n\n";

    arma::arma_rng::set_seed(42);

    // ===========================================
    // 1. Rolling Window Mean
    // ===========================================
    std::cout << "[Rolling Window] arma::conv() replaces manual loops\n";
    arma::vec ts = {10, 12, 11.5, 14, 13, 16, 15.5, 18, 17, 20, 19.5, 22};
    std::cout << "  Original: " << ts.t();
    auto rolled = rollingMeanArma(ts, 3);
    std::cout << "  w=3 mean: " << rolled.t() << "\n";
    std::cout << "  Compare manual: ~15 lines of loops vs 3 lines Armadillo\n\n";

    // ===========================================
    // 2. Exponential Smoothing
    // ===========================================
    std::cout << "[Exponential Smoothing] α control over smoothing level\n";
    auto es03 = expSmoothArma(ts, 0.3);
    auto es07 = expSmoothArma(ts, 0.7);
    std::cout << "  α=0.3: " << es03.t();
    std::cout << "  α=0.7: " << es07.t();
    std::cout << "  α=0.3: heavy smoothing; α=0.7: follows data closely\n\n";

    // ===========================================
    // 3. Z-Score Normalization
    // ===========================================
    std::cout << "[Z-Score] arma::mean() + arma::stddev() = 2 lines\n";
    arma::vec raw = {1, 5, 10, 50, 100, 500};
    auto z = zScoreArma(raw);
    std::cout << "  Raw:  " << raw.t();
    std::cout << "  Norm: " << z.t() << "\n\n";

    // ===========================================
    // 4. Min-Max Scaling
    // ===========================================
    std::cout << "[Min-Max] Scale to [0,1] with Armadillo\n";
    auto mm = minMaxArma(raw, 0.0, 1.0);
    std::cout << "  Scaled: " << mm.t() << "\n\n";

    // ===========================================
    // 5. PCA via Armadillo (princomp)
    // ===========================================
    std::cout << "[PCA] arma::princomp() uses SVD for better numerics\n";
    std::cout << "  PDF p52-54: SVD-based PCA is more stable than EVD\n";

    // 10 samples x 5 features
    arma::mat data = {
        {2.5, 2.4, 0.5, 0.7, 1.0},
        {0.5, 0.7, 1.2, 0.3, 0.8},
        {2.2, 2.9, 0.3, 0.9, 1.1},
        {1.9, 2.2, 0.8, 0.6, 0.9},
        {3.1, 3.0, 0.2, 0.8, 1.2},
        {2.3, 2.7, 0.4, 0.5, 1.0},
        {2.0, 1.6, 1.0, 0.4, 0.7},
        {1.0, 1.1, 1.5, 0.2, 0.5},
        {1.5, 1.6, 1.1, 0.5, 0.8},
        {1.1, 0.9, 1.4, 0.3, 0.6}};

    auto pca = pcaArma(data, 2);
    std::cout << "  Input:        10x5\n";
    std::cout << "  Output:       10x2 (top 2 PCs)\n";
    std::cout << "  Eigenvalues:  " << pca.eigenvalues.t();
    std::cout << "  Variance retained: "
              << arma::sum(pca.eigenvalues) / arma::sum(arma::eig_sym(data.t() * data)) * 100
              << "%\n";
    std::cout << "  Scores (first 5):\n"
              << pca.scores.rows(0, 4) << "\n";

    std::cout << "---\n";
    std::cout << "Key library advantages (PDF p51):\n";
    std::cout << "  1. Concise syntax — fewer bugs, faster prototyping\n";
    std::cout << "  2. SIMD/GPU acceleration — LAPACK/BLAS under the hood\n";
    std::cout << "  3. Battle-tested numerics — SVD vs EVD for PCA stability\n";
    std::cout << "  4. Easy to switch backends — Armadillo->Eigen->mlpack\n";

    return 0;
}
