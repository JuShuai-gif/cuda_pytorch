/*
 * feature_interaction_engineering.cpp
 * 第2章：C++中的数据准备与预处理
 *
 * 线性模型只能捕捉特征之间的加法关系。
 * 特征交互工程创建表示乘法（非加法）关系的新特征，
 * 在不增加更深架构的情况下增强模型的表达能力。
 *
 * 涵盖的技术（来自 PDF "特征交互工程" 章节）：
 *   - 多项式特征：从基础特征生成高阶项（x^2, x^3, ...）
 *     和交叉项（x_i * x_j）。使线性模型能够拟合非线性决策边界。
 *     警告：维度会组合爆炸式增长——请使用正则化。
 *   - 交互项：显式创建表示两个或多个基础特征
 *     乘积的特征。有助于捕捉协同效应（例如，用 "price * quantity" 预测收入）。
 *
 * 适用场景：
 *   - 在非线性问题上使用线性/逻辑回归
 *   - 深度网络成本过高时的浅层模型
 *   - 需要可解释性时（可以检查哪些交互起作用）
 *
 * 避免场景：
 *   - 已使用具有非线性激活的深度网络
 *   - 高维数据（爆炸：d 个特征 -> ~d^2 个交互）
 */

#include <vector>
#include <iostream>
#include <iomanip>
#include <string>

// ----------------------------------------------------------------
// 多项式特征：从 [x1, x2] 生成 [1, x1, x2, x1^2, x1*x2, x2^2]
// 设置 include_bias=true 添加常数 1.0 项。
// degree=2 创建二次特征；degree=3 添加三次项。
// 当 d 很大时，请使用正则化以避免过拟合。
// ----------------------------------------------------------------
std::vector<std::vector<double>> polynomialFeatures(
    const std::vector<std::vector<double>> &X,
    int degree, bool includeBias = true) {
    size_t nSamples = X.size();
    size_t nFeatures = X[0].size();

    // 收集单项式索引：对于 degree=2，我们想要所有满足 i<=j 的对 (i,j)
    std::vector<std::pair<int, int>> monomials;
    if (includeBias)
        monomials.push_back({-1, -1}); // 偏置项
    for (int d = 1; d <= degree; ++d) {
        // 为简单起见，这里只完整实现了二次的情况
        if (d == 1) {
            for (size_t i = 0; i < nFeatures; ++i)
                monomials.push_back({(int)i, -1}); // 单一特征
        } else if (d == 2) {
            for (size_t i = 0; i < nFeatures; ++i)
                for (size_t j = i; j < nFeatures; ++j)
                    monomials.push_back({(int)i, (int)j}); // 对
        }
    }

    size_t nOutFeatures = monomials.size();
    std::vector<std::vector<double>> result(
        nSamples, std::vector<double>(nOutFeatures));

    for (size_t s = 0; s < nSamples; ++s) {
        for (size_t f = 0; f < nOutFeatures; ++f) {
            auto [i, j] = monomials[f];
            if (i < 0) {
                result[s][f] = 1.0; // 偏置
            } else if (j < 0) {
                result[s][f] = X[s][i]; // 线性
            } else {
                result[s][f] = X[s][i] * X[s][j]; // 交互 / 平方
            }
        }
    }
    return result;
}

// ----------------------------------------------------------------
// 成对交互项：从 [x1, x2, x3] 生成所有满足 i < j 的
// 乘积 x_i * x_j（不包含自身乘积）。
// 比完整的多项式展开更有针对性；专注于
// 跨特征的协同效应。
// ----------------------------------------------------------------
std::vector<std::vector<double>> interactionTerms(
    const std::vector<std::vector<double>> &X) {
    size_t nSamples = X.size();
    size_t nFeatures = X[0].size();

    // 计算对数
    size_t nPairs = nFeatures * (nFeatures - 1) / 2;

    std::vector<std::vector<double>> result(
        nSamples, std::vector<double>(nPairs));

    for (size_t s = 0; s < nSamples; ++s) {
        size_t idx = 0;
        for (size_t i = 0; i < nFeatures; ++i) {
            for (size_t j = i + 1; j < nFeatures; ++j) {
                result[s][idx++] = X[s][i] * X[s][j];
            }
        }
    }
    return result;
}

// 辅助函数：打印二维数据
void print2D(const std::string &label,
             const std::vector<std::vector<double>> &mat) {
    std::cout << label << ":\n";
    for (const auto &row : mat) {
        std::cout << "  [";
        for (size_t j = 0; j < row.size(); ++j) {
            std::cout << std::fixed << std::setprecision(1) << row[j];
            if (j + 1 < row.size()) std::cout << ", ";
        }
        std::cout << "]\n";
    }
}

int main() {
    // 3 samples x 2 features
    std::vector<std::vector<double>> X = {
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0}};

    std::cout << "=== Feature Interaction Engineering Demos ===\n\n";
    print2D("Original data (3x2)", X);

    // 1. Polynomial features (degree=2, with bias)
    // [x1, x2] -> [1, x1, x2, x1^2, x1*x2, x2^2]
    auto poly = polynomialFeatures(X, 2, true);
    print2D("\n[Polynomial deg=2] [1, x1, x2, x1^2, x1*x2, x2^2]", poly);
    std::cout << "  Enables linear models to fit quadratic relationships.\n"
                 "  Warning: d features -> O(d^2) output dims. Use regularization.\n";

    // 2. Interaction terms only
    // [x1, x2] -> [x1*x2] (only cross-terms, no self-products)
    auto inter = interactionTerms(X);
    print2D("\n[Interaction Terms] [x1*x2]", inter);
    std::cout << "  Targeted cross-feature products only (no self-products).\n"
                 "  Captures synergies between feature pairs.\n";

    return 0;
}
