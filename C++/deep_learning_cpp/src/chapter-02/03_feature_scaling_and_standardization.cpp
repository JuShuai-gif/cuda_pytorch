/*
 * feature_scaling_and_standardization.cpp
 * 第 2 章：C++ 中的数据准备与预处理
 *
 * 缩放可以防止大范围特征主导损失景观，导致梯度下降
 * 之字形移动。归一化后的特征将等高线重塑为近似圆形，
 * 从而支持更大、更稳定的步长。
 *
 * 涵盖的技术（来自 PDF“特征缩放与标准化”一节）：
 *   - Min-max 缩放：(x-min)/(max-min) -> [0,1]（或自定义范围）。
 *     简单，保留相对距离。对离群值敏感。
 *     在训练集上估计边界；使用截断或鲁棒变体。
 *   - Z-score 归一化：(x-mean)/stddev -> N(0,1)。
 *     适用于近似高斯分布的数据。对离群值敏感。
 *   - 鲁棒缩放：(x-median)/IQR。对离群值/重尾具有抵抗力。
 *     对接近零的 IQR 使用截断或回退方案。
 *   - 对数变换：log(1+x) 压缩大值，减少右偏。
 *     对零值使用 log1p。对负值使用 Yeo-Johnson。
 *   - 幂变换 (Box-Cox)：x^power 稳定方差。
 *     需要正值；Box-Cox 通过 lambda 参数进行泛化。
 */

#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>
#include <iomanip>

// ----------------------------------------------------------------
// Min-max 缩放：线性映射到 [minRange, maxRange]。
// 保留相对距离。对离群值敏感。
// 仅在训练集上拟合边界以避免数据泄露。
// ----------------------------------------------------------------
std::vector<double> minMaxScale(
    const std::vector<double> &data,
    double minRange, double maxRange) {
    double minVal = *std::min_element(data.begin(), data.end());
    double maxVal = *std::max_element(data.begin(), data.end());
    double range = maxVal - minVal;
    if (range < 1e-12) range = 1.0; // 防止常量特征

    std::vector<double> scaledData;
    for (const auto &val : data) {
        double scaled = minRange + ((val - minVal) / range) * (maxRange - minRange);
        scaledData.push_back(scaled);
    }
    return scaledData;
}

// ----------------------------------------------------------------
// Z-score 归一化：中心化为零均值，缩放为单位方差。
// 对近似高斯分布的数据有效。对离群值敏感。
// 仅在训练集上计算均值/标准差。
// ----------------------------------------------------------------
std::vector<double> zScoreNormalize(
    const std::vector<double> &data) {
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    double variance = 0.0;
    for (const auto &val : data) {
        variance += std::pow(val - mean, 2);
    }
    variance /= data.size();
    double stddev = std::sqrt(variance);
    if (stddev < 1e-12) stddev = 1.0;

    std::vector<double> normalizedData;
    for (const auto &val : data) {
        normalizedData.push_back((val - mean) / stddev);
    }
    return normalizedData;
}

// ----------------------------------------------------------------
// 鲁棒缩放：以中位数为中心，以 IQR（Q3-Q1）为尺度。
// 对离群值和重尾具有抵抗力。在训练集上拟合统计量。
// 防止接近零的 IQR。
// ----------------------------------------------------------------
std::vector<double> robustScale(
    const std::vector<double> &data) {
    std::vector<double> sortedData = data;
    std::sort(sortedData.begin(), sortedData.end());
    size_t n = sortedData.size();

    double median = sortedData[n / 2];
    double q1 = sortedData[n / 4];
    double q3 = sortedData[3 * n / 4];
    double iqr = q3 - q1;
    if (iqr < 1e-12) iqr = 1.0;

    std::vector<double> scaledData;
    for (const auto &val : data) {
        scaledData.push_back((val - median) / iqr);
    }
    return scaledData;
}

// ----------------------------------------------------------------
// 对数变换：log(1+x) 压缩大值，减少右偏，
// 通常稳定方差。使用 log1p 处理零值。
// 对负值或混合符号数据，考虑使用 Yeo-Johnson。
// ----------------------------------------------------------------
std::vector<double> logTransform(
    const std::vector<double> &data) {
    std::vector<double> transformedData;
    for (const auto &val : data) {
        // log1p(x) = log(1+x)，安全处理零值
        transformedData.push_back(std::log1p(std::max(0.0, val)));
    }
    return transformedData;
}

// ----------------------------------------------------------------
// 幂变换（泛化版）：对每个值应用 x^power。
// 为异方差数据稳定方差。Box-Cox 要求正值；
// 这是简化版本。
// ----------------------------------------------------------------
std::vector<double> powerTransform(
    const std::vector<double> &data, double power) {
    std::vector<double> transformedData;
    for (const auto &val : data) {
        if (std::abs(power) < 1e-12) {
            // power=0 等价于取对数
            transformedData.push_back(std::log(std::max(1e-12, val)));
        } else {
            transformedData.push_back(std::pow(val, power));
        }
    }
    return transformedData;
}

// 辅助函数：打印向量
void printVec(const std::string &label, const std::vector<double> &v) {
    std::cout << label << " [";
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << std::fixed << std::setprecision(3) << v[i];
        if (i + 1 < v.size()) std::cout << ", ";
    }
    std::cout << "]\n";
}

int main() {
    // 带有偏态分布和一个离群值的样本数据
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 100.0};

    std::cout << "=== Feature Scaling and Standardization Demos ===\n\n";
    printVec("Original data:   ", data);
    std::cout << "  （注意：值 100 是一个会影响 min-max 和 z-score 的离群值）\n\n";

    // 1. Min-max 缩放到 [0, 1]
    {
        auto scaled = minMaxScale(data, 0.0, 1.0);
        printVec("[Min-Max 0-1]  ", scaled);
        std::cout << "  简单且快速。离群值 (100) 将所有其他值压缩到 0 附近。\n\n";
    }

    // 2. Z-score 归一化
    {
        auto scaled = zScoreNormalize(data);
        printVec("[Z-Score]       ", scaled);
        std::cout << "  均值≈0，标准差≈1。对高斯分布数据有效。离群值会膨胀标准差。\n\n";
    }

    // 3. 鲁棒缩放
    {
        auto scaled = robustScale(data);
        printVec("[Robust Scale]  ", scaled);
        std::cout << "  使用中位数和 IQR。对离群值具有抵抗力 - 100 几乎不会扭曲其他值。\n\n";
    }

    // 4. 对数变换
    {
        auto scaled = logTransform(data);
        printVec("[Log Transform] ", scaled);
        std::cout << "  log(1+x) 压缩大值。非常适合带零值的右偏数据。\n\n";
    }

    // 5. 幂变换 (sqrt = 0.5, square = 2.0)
    {
        auto sqrtTrans = powerTransform(data, 0.5);
        auto sqTrans = powerTransform(data, 2.0);
        printVec("[Power sqrt]    ", sqrtTrans);
        std::cout << "  x^0.5：压缩范围，减少偏度。\n";
        printVec("[Power square]  ", sqTrans);
        std::cout << "  x^2.0：放大差异，强调大值。\n";
    }

    return 0;
}
