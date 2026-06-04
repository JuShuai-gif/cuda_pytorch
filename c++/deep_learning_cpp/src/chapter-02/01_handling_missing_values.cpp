/*
 * handling_missing_values.cpp
 * 第 2 章：C++ 中的数据准备与预处理
 *
 * 缺失值在真实数据集中十分常见，必须在训练前处理。
 * 方法的选择取决于缺失机制（MCAR、MAR、MNAR）和下游模型的需求。
 *
 * 涵盖的技术（来自 PDF“处理缺失值”一节）：
 *   - 均值填充：用列均值替换 NaN。
 *     快速、单遍扫描的基线方案；但会扭曲方差，减弱相关性。
 *     最适合缺失稀疏且为 MCAR 的情况。
 *   - 前向填充：将最近一次观测值向前传递。
 *     快速，对有状态信号（如设备模式）保留短期水平。
 *     可能将过时值传播到较长间隙。
 *   - 后向填充：用下一个观测值填充。
 *     简单，但在训练中可能泄露未来信息；对长间隙过度平滑。
 *   - k-NN 插补：在已观测特征上按距离找到 k 个最近行，
 *     聚合邻居（均值/中位数）。需要缩放，谨慎选择 k。
 *     在高维中退化。
 *   - 回归插补：使用训练好的模型从其他已观测特征预测缺失值。
 *     使用交叉拟合避免泄露。
 */

#include <vector>
#include <numeric>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <limits>
#include <iomanip>

// ----------------------------------------------------------------
// 均值填充：用该特征的列均值替换 NaN。
// 快速基线方案；缩小方差，如果数据不是完全随机缺失，
// 可能会使模型产生偏差。
// ----------------------------------------------------------------
std::vector<double> meanSubstitution(
    const std::vector<double> &data) {
    double sum = 0.0;
    int count = 0;
    for (const auto &val : data) {
        if (!std::isnan(val)) {
            sum += val;
            count++;
        }
    }
    double mean = (count > 0) ? sum / count : 0.0;
    std::vector<double> filledData = data;
    for (auto &val : filledData) {
        if (std::isnan(val)) {
            val = mean;
        }
    }
    return filledData;
}

// ----------------------------------------------------------------
// 前向填充：将最近一次观测值向前传递。
// 快速，保留短期水平连续性，但可能将过时/异常值传播到
// 较长间隙。前导 NaN 保持为 NaN。
// ----------------------------------------------------------------
std::vector<double> forwardFill(
    const std::vector<double> &data) {
    std::vector<double> filledData = data;
    for (size_t i = 1; i < filledData.size(); ++i) {
        if (std::isnan(filledData[i])) {
            filledData[i] = filledData[i - 1];
        }
    }
    return filledData;
}

// ----------------------------------------------------------------
// 后向填充：将每个 NaN 用下一个观测值填充。
// 简单，但在训练流程中可能泄露未来信息。
// ----------------------------------------------------------------
std::vector<double> backwardFill(
    const std::vector<double> &data) {
    std::vector<double> filledData = data;
    for (size_t i = filledData.size() - 1; i > 0; --i) {
        if (std::isnan(filledData[i - 1])) {
            filledData[i - 1] = filledData[i];
        }
    }
    return filledData;
}

// ----------------------------------------------------------------
// 回归插补：使用在完整样本上拟合的线性模型预测缺失值：
// target_i = slope * predictor_i + intercept。
// 在生产中，使用交叉拟合以避免数据泄露。
// ----------------------------------------------------------------
std::vector<double> regressionImputation(
    const std::vector<double> &target,
    const std::vector<double> &predictor,
    double slope, double intercept) {
    std::vector<double> filledData = target;
    for (size_t i = 0; i < filledData.size(); ++i) {
        if (std::isnan(filledData[i])) {
            filledData[i] = slope * predictor[i] + intercept;
        }
    }
    return filledData;
}

// ----------------------------------------------------------------
// k-NN 插补：在两者均已观测的特征子集上使用欧氏距离
// 找到 k 个最相似的行，然后将缺失值插补为邻居的
// 距离加权均值。需要特征缩放和谨慎选择 k。
// ----------------------------------------------------------------
std::vector<double> knnImputation(
    const std::vector<std::vector<double>> &data,
    int k, int targetRow) {
    size_t nCols = data[0].size();
    // 查找目标行中哪一列有 NaN
    int nanCol = -1;
    for (size_t j = 0; j < nCols; ++j) {
        if (std::isnan(data[targetRow][j])) {
            nanCol = static_cast<int>(j);
            break;
        }
    }
    if (nanCol < 0) return data[targetRow]; // 没有 NaN 需要修复

    // 计算到所有其他行的距离
    std::vector<std::pair<double, size_t>> distances;
    for (size_t r = 0; r < data.size(); ++r) {
        if ((int)r == targetRow) continue;
        // 跳过在同一列或其他位置也有 NaN 的行
        bool skip = false;
        for (size_t j = 0; j < nCols; ++j) {
            if (std::isnan(data[r][j])) {
                skip = true;
                break;
            }
        }
        if (skip) continue;

        // 对除含 NaN 列以外的所有列计算欧氏距离
        double dist = 0.0;
        for (size_t j = 0; j < nCols; ++j) {
            if ((int)j == nanCol) continue;
            double diff = data[targetRow][j] - data[r][j];
            dist += diff * diff;
        }
        distances.push_back({std::sqrt(dist), r});
    }

    // 选择 k 个最近的邻居
    std::sort(distances.begin(), distances.end());
    int actualK = std::min(k, (int)distances.size());
    if (actualK == 0) return data[targetRow];

    // 邻居的 NaN 列值的距离加权均值
    double weightedSum = 0.0, weightTotal = 1e-12;
    for (int i = 0; i < actualK; ++i) {
        size_t neighborRow = distances[i].second;
        double d = distances[i].first + 1e-12;
        double w = 1.0 / d;
        weightedSum += w * data[neighborRow][nanCol];
        weightTotal += w;
    }

    auto result = data[targetRow];
    result[nanCol] = weightedSum / weightTotal;
    return result;
}

// ----------------------------------------------------------------
// 辅助函数：打印向量，对 NaN 值显示 "NaN"
// ----------------------------------------------------------------
void printRow(const std::string &label, const std::vector<double> &r) {
    std::cout << label << " [";
    for (size_t j = 0; j < r.size(); ++j) {
        if (std::isnan(r[j]))
            std::cout << "NaN";
        else
            std::cout << std::fixed << std::setprecision(2) << r[j];
        if (j + 1 < r.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

int main() {
    const double NaN = std::numeric_limits<double>::quiet_NaN();

    std::cout << "=== Missing Value Handling Demos ===\n\n";

    // --- 均值填充 ---
    {
        std::vector<double> data = {1.0, 2.0, NaN, 4.0, 5.0, NaN, 7.0};
        std::cout << "[均值填充] 用列均值替换 NaN。\n";
        std::cout << "  当缺失稀疏时，作为快速基线方案效果良好。\n";
        printRow("  Before:", data);
        printRow("  After: ", meanSubstitution(data));
        std::cout << std::endl;
    }

    // --- 前向填充 ---
    {
        std::vector<double> data = {1.0, NaN, NaN, 4.0, NaN, 6.0};
        std::cout << "[前向填充] 将最近一次观测值向前传递。\n";
        std::cout << "  快速；适用于有状态信号（如设备模式）。\n";
        printRow("  Before:", data);
        printRow("  After: ", forwardFill(data));
        std::cout << std::endl;
    }

    // --- 后向填充 ---
    {
        std::vector<double> data = {1.0, NaN, NaN, 4.0, NaN, 6.0};
        std::cout << "[后向填充] 用下一个观测值填充。\n";
        std::cout << "  简单，但在训练中可能泄露未来信息。\n";
        printRow("  Before:", data);
        printRow("  After: ", backwardFill(data));
        std::cout << std::endl;
    }

    // --- 回归插补 ---
    {
        std::vector<double> target = {10.0, 20.0, NaN, 40.0, 50.0};
        std::vector<double> predictor = {1.0, 2.0, 3.0, 4.0, 5.0};
        double slope = 10.0, intercept = 0.0; // 在完整样本上拟合得到
        std::cout << "[回归插补] 从其他特征预测缺失值。\n";
        std::cout << "  模型: target_i = " << slope << " * predictor_i + " << intercept << "\n";
        printRow("  Before:", target);
        printRow("  After: ", regressionImputation(target, predictor, slope, intercept));
        std::cout << std::endl;
    }

    // --- k-NN 插补 ---
    {
        std::cout << "[k-NN 插补] 使用 k=2 个最近邻居。\n";
        std::cout << "  异构数据需要特征缩放。\n";
        std::vector<std::vector<double>> data = {
            {1.00, 2.00, 3.00},
            {1.10, NaN, 3.20}, // <- 目标行，需要插补
            {0.90, 2.10, 2.90},
            {1.20, 1.90, 3.10}};
        int targetIndex = 1;
        int k = 2;

        printRow("  Before:", data[targetIndex]);
        auto imputed = knnImputation(data, k, targetIndex);
        printRow("  After: ", imputed);
    }

    return 0;
}
