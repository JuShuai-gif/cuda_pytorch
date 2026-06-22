/*
 * time_series_engineering.cpp
 * 第2章：C++中的数据准备与预处理
 *
 * 时间序列数据需要针对特定领域的预处理，以便在输入深度学习模型之前
 * 提取有意义的时间模式。其目标是揭示原始时间戳本身无法捕捉的
 * 趋势、季节性和滞后依赖关系。
 *
 * 涵盖的技术（来自 PDF "时间序列工程" 章节）：
 *   - 滚动窗口聚合：在固定窗口上计算移动平均值，
 *     以平滑噪声并揭示潜在趋势。
 *   - 指数平滑：加权平均，其中最近的观测值权重更大；
 *     由平滑因子 alpha (0,1) 控制。
 *   - 差分：减去连续值以消除趋势并
 *     使序列平稳（许多模型需要这样做）。
 *   - 傅里叶变换（FFT）：将信号分解为频率分量。
 *     用于检测周期性模式/季节性。
 *   - 基于时间的特征提取：从时间戳中导出周期性特征（sin/cos）
 *     以编码周期信息（一天中的小时、一周中的天）。
 */

#include <vector>
#include <complex>
#include <cmath>
#include <iostream>
#include <iomanip>

// ----------------------------------------------------------------
// 滚动窗口均值：将每个位置替换为前 windowSize 个值的平均值。
// 平滑噪声，揭示趋势。
// 前 (windowSize-1) 个位置保持为 0（尚未有完整窗口）。
// ----------------------------------------------------------------
std::vector<double> rollingMean(
    const std::vector<double> &data, int windowSize) {
    std::vector<double> result(data.size(), 0.0);
    for (size_t i = 0; i + windowSize <= data.size(); ++i) {
        double sum = 0.0;
        for (int j = 0; j < windowSize; ++j) {
            sum += data[i + j];
        }
        result[i + windowSize - 1] = sum / windowSize;
    }
    return result;
}

// ----------------------------------------------------------------
// 指数平滑：smoothed_t = alpha * x_t + (1-alpha) * s_{t-1}。
// alpha 小 -> 平滑力度大（反应慢）。
// alpha 大 -> 平滑力度小（紧跟数据）。
// 常用于预测基线和降噪。
// ----------------------------------------------------------------
std::vector<double> exponentialSmoothing(
    const std::vector<double> &data, double alpha) {
    if (data.empty()) return {};
    std::vector<double> smoothed(data.size(), 0.0);
    smoothed[0] = data[0];
    for (size_t i = 1; i < data.size(); ++i) {
        smoothed[i] = alpha * data[i] + (1.0 - alpha) * smoothed[i - 1];
    }
    return smoothed;
}

// ----------------------------------------------------------------
// 一阶差分：d_t = x_{t+1} - x_t。
// 消除线性趋势，使序列平稳。
// 结果有 (n-1) 个元素。
// ----------------------------------------------------------------
std::vector<double> differencing(
    const std::vector<double> &data) {
    std::vector<double> differencedData;
    for (size_t i = 1; i < data.size(); ++i) {
        differencedData.push_back(data[i] - data[i - 1]);
    }
    return differencedData;
}

// ----------------------------------------------------------------
// 离散傅里叶变换（DFT）：将信号分解为频率分量。
// O(n^2) 参考实现；生产环境中请使用 FFTW。
// 用于检测周期性模式和季节性。
// ----------------------------------------------------------------
std::vector<std::complex<double>> performDFT(
    const std::vector<double> &data) {
    size_t n = data.size();
    std::vector<std::complex<double>> transformed(n);
    for (size_t k = 0; k < n; ++k) {
        std::complex<double> sum(0.0, 0.0);
        for (size_t t_val = 0; t_val < n; ++t_val) {
            double angle = 2.0 * M_PI * t_val * k / n;
            sum += std::polar(data[t_val], -angle);
        }
        transformed[k] = sum;
    }
    return transformed;
}

// ----------------------------------------------------------------
// 基于时间的特征提取：使用 sin/cos 变换编码周期性时间分量。
// 这保留了时间的循环特性（例如，23点与0点很接近）。
//
// 用例：数据集中的每个时间戳都可以用这些特征来丰富，
// 帮助模型学习每日/每周/每月的模式。
// ----------------------------------------------------------------
struct TimeFeatures {
    int hour;
    int dayOfWeek;
    int month;
    double hourSin;
    double hourCos;
    double dayOfWeekSin;
    double dayOfWeekCos;
};

TimeFeatures extractTimeFeatures(int hour, int dayOfWeek, int month) {
    TimeFeatures f;
    f.hour = hour;
    f.dayOfWeek = dayOfWeek;
    f.month = month;

    // 周期性小时编码：sin/cos(2*pi*hour/24)
    f.hourSin = std::sin(2.0 * M_PI * hour / 24.0);
    f.hourCos = std::cos(2.0 * M_PI * hour / 24.0);

    // 周期性星期编码：sin/cos(2*pi*dow/7)
    f.dayOfWeekSin = std::sin(2.0 * M_PI * dayOfWeek / 7.0);
    f.dayOfWeekCos = std::cos(2.0 * M_PI * dayOfWeek / 7.0);

    return f;
}

// 辅助函数：打印向量
void printVec(const std::string &label, const std::vector<double> &v,
              int maxN = 10) {
    std::cout << label << " [";
    size_t n = std::min(v.size(), (size_t)maxN);
    for (size_t i = 0; i < n; ++i) {
        std::cout << std::fixed << std::setprecision(2) << v[i];
        if (i + 1 < v.size()) std::cout << ", ";
    }
    if (v.size() > (size_t)maxN) std::cout << "...";
    std::cout << "]\n";
}

int main() {
    // 样本时间序列：12 个带趋势和噪声的值
    std::vector<double> ts = {
        10.0, 12.0, 11.5, 14.0, 13.0, 16.0,
        15.5, 18.0, 17.0, 20.0, 19.5, 22.0};

    std::cout << "=== Time-Series Engineering Demos ===\n\n";
    printVec("Original series:    ", ts);

    // 1. Rolling mean (window=3)
    auto rm = rollingMean(ts, 3);
    printVec("[Rolling mean w=3]  ", rm);
    std::cout << "  Smooths noise, reveals trend (first 2 values are 0).\n\n";

    // 2. Exponential smoothing (alpha=0.3 and alpha=0.7)
    auto esLow = exponentialSmoothing(ts, 0.3);
    auto esHigh = exponentialSmoothing(ts, 0.7);
    printVec("[Exp smooth α=0.3] ", esLow);
    std::cout << "  Heavy smoothing; slow to react to changes.\n";
    printVec("[Exp smooth α=0.7] ", esHigh);
    std::cout << "  Light smoothing; follows data more closely.\n\n";

    // 3. Differencing
    auto diff = differencing(ts);
    printVec("[Differencing]      ", diff);
    std::cout << "  Removes linear trend; result has (n-1) elements.\n\n";

    // 4. DFT (magnitudes)
    auto dft = performDFT(ts);
    std::cout << "[DFT Magnitudes]    [";
    for (size_t i = 0; i < std::min(dft.size(), (size_t)6); ++i) {
        std::cout << std::fixed << std::setprecision(1) << std::abs(dft[i]);
        if (i + 1 < std::min(dft.size(), (size_t)6)) std::cout << ", ";
    }
    std::cout << "...]\n";
    std::cout << "  Decomposes signal into frequency components.\n";
    std::cout << "  For production, use FFTW (O(n log n) vs O(n^2)).\n\n";

    // 5. Time-based feature extraction
    std::cout << "[Time Features]\n";
    std::cout << "  Encode cyclical time components with sin/cos to\n";
    std::cout << "  preserve circular structure (e.g., 23:00 ~ 01:00).\n";
    auto tf = extractTimeFeatures(14, 3, 6); // 2pm, Wednesday, June
    std::cout << "  hour=" << tf.hour << " -> sin="
              << std::fixed << std::setprecision(3) << tf.hourSin
              << " cos=" << tf.hourCos << "\n";
    std::cout << "  dayOfWeek=" << tf.dayOfWeek << " -> sin="
              << std::setprecision(3) << tf.dayOfWeekSin
              << " cos=" << tf.dayOfWeekCos << "\n";

    return 0;
}
