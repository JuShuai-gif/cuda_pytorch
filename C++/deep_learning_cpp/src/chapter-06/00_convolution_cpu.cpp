/*
 * 00_convolution_cpu.cpp - 第 6 章：卷积神经网络
 * 基于嵌套循环的 CPU 基本卷积实现（对应原书第 173-176 页）
 *
 * 演示内容：
 *   1. ConvolutionalLayer 类：手动 4 层嵌套循环实现卷积
 *   2. 高斯随机初始化卷积核（mt19937, seed=42）
 *   3. 逐元素乘积累加 + ReLU 激活
 *   4. 时间复杂度分析：O(F × H_out × W_out × K²)
 */

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cmath>

/* ===================== ConvolutionalLayer 卷积层 ======================== */
class ConvolutionalLayer {
private:
    std::vector<std::vector<std::vector<float>>> filters; // 多个 2D 卷积核
    int stride;                                           // 步长
    int padding;                                          // 填充（当前实现仅支持 padding=0）

public:
    /* ------- 构造函数：高斯随机初始化卷积核 ------- */
    ConvolutionalLayer(int numFilters, int filterSize,
                       int strideSize, int paddingSize) : stride(strideSize), padding(paddingSize) {
        // 使用固定种子的 Mersenne Twister 生成可复现的随机数
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.1f);

        filters.resize(numFilters);
        for (int f = 0; f < numFilters; ++f) {
            filters[f].resize(filterSize);
            for (int i = 0; i < filterSize; ++i) {
                filters[f][i].resize(filterSize);
                for (int j = 0; j < filterSize; ++j) {
                    filters[f][i][j] = dist(rng); // 从 N(0, 0.1²) 采样
                }
            }
        }
    }

    /* ------- 前向传播：逐滤波器滑动窗口计算 ----------- */
    std::vector<std::vector<std::vector<float>>> forward(
        const std::vector<std::vector<float>> &input) {
        int inputRows = static_cast<int>(input.size());
        int inputCols = static_cast<int>(input[0].size());
        int numFilters = static_cast<int>(filters.size());
        int filterSize = static_cast<int>(filters[0].size());

        // 输出特征图尺寸：H_out = (H_in - K) / stride + 1, 同理 W_out
        int outputRows = (inputRows - filterSize) / stride + 1;
        int outputCols = (inputCols - filterSize) / stride + 1;

        // 第 1 层循环：遍历每个卷积核（f = 0, 1, ..., F-1）
        std::vector<std::vector<std::vector<float>>> featureMaps(numFilters);
        for (int f = 0; f < numFilters; ++f) {
            featureMaps[f].resize(outputRows,
                                  std::vector<float>(outputCols, 0.0f));

            // 第 2 层循环：输出特征图的行索引
            for (int i = 0; i < outputRows; ++i) {
                // 第 3 层循环：输出特征图的列索引
                for (int j = 0; j < outputCols; ++j) {
                    float sum = 0.0f;
                    // 第 4-a 层循环：卷积核内的行偏移 k
                    for (int k = 0; k < filterSize; ++k) {
                        // 第 4-b 层循环：卷积核内的列偏移 l
                        for (int l = 0; l < filterSize; ++l) {
                            sum += input[i + k][j + l] // 输入元素
                                   * filters[f][k][l]; // 卷积核权重
                        }
                    }
                    // ReLU 激活：max(0, sum)，引入非线性
                    featureMaps[f][i][j] = std::max(0.0f, sum);
                }
            }
        }
        return featureMaps;
    }
};

/* ================================ main ================================= */
int main() {
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║     第 6 章：卷积神经网络  -  CPU 基本卷积实现         ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";

    // 构造 5×5 测试输入矩阵（值 1-25）
    std::vector<std::vector<float>> input(5, std::vector<float>(5));
    float val = 1.0f;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            input[i][j] = val++;
        }
    }

    std::cout << "【输入矩阵】5×5（值 1~25）:\n";
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            std::cout << std::setw(5) << std::fixed
                      << std::setprecision(0) << input[i][j];
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // 创建卷积层：2 个卷积核，3×3 大小，stride=1，padding=0
    ConvolutionalLayer conv(/*numFilters=*/2, /*filterSize=*/3,
                            /*stride=*/1, /*padding=*/0);

    // 运行前向传播
    auto featureMaps = conv.forward(input);

    // 打印每个卷积核产生的特征图
    std::cout << "【卷积核参数】\n";
    // 注：通过友元或 getter 访问——此处简化处理，实际使用中可扩展接口
    std::cout << "  (见代码中高斯随机初始化，均值=0.0，标准差=0.1)\n\n";

    for (size_t f = 0; f < featureMaps.size(); ++f) {
        std::cout << "【特征图 #" << (f + 1) << "】"
                  << " (ReLU 激活后):\n";
        for (size_t i = 0; i < featureMaps[f].size(); ++i) {
            for (size_t j = 0; j < featureMaps[f][i].size(); ++j) {
                std::cout << std::setw(10) << std::fixed
                          << std::setprecision(4) << featureMaps[f][i][j];
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    // 时间复杂度分析
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║               时间复杂度分析                            ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
    std::cout << "║  O(F × H_out × W_out × K²)                             ║\n";
    std::cout << "║                                                        ║\n";
    std::cout << "║  F     = 卷积核数量（此处 = 2）                         ║\n";
    std::cout << "║  H_out = 输出特征图高度 = (H_in - K)/S + 1             ║\n";
    std::cout << "║  W_out = 输出特征图宽度  = (W_in - K)/S + 1             ║\n";
    std::cout << "║  K²    = 卷积核面积（此处 = 3×3 = 9）                  ║\n";
    std::cout << "║                                                        ║\n";
    std::cout << "║  本例：2 × 3 × 3 × 9 = 162 次乘加运算                  ║\n";
    std::cout << "║  注：嵌套循环方式在大尺寸输入时效率低下，               ║\n";
    std::cout << "║       后续将介绍 im2col + 矩阵乘法优化                  ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";

    return 0;
}
