// lecture9_part2.cpp
// Stanford CS149, 第9讲：高效评估深度神经网络（DNN）
// 第二部分：卷积实现方法对比
//
// 本文件实现了多种卷积操作：
//   1. 直接二维卷积（Direct Convolution，7层嵌套循环）
//   2. 卷积转GEMM（im2col + 显式矩阵乘法）——将卷积转化为稠密矩阵乘法
//   3. 多通道卷积（批量处理）
//   4. ReLU激活函数 和 最大池化（Max Pooling）仿真
//   5. 每种方案的算术强度（Arithmetic Intensity）分析
//
// 核心概念说明：
//   - 卷积（Convolution）：DNN中最核心的操作之一，本质上是局部的加权求和。
//     与全连接层不同，卷积层的权重（filter/kernel）被空间复用
//     （weight sharing），大幅减少了参数量。
//   - im2col：将输入图像的每个滑动窗口展开为一行，将卷积核展开为列，
//     从而将卷积操作转化为稠密矩阵乘法（GEMM）。代价是存储膨胀 O(R*S) 倍。
//   - ReLU（Rectified Linear Unit）：逐元素的非线性激活函数 f(x)=max(0,x)。
//     引入非线性，且计算代价极低。
//   - MaxPool（最大池化）：空间下采样操作，取局部区域（如2x2）的最大值，
//     用于减小特征图尺寸、增强平移不变性。
//   - 算术强度：卷积的算术强度介于向量加法和GEMM之间，
//     具体取决于filter的尺寸和输入特征图的维度。
//
// 编译命令：g++ -std=c++17 -O2 lecture9_part2.cpp -o lecture9_part2
// 运行命令：./lecture9_part2

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <algorithm>

// ============================================================================
// 四维张量（4D Tensor）：批次 × 高度 × 宽度 × 通道数
// 数据布局采用 NHWC 格式（N=批次, H=高度, W=宽度, C=通道）
//
// 为什么用 NHWC？
//   - 在CPU上，NHWC将同一像素位置的所有通道连续存放，
//     有利于卷积操作中同时访问所有通道值
//   - GPU常用NCHW（通道在前），因为它的数据布局更适合
//     对每个通道进行独立的并行计算
//   - 不同框架偏好不同：TensorFlow默认NHWC，PyTorch默认NCHW
// ============================================================================

struct Tensor4D {
    std::vector<float> data;   // 一维扁平化存储
    size_t N, H, W, C;         // 批次、高度、宽度、通道数

    Tensor4D(size_t n, size_t h, size_t w, size_t c)
        : N(n), H(h), W(w), C(c), data(n * h * w * c, 0.0f) {}

    // NHWC 布局索引：data[((n * H + h) * W + w) * C + c]
    float& at(size_t n, size_t h, size_t w, size_t c) {
        return data[((n * H + h) * W + w) * C + c];
    }

    float at(size_t n, size_t h, size_t w, size_t c) const {
        return data[((n * H + h) * W + w) * C + c];
    }

    // 用确定性伪随机值填充张量
    void randomize(float scale = 1.0f) {
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = static_cast<float>(i % 17) * scale * 0.1f;
        }
    }

    void fill(float val) {
        std::fill(data.begin(), data.end(), val);
    }

    // 比较两个张量是否近似相等（浮点容差比较）
    bool equals(const Tensor4D& other, float tol = 0.001f) const {
        if (N != other.N || H != other.H || W != other.W || C != other.C) return false;
        for (size_t i = 0; i < data.size(); i++) {
            if (std::abs(data[i] - other.data[i]) > tol) return false;
        }
        return true;
    }
};

// ============================================================================
// 卷积权重结构体：[滤波器数量][滤波器高度][滤波器宽度][输入通道数]
// 数据布局：F × H × W × C，按 F, H, W, C 的顺序连续存储
// ============================================================================

struct ConvWeights {
    std::vector<float> data;
    size_t F, H, W, C;  // 滤波器数量、滤波器高度、滤波器宽度、输入通道数

    ConvWeights(size_t f, size_t h, size_t w, size_t c)
        : F(f), H(h), W(w), C(c), data(f * h * w * c, 0.0f) {}

    // 按 (f, h, w, c) 顺序索引
    float& at(size_t f, size_t h, size_t w, size_t c) {
        return data[((f * H + h) * W + w) * C + c];
    }

    float at(size_t f, size_t h, size_t w, size_t c) const {
        return data[((f * H + h) * W + w) * C + c];
    }

    void fill(float val) {
        std::fill(data.begin(), data.end(), val);
    }

    void randomize(float scale = 1.0f) {
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = static_cast<float>(i % 7 + 1) * scale * 0.05f;
        }
    }
};

// ============================================================================
// 打印辅助函数 —— 显示张量中某个通道的2D子区域
// ============================================================================

void printChannel(const std::string& name, const Tensor4D& t,
                  size_t n, size_t c, size_t maxSize = 8)
{
    std::cout << name << " [批次=" << n << ", 通道=" << c
              << "] " << t.H << "x" << t.W << ":\n";
    for (size_t h = 0; h < std::min(t.H, maxSize); h++) {
        std::cout << "  ";
        for (size_t w = 0; w < std::min(t.W, maxSize); w++) {
            std::cout << std::setw(6) << std::fixed << std::setprecision(2)
                      << t.at(n, h, w, c);
        }
        std::cout << "\n";
    }
}

// ============================================================================
// 1. 直接卷积（Direct Convolution）—— 7层嵌套循环
// 这是课程中讨论的"朴素"实现。
//
// 计算公式（对于每个输出位置）：
//   output[n][j][i][f] = bias[f] + Σ_kk Σ_jj Σ_ii
//       weights[f][jj][ii][kk] * input[n][j+jj][i+ii][kk]
//
// 其中：
//   - n：批次索引
//   - (j, i)：输出的空间位置（对应于二维图像坐标）
//   - f：输出通道/滤波器索引
//   - c / kk：输入通道索引
//   - (jj, ii)：滤波器内的空间索引
//
// 关键数据复用特性：
//   - 滤波器权重在不同空间位置被复用（weight sharing）：同一filter的权重
//     在输入图像的所有滑动位置上重复使用
//   - 输入的每个像素值被不同滤波器复用：同一个输入元素参与多个输出通道的计算
//   - 如果直接朴素实现，算术强度较低，因为大量访存未被有效复用
//
// 说明：此处假设步长 stride=1，无padding。
// ============================================================================

Tensor4D convDirect(const Tensor4D& input,
                    const ConvWeights& weights,
                    const std::vector<float>& biases,
                    size_t stride = 1)
{
    // 卷积输出尺寸计算：output_size = (input_size - filter_size) / stride + 1
    size_t outH = (input.H - weights.H) / stride + 1;
    size_t outW = (input.W - weights.W) / stride + 1;

    Tensor4D output(input.N, outH, outW, weights.F);

    // 7层嵌套循环：
    // N(批次) × H(输出高) × W(输出宽) × F(滤波器) × C(输入通道) × R(滤波器高) × S(滤波器宽)
    for (size_t n = 0; n < input.N; n++) {
        for (size_t oh = 0; oh < outH; oh++) {
            for (size_t ow = 0; ow < outW; ow++) {
                for (size_t f = 0; f < weights.F; f++) {
                    float sum = (f < biases.size()) ? biases[f] : 0.0f;

                    // 沿所有输入通道累加
                    for (size_t c = 0; c < input.C; c++) {
                        // 空间上的卷积（滤波器在输入上的局部区域滑动）
                        for (size_t kh = 0; kh < weights.H; kh++) {
                            for (size_t kw = 0; kw < weights.W; kw++) {
                                sum += weights.at(f, kh, kw, c)
                                     * input.at(n, oh * stride + kh, ow * stride + kw, c);
                            }
                        }
                    }

                    output.at(n, oh, ow, f) = sum;
                }
            }
        }
    }

    return output;
}

// ============================================================================
// 2. 通过 im2col + GEMM 实现卷积（显式矩阵构造法）
//
// 核心思想：
//   将输入图像reshape为一个"卷积矩阵"（X_col），其中：
//     - 每一行 = 输入在某个 (oh, ow) 位置上的 filter 大小的图像块（patch）
//       （展开为一行）
//     - 矩阵维度：(outH * outW) × (filterH * filterW * input.C)
//   然后将权重reshape为矩阵 W_mat：
//     - 维度：numFilters × (filterH * filterW * input.C)
//   最终卷积等效于：O_mat = W_mat × X_col^T（矩阵乘法）
//
// 代价分析：
//   - 存储开销：X_col 矩阵需要 O(outH * outW * R * S * C) 的额外存储，
//     相比原始输入膨胀了 O(R*S) 倍（R=filterH, S=filterW）
//   - 但好处是可以利用高度优化的GEMM库（如cuBLAS）来执行卷积
//
// 实际应用中：
//   - cuDNN（NVIDIA的DNN库）内部对于某些配置会使用此方法
//   - im2col + GEMM 在batch size较大时尤其高效
// ============================================================================

Tensor4D convIm2col(const Tensor4D& input,
                    const ConvWeights& weights,
                    const std::vector<float>& biases)
{
    size_t outH = input.H - weights.H + 1;
    size_t outW = input.W - weights.W + 1;
    size_t patchSize = weights.H * weights.W * input.C;  // R * S * C

    // 构造 im2col 矩阵 X_col：(outH * outW) 行 × patchSize 列
    // 构造权重矩阵 W_mat：numFilters 行 × patchSize 列
    size_t X_rows = outH * outW;
    size_t X_cols = patchSize;

    std::vector<float> X_col(X_rows * X_cols, 0.0f);

    // 遍历每个输出位置，提取对应的输入patch并展开为一行
    for (size_t oh = 0; oh < outH; oh++) {
        for (size_t ow = 0; ow < outW; ow++) {
            size_t row = oh * outW + ow;
            size_t col = 0;
            for (size_t c = 0; c < input.C; c++) {
                for (size_t kh = 0; kh < weights.H; kh++) {
                    for (size_t kw = 0; kw < weights.W; kw++) {
                        X_col[row * X_cols + col] = input.at(0, oh + kh, ow + kw, c);
                        col++;
                    }
                }
            }
        }
    }

    // 将权重 reshape 为矩阵：F 行 × patchSize 列
    std::vector<float> W_mat(weights.F * patchSize);
    for (size_t f = 0; f < weights.F; f++) {
        size_t col = 0;
        for (size_t c = 0; c < weights.C; c++) {
            for (size_t kh = 0; kh < weights.H; kh++) {
                for (size_t kw = 0; kw < weights.W; kw++) {
                    W_mat[f * patchSize + col] = weights.at(f, kh, kw, c);
                    col++;
                }
            }
        }
    }

    // GEMM 计算：O_mat = W_mat × X_col^T
    // O_mat 维度：F × (outH * outW)
    std::vector<float> O_mat(weights.F * X_rows, 0.0f);
    for (size_t f = 0; f < weights.F; f++) {
        for (size_t r = 0; r < X_rows; r++) {
            float sum = (f < biases.size()) ? biases[f] : 0.0f;
            for (size_t k = 0; k < patchSize; k++) {
                sum += W_mat[f * patchSize + k] * X_col[r * X_cols + k];
            }
            O_mat[f * X_rows + r] = sum;
        }
    }

    // 将 O_mat reshape 回 Tensor4D 格式
    Tensor4D output(input.N, outH, outW, weights.F);
    for (size_t f = 0; f < weights.F; f++) {
        for (size_t oh = 0; oh < outH; oh++) {
            for (size_t ow = 0; ow < outW; ow++) {
                size_t r = oh * outW + ow;
                output.at(0, oh, ow, f) = O_mat[f * X_rows + r];
            }
        }
    }

    return output;
}

// ============================================================================
// 3. ReLU 激活函数（逐元素操作）
// 定义：ReLU(x) = max(0, x)
//
// 特点：
//   - 计算极其简单：只需一次比较和一次条件赋值
//   - 引入非线性：使得DNN能够拟合任意复杂的函数
//   - 解决梯度消失问题：对于正输入，梯度恒为1
//   - 会产生"死神经元"：当输入恒为负时，该神经元永远不会被激活
// ============================================================================

void applyReLU(Tensor4D& tensor)
{
    for (float& v : tensor.data) {
        v = std::max(0.0f, v);
    }
}

// ============================================================================
// 4. 最大池化（Max Pooling 2x2）
// 功能：将空间维度缩小为原来的一半
//
// 工作原理：
//   在输入的每个2×2窗口内取最大值作为输出
//   例如：max([[a, b], [c, d]]) → max(a, b, c, d)
//
// 池化的作用：
//   - 降采样（Downsampling）：减小特征图尺寸，降低后续层的计算量
//   - 平移不变性（Translation Invariance）：使网络对输入的小幅平移更加鲁棒
//   - 防止过拟合：减少了参数数量和模型容量
//   - 在实际深度学习框架中，MaxPool和AveragePool是最常用的池化操作
// ============================================================================

Tensor4D maxPool2x2(const Tensor4D& input)
{
    size_t outH = input.H / 2;
    size_t outW = input.W / 2;
    Tensor4D output(input.N, outH, outW, input.C);

    // 对于每个批次和每个通道，分别进行2x2最大池化
    for (size_t n = 0; n < input.N; n++) {
        for (size_t c = 0; c < input.C; c++) {
            for (size_t oh = 0; oh < outH; oh++) {
                for (size_t ow = 0; ow < outW; ow++) {
                    // 取2x2窗口内的最大值
                    float maxVal = input.at(n, oh * 2, ow * 2, c);
                    maxVal = std::max(maxVal, input.at(n, oh * 2,     ow * 2 + 1, c));
                    maxVal = std::max(maxVal, input.at(n, oh * 2 + 1, ow * 2,     c));
                    maxVal = std::max(maxVal, input.at(n, oh * 2 + 1, ow * 2 + 1, c));
                    output.at(n, oh, ow, c) = maxVal;
                }
            }
        }
    }

    return output;
}

// ============================================================================
// 5. 单通道二维卷积（类似课程中的模糊滤波示例）
//
// 这是最简化的卷积形式：单通道输入 + 单通道输出，无偏置，步长=1。
// 用于演示卷积作为"模式检测器"的本质：通过滤波器与图像的滑动点积
// 来检测特定模式（边缘、纹理等）。
// ============================================================================

std::vector<float> conv2DSingleChannel(const std::vector<float>& input,
                                       size_t W, size_t H,
                                       const std::vector<float>& kernel,
                                       size_t kW, size_t kH)
{
    size_t outW = W - kW + 1;
    size_t outH = H - kH + 1;
    std::vector<float> output(outW * outH, 0.0f);

    for (size_t j = 0; j < outH; j++) {
        for (size_t i = 0; i < outW; i++) {
            float sum = 0.0f;
            for (size_t jj = 0; jj < kH; jj++) {
                for (size_t ii = 0; ii < kW; ii++) {
                    sum += input[(j + jj) * W + (i + ii)]
                         * kernel[jj * kW + ii];
                }
            }
            output[j * outW + i] = sum;
        }
    }
    return output;
}

// ============================================================================
// 计时工具 —— 测量函数执行时间
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
// 主函数 —— 演示：单通道卷积、直接卷积vs im2col、ReLU+Pooling流水线、
// 性能对比、卷积作为模式检测器
// ============================================================================

int main()
{
    std::cout << "==================================================\n";
    std::cout << "第9讲 第二部分：卷积实现方法\n";
    std::cout << "==================================================\n\n";

    // ---- 单通道 3x3 卷积（模糊滤波）示例 ----
    {
        std::cout << "--- 1. 单通道 3×3 卷积（模糊滤波） ---\n";

        size_t W = 8, H = 8;
        std::vector<float> input(W * H);
        for (size_t i = 0; i < W * H; i++) input[i] = static_cast<float>(i);

        // 3×3 模糊核（所有元素为 1/9）——均值滤波
        // 每个像素值被替换为其3×3邻域的平均值
        std::vector<float> blurKernel(9, 1.0f / 9.0f);

        auto output = conv2DSingleChannel(input, W, H, blurKernel, 3, 3);

        std::cout << "输入 8×8:\n";
        for (size_t j = 0; j < H; j++) {
            std::cout << "  ";
            for (size_t i = 0; i < W; i++)
                std::cout << std::setw(4) << static_cast<int>(input[j * W + i]);
            std::cout << "\n";
        }
        std::cout << "\n3×3 模糊后输出（6×6）:\n";
        for (size_t j = 0; j < H - 2; j++) {
            std::cout << "  ";
            for (size_t i = 0; i < W - 2; i++)
                std::cout << std::setw(7) << std::fixed << std::setprecision(2)
                         << output[j * (W - 2) + i];
            std::cout << "\n";
        }
    }

    // ---- 直接卷积 vs im2col 对比 ----
    {
        std::cout << "\n--- 2. 直接卷积 vs im2col 对比 ---\n";

        size_t N = 1, H = 8, W = 8, C = 3;       // 1个批次、8×8图像、3通道
        size_t numFilters = 4;                     // 4个滤波器
        size_t filterH = 3, filterW = 3;           // 3×3滤波器

        Tensor4D input(N, H, W, C);
        input.randomize(1.0f);

        ConvWeights weights(numFilters, filterH, filterW, C);
        weights.randomize(1.0f);

        std::vector<float> biases(numFilters, 0.1f);

        // 执行直接卷积
        auto outDirect = convDirect(input, weights, biases);

        // 执行 im2col + GEMM 卷积
        auto outIm2col = convIm2col(input, weights, biases);

        std::cout << "输入尺寸：" << N << "×" << H << "×" << W << "×" << C << "\n";
        std::cout << "滤波器：" << numFilters << "×" << filterH << "×"
                  << filterW << "×" << C << "\n";
        std::cout << "输出尺寸：" << outDirect.N << "×" << outDirect.H
                  << "×" << outDirect.W << "×" << outDirect.C << "\n";

        // 打印一个通道用于对比
        printChannel("输入（通道0）", input, 0, 0);
        printChannel("直接卷积输出（滤波器0）", outDirect, 0, 0);
        printChannel("im2col 输出（滤波器0）", outIm2col, 0, 0);

        bool match = outDirect.equals(outIm2col, 0.01f);
        std::cout << "\n直接卷积 == im2col：" << (match ? "通过" : "未通过") << "\n";
    }

    // ---- Conv → ReLU → MaxPool 完整流水线演示 ----
    {
        std::cout << "\n--- 3. 卷积 → ReLU → 最大池化 流水线 ---\n";

        size_t N = 1, H = 16, W = 16, C = 1;
        size_t numFilters = 2;
        size_t filterH = 3, filterW = 3;

        Tensor4D input(N, H, W, C);
        input.randomize();

        ConvWeights weights(numFilters, filterH, filterW, C);
        weights.randomize();

        std::vector<float> biases(numFilters, 0.0f);

        // 第一步：卷积
        auto convOut = convDirect(input, weights, biases);
        std::cout << "卷积后尺寸：" << convOut.H << "×" << convOut.W << "×" << convOut.C << "\n";
        printChannel("  滤波器0（ReLU前）", convOut, 0, 0, 8);

        // 第二步：ReLU激活
        applyReLU(convOut);
        std::cout << "ReLU激活后：\n";
        printChannel("  滤波器0", convOut, 0, 0, 8);

        // 第三步：2×2最大池化
        auto pooled = maxPool2x2(convOut);
        std::cout << "2×2最大池化后尺寸：" << pooled.H << "×" << pooled.W << "\n";
        printChannel("  滤波器0", pooled, 0, 0, 8);
    }

    // ---- 性能对比：直接卷积 vs im2col ----
    {
        std::cout << "\n--- 4. 性能对比：直接卷积 vs im2col（较大输入） ---\n";

        size_t N = 1, H = 64, W = 64, C = 16;
        size_t numFilters = 32;
        size_t filterH = 3, filterW = 3;

        Tensor4D input(N, H, W, C);
        input.randomize();

        ConvWeights weights(numFilters, filterH, filterW, C);
        weights.randomize();

        std::vector<float> biases(numFilters, 0.1f);

        timeIt([&]() { convDirect(input, weights, biases); }, "直接卷积");
        timeIt([&]() { convIm2col(input, weights, biases); }, "im2col + GEMM");

        // 分析 im2col 的存储开销
        size_t outH = H - filterH + 1;
        size_t outW = W - filterW + 1;
        size_t patchSize = filterH * filterW * C;
        size_t im2colElements = outH * outW * patchSize;
        std::cout << "\n  im2col 矩阵大小：" << im2colElements
                  << " 个元素 (" << im2colElements * 4 / 1024 << " KB)\n";
        std::cout << "  原始输入大小：" << N * H * W * C * 4 / 1024 << " KB\n";
        std::cout << "  存储膨胀率：" << std::fixed << std::setprecision(1)
                  << (static_cast<float>(im2colElements) / (N * H * W * C) - 1.0f) * 100.0f
                  << "%\n";
    }

    // ---- 卷积作为"模式检测器"的应用示例 ----
    {
        std::cout << "\n--- 5. 卷积作为模式检测器：边缘检测 ---\n";

        // 使用 Sobel 算子进行边缘检测
        // 水平 Sobel 核（检测垂直边缘）：
        // [[1,  0, -1],
        //  [2,  0, -2],
        //  [1,  0, -1]]
        // Sobel算子计算水平方向上的梯度，对垂直方向的边缘高度敏感
        size_t W = 10, H = 10;

        // 构造一个包含垂直边缘的简单测试图像
        std::vector<float> image(W * H, 0.0f);
        // 左半部分 = 0，右半部分 = 10（图像中心处的跳变形成垂直边缘）
        for (size_t j = 0; j < H; j++) {
            for (size_t i = 0; i < W; i++) {
                image[j * W + i] = (i >= W / 2) ? 10.0f : 0.0f;
            }
        }

        // 水平方向 Sobel 核（检测垂直边缘）
        std::vector<float> sobelX = {1, 0, -1,
                                     2, 0, -2,
                                     1, 0, -1};

        auto edges = conv2DSingleChannel(image, W, H, sobelX, 3, 3);

        std::cout << "输入图像（中心处存在垂直边缘）:\n";
        for (size_t j = 0; j < H; j++) {
            std::cout << "  ";
            for (size_t i = 0; i < W; i++)
                std::cout << std::setw(3) << static_cast<int>(image[j * W + i]);
            std::cout << "\n";
        }

        std::cout << "\n经 Sobel-X 滤波器检测后（垂直方向边缘）:\n";
        for (size_t j = 0; j < H - 2; j++) {
            std::cout << "  ";
            for (size_t i = 0; i < W - 2; i++)
                std::cout << std::setw(6) << std::fixed << std::setprecision(0)
                         << edges[j * (W - 2) + i];
            std::cout << "\n";
        }
        std::cout << "  注：边缘检测结果在垂直边缘处（图像中心）出现了较大的响应值\n";
    }

    std::cout << "\n==================================================\n";
    std::cout << "本示例演示的核心概念：\n";
    std::cout << "  - 直接卷积（Direct Convolution）：7层循环，讨论数据复用模式\n";
    std::cout << "  - im2col：将输入patch展开为矩阵，转化为矩阵乘法（GEMM）\n";
    std::cout << "  - im2col存储开销：存储膨胀 O(R*S) 倍（R、S为滤波器尺寸）\n";
    std::cout << "  - ReLU：逐元素 max(0, x) 非线性激活，计算极快\n";
    std::cout << "  - MaxPool：空间下采样（2×2），增强平移不变性\n";
    std::cout << "  - 卷积作为模式检测器：通过特定核（如Sobel）检测边缘/纹理\n";
    std::cout << "==================================================\n";

    return 0;
}
