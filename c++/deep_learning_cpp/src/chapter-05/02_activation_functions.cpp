/*
 * 02_activation_functions.cpp - 第 5 章：激活函数一览
 * 纯 C++ STL 实现书中 144-152 页所有激活函数及导数
 *
 * 章节组织：
 *   1. Sigmoid 家族（Sigmoid、Tanh、Softplus、Softsign）
 *   2. ReLU  家族（ReLU、Leaky ReLU、PReLU、ELU、SELU、GELU）
 *   3. 高级激活（Swish、Mish、Hard Swish、Hard Tanh）
 */

#include <iostream>
#include <cmath>
#include <iomanip>
#include <algorithm>

constexpr double kPi = 3.14159265358979323846;

/* ================================================================
 * 第一部分：Sigmoid 家族（第 144-147 页）
 * ================================================================ */

/* Sigmoid: 将任意实值映射到 (0, 1) */
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

/* Sigmoid 导数：s'(x) = s(x) * (1 - s(x)) */
double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

/* Tanh: 将任意实值映射到 (-1, 1)，零中心 */
double tanh_custom(double x) {
    double ep = std::exp(x);
    double en = std::exp(-x);
    return (ep - en) / (ep + en);
}

/* Tanh 导数：tanh'(x) = 1 - tanh²(x) */
double tanh_derivative(double x) {
    double t = tanh_custom(x);
    return 1.0 - t * t;
}

/* Softplus: 光滑近似于 ReLU，输出 (0, +∞) */
double softplus(double x) {
    // 对于较大的 x，exp(x) 可能溢出，使用分段策略
    if (x > 20.0) return x;
    return std::log(1.0 + std::exp(x));
}

/* Softsign: 除以 L1 范数归一化，输出 (-1, 1) */
double softsign(double x) {
    return x / (1.0 + std::abs(x));
}

/* ================================================================
 * 第二部分：ReLU 家族（第 147-150 页）
 * ================================================================ */

/* ReLU: max(0, x)，最广泛使用的激活函数 */
double relu(double x) {
    return std::max(0.0, x);
}

/* ReLU 导数：x>0 时为 1，否则为 0 */
double relu_derivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

/* Leaky ReLU: 负半轴保留微小斜率 alpha，缓解神经元"死亡" */
double leaky_relu(double x, double alpha = 0.01) {
    return x > 0.0 ? x : alpha * x;
}

/* PReLU: 参数化 ReLU，alpha 作为可学习参数 */
double prelu(double x, double alpha) {
    return x > 0.0 ? x : alpha * x;
}

/* ELU: 指数线性单元，负半轴以 exp(x)-1 平滑过渡到 -alpha */
double elu(double x, double alpha = 1.0) {
    return x > 0.0 ? x : alpha * (std::exp(x) - 1.0);
}

/* SELU: 自归一化神经网络专用，自动将激活值拉向零均值和单位方差 */
double selu(double x) {
    constexpr double kLambda = 1.05070098735548;
    constexpr double kAlpha = 1.67326324235438;
    return x > 0.0 ? kLambda * x : kLambda * kAlpha * (std::exp(x) - 1.0);
}

/* GELU: 高斯误差线性单元，BERT / GPT 等 Transformer 常用 */
double gelu(double x) {
    constexpr double kCoeff = 0.044715;
    double inner = std::sqrt(2.0 / kPi) * (x + kCoeff * x * x * x);
    return 0.5 * x * (1.0 + tanh_custom(inner));
}

/* ================================================================
 * 第三部分：高级激活函数（第 150-152 页）
 * ================================================================ */

/* Swish: x·σ(βx)，Google 提出的自门控激活 */
double swish(double x, double beta = 1.0) {
    return x * sigmoid(beta * x);
}

/* Mish: x·tanh(softplus(x))，连续光滑的非单调激活函数 */
double mish(double x) {
    return x * tanh_custom(softplus(x));
}

/* Hard Swish: Swish 的硬件友好分段线性近似 */
double hard_swish(double x) {
    double clamped = std::clamp((x + 3.0) / 6.0, 0.0, 1.0);
    return x * clamped;
}

/* Hard Tanh: 将输出硬裁剪到 [-1, 1] */
double hard_tanh(double x) {
    return std::clamp(x, -1.0, 1.0);
}

/* ================================================================
 * 演示程序入口
 * ================================================================ */
int main() {
    std::cout << std::fixed << std::setprecision(4);

    const int kSteps = 13; // -3.0 至 3.0 步长 0.5
    double x_vals[kSteps];
    for (int i = 0; i < kSteps; ++i)
        x_vals[i] = -3.0 + i * 0.5;

    const int kColW = 10; // 列宽

    /* ===================== 表头 ===================== */
    std::cout << "/* ======================================================== */\n";
    std::cout << "/*       第 5 章 激活函数一览 — 纯 C++ STL 实现               */\n";
    std::cout << "/* ======================================================== */\n\n";

    std::cout << std::setw(kColW) << "x";
    for (int i = 0; i < kSteps; ++i)
        std::cout << std::setw(kColW) << x_vals[i];
    std::cout << "\n";
    std::cout << std::string(kColW, '-');
    for (int i = 0; i < kSteps; ++i)
        std::cout << std::string(kColW, '-');
    std::cout << "\n";

    /* ===================== 辅助打印宏 ===================== */
    auto print_row = [&](const char *name, double (*fn)(double)) {
        std::cout << std::setw(kColW) << name;
        for (int i = 0; i < kSteps; ++i)
            std::cout << std::setw(kColW) << fn(x_vals[i]);
        std::cout << "\n";
    };

    /* ================================================================
     * 第一部分：Sigmoid 家族
     * ================================================================ */
    std::cout << "\n/* ---------- 1. Sigmoid 家族（第 144-147 页）---------- */\n\n";

    print_row("Sigmoid", sigmoid);
    print_row("σ'(x)", sigmoid_derivative);
    print_row("Tanh", tanh_custom);
    print_row("tanh'(x)", tanh_derivative);
    print_row("Softplus", softplus);
    print_row("Softsign", softsign);

    std::cout << "\n// Sigmoid:   输出范围 (0, 1)，概率输出层常用，易发生梯度饱和\n";
    std::cout << "// Tanh:      输出范围 (-1, 1)，零中心化，比 Sigmoid 收敛更快\n";
    std::cout << "// Softplus:  输出范围 (0, +∞)，ReLU 的光滑近似\n";
    std::cout << "// Softsign:  输出范围 (-1, 1)，L1 归一化的多项式替代\n";

    /* ================================================================
     * 第二部分：ReLU 家族
     * ================================================================ */
    std::cout << "\n/* ---------- 2. ReLU 家族（第 147-150 页）---------- */\n\n";

    std::cout << std::setw(kColW) << "x";
    for (int i = 0; i < kSteps; ++i)
        std::cout << std::setw(kColW) << x_vals[i];
    std::cout << "\n";
    std::cout << std::string(kColW, '-');
    for (int i = 0; i < kSteps; ++i)
        std::cout << std::string(kColW, '-');
    std::cout << "\n";

    print_row("ReLU", relu);
    print_row("ReLU'(x)", relu_derivative);

    // Leaky ReLU 需要 lambda 包装默认参数
    auto leaky_relu_01 = [](double x) { return leaky_relu(x, 0.01); };
    print_row("LReLU(0.01)", leaky_relu_01);

    auto prelu_25 = [](double x) { return prelu(x, 0.25); };
    print_row("PReLU(0.25)", prelu_25);

    auto elu_1 = [](double x) { return elu(x, 1.0); };
    print_row("ELU(1.0)", elu_1);

    print_row("SELU", selu);
    print_row("GELU", gelu);

    std::cout << "\n// ReLU:   输出范围 [0, +∞)，计算简单，可能产生「死神经元」\n";
    std::cout << "// LReLU:  输出范围 (-∞, +∞)，负半轴微小平滑防止神经元死亡\n";
    std::cout << "// PReLU:  输出范围 (-∞, +∞)，alpha 通过反向传播学习\n";
    std::cout << "// ELU:    输出范围 (-alpha, +∞)，负值允许均值靠近零\n";
    std::cout << "// SELU:   自归一化，配合特定权重初始化可实现固定点收敛\n";
    std::cout << "// GELU:   高斯误差线性单元，Transformer 模型事实标准\n";

    /* ================================================================
     * 第三部分：高级激活函数
     * ================================================================ */
    std::cout << "\n/* ---------- 3. 高级激活函数（第 150-152 页）---------- */\n\n";

    std::cout << std::setw(kColW) << "x";
    for (int i = 0; i < kSteps; ++i)
        std::cout << std::setw(kColW) << x_vals[i];
    std::cout << "\n";
    std::cout << std::string(kColW, '-');
    for (int i = 0; i < kSteps; ++i)
        std::cout << std::string(kColW, '-');
    std::cout << "\n";

    auto swish_1 = [](double x) { return swish(x, 1.0); };
    print_row("Swish(1.0)", swish_1);

    print_row("Mish", mish);
    print_row("HardSwish", hard_swish);
    print_row("HardTanh", hard_tanh);

    std::cout << "\n// Swish:     x·σ(x)，自门控机制，比 ReLU 更平滑，性能常优于 ReLU\n";
    std::cout << "// Mish:      x·tanh(softplus(x))，无上界有下界，非单调光滑\n";
    std::cout << "// HardSwish: Swish 的分段线性近似，移动端部署友好\n";
    std::cout << "// HardTanh:  硬裁剪至 [-1, 1]，用于循环网络梯度裁剪\n";

    std::cout << "\n/* ======================== 完成 ======================== */\n";

    return 0;
}
