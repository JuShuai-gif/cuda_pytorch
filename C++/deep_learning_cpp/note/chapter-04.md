# 第 4 章：构建基础神经网络

基于 *Deep Learning with C++*（Packt，ISBN 9781835880036）第 4 章，第 110–126 页。本章位于 Part 2（"Building and Training Neural Networks in C++"）的开篇章节，从零构建——从线性回归到 MLP 反向传播——奠定全书深度网络实现的理论和代码基础。

---

## 目录

1. [章节概述](#章节概述)
2. [核心概念：从感知机到 MLP](#核心概念从感知机到-mlp)
3. [文件索引](#文件索引)
4. [代码演进：前向传播与反向传播](#代码演进前向传播与反向传播)
5. [编译与运行](#编译与运行)
6. [技术速查](#技术速查)
7. [PDF 完整内容对照](#pdf-完整内容对照)
8. [注意事项](#注意事项)

---

## 章节概述

机器学习让算法从数据中发现模式并用其预测未见过的样本——无需为每种情况硬编码规则。神经网络是受大脑启发的一类 ML 模型：感知机（输入 × 权重 + 偏置 → 激活函数 → 输出）是基本构建块，堆叠多层就构成了多层感知机（MLP），能近似从像素到标签、从特征到概率的复杂非线性映射。

本章通过一条清晰的代码递进路径：**线性回归 → 逻辑回归 → LibTorch 神经元 → Eigen MLP → 手写反向传播 + SGD 变体**，让你透彻理解前向传播、损失函数、梯度计算、参数更新的完整流程。

### 核心学习目标

| 目标                 | 说明                                                                                     |
| -------------------- | ---------------------------------------------------------------------------------------- |
| 理解前向传播         | 神经元 = 仿射变换 (`wx+b`) + 非线性激活 (`σ`/`ReLU`)，堆叠层级实现层次化特征学习         |
| 损失函数与优化       | MSE（回归）、BCE（分类），梯度下降变体（Batch/SGD/Mini-batch）                            |
| LibTorch 神经元      | `torch::nn::Linear` + 激活函数，Kaiming 初始化，设备感知（CPU/CUDA）                      |
| 从零实现 MLP         | 用 Eigen 写 `NeuralNetwork` 类：构造→`feedforward`→`train`，随机初始化打破对称性          |
| 反向传播数学         | 链式法则 + delta 传递：`output_delta → hidden_error → hidden_delta`，外积计算梯度         |
| SGD 变体比较         | Batch GD（全量）、SGD（单样本）、Mini-batch SGD（小批量）——速度 vs 稳定性 vs 硬件效率     |

### 知识脉络（代码递进 5 步走）

```
线性回归（1 个神经元，恒等激活，MSE）
    ↓
逻辑回归（1 个神经元，sigmoid 激活，BCE）
    ↓
LibTorch 单层 Linear + ReLU/Sigmoid/Tanh
    ↓
Eigen 手写 2 层 MLP + 反向传播 + SGD
    ↓
Batch GD vs SGD vs Mini-batch SGD 三种训练模式对比
```

---

## 核心概念：从感知机到 MLP

### 感知机——神经网络的原子单元

感知机 = 线性仿射变换 + 可选的激活函数：

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b  （加权和 + 偏置）
ŷ = φ(z)                            （激活函数输出）
```

| 激活函数      | 公式                              | 输出范围  | 适用场景                |
| ------------- | --------------------------------- | --------- | ----------------------- |
| Identity (无) | φ(z) = z                          | (-∞, +∞)  | 线性回归                |
| Sigmoid       | φ(z) = 1/(1+e⁻ᶻ)                  | (0, 1)    | 二分类概率输出          |
| ReLU          | φ(z) = max(0, z)                  | [0, +∞)   | 隐藏层默认选择          |
| Tanh          | φ(z) = (eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ)         | (-1, 1)   | 零中心化输出            |

**为什么必须要有非线性激活？** 如果所有层都只有线性变换，堆叠再多层也等价于单个线性层——无法学习弯曲的决策边界。激活函数引入非线性，让网络具备分层特征学习能力（浅层检测边缘 → 中层组合形状 → 深层识别物体）。

### 损失函数速查

| 任务类型   | 损失函数          | 公式                                                 | 输出激活   |
| ---------- | ----------------- | ---------------------------------------------------- | ---------- |
| 回归       | MSE               | L = (1/n)Σ(ŷ-y)²                                    | Identity   |
| 二分类     | Binary Cross-Entropy | L = -(y·log(ŷ) + (1-y)·log(1-ŷ))                 | Sigmoid    |
| 多分类     | Cross-Entropy     | L = -Σ(yᵢ·log(ŷᵢ))                                   | Softmax    |

### 梯度下降三兄弟

| 变体          | 更新方式               | 优点               | 缺点             | 何时用             |
| ------------- | ---------------------- | ------------------ | ---------------- | ------------------ |
| Batch GD      | 全量数据求梯度后更新   | 稳定，收敛路径平滑 | 每步慢，内存大   | 小数据集（<1000）  |
| SGD           | 每个样本更新一次       | 快，噪声助跳出局部 | 抖动大，步长不稳定 | 流式数据、快速试探 |
| Mini-batch SGD| 小批量（32-256）平均梯度 | 速度+稳定平衡       | 需调 batch size  | **默认推荐**         |

> **关键建议（来自 PDF 第 124-125 页）：**
> - epoch = 完整遍历数据集一次；step/iteration = 一次参数更新（一个 batch）
> - batch size 增大时，学习率通常可线性缩放（`lr *= batch_size/ref_batch`）
> - 每个 epoch 前 shuffle 数据以减少批次间相关性
> - 大 batch 提高吞吐但可能影响泛化——监控验证集指标

---

## 文件索引

### 一、回归基础（从零实现，无外部依赖）— PDF 第 111–115 页

| 文件                           | PDF 页    | 涵盖知识点                                                   | 依赖 |
| ------------------------------ | --------- | ------------------------------------------------------------ | ---- |
| `00_linear_regression.cpp`     | 111–112   | 线性回归 from Scratch：`y=2.5x+0.7+ε` 合成数据、MSE 损失、梯度下降更新 | STL  |
| `01_logistic_regression.cpp`   | 112–115   | 二分类逻辑回归 from Scratch：sigmoid、BCE 损失、2D 高斯簇数据集、准确率评估 | STL  |

### 二、LibTorch 神经元 — PDF 第 116–118 页

| 文件                   | PDF 页    | 涵盖知识点                                                             | 依赖     |
| ---------------------- | --------- | ---------------------------------------------------------------------- | -------- |
| `02_neuron_demo.cpp`   | 116–118   | `torch::nn::Linear` 线性层 + ReLU/Sigmoid/Tanh 激活、Kaiming uniform 初始化、CPU/CUDA 设备选择 | LibTorch |

### 三、从零实现 MLP（Eigen）— PDF 第 118–121 页

| 文件                 | PDF 页    | 涵盖知识点                                                                        | 依赖 |
| -------------------- | --------- | --------------------------------------------------------------------------------- | ---- |
| `03_mlp_eigen.cpp`   | 118–121   | `NeuralNetwork` 类（2→3→1）、`MatrixXd::Random` 随机初始化、`feedforward`（双 sigmoid 层）、`train`（MSE + 反向传播 + SGD 更新）、XOR 问题训练 | Eigen|

### 四、反向传播深度剖析 + SGD 变体 — PDF 第 121–125 页

| 文件                         | PDF 页    | 涵盖知识点                                                                                   | 依赖 |
| ---------------------------- | --------- | -------------------------------------------------------------------------------------------- | ---- |
| `04_backprop_training.cpp`   | 121–125   | 手写 2 层 MLP 反向传播（delta_out → delta_hid → 外积梯度）、Batch GD vs SGD vs Mini-batch SGD 三模式损失曲线对比 | Eigen|

---

## 代码演进：前向传播与反向传播

### 第 1 步：线性回归 from Scratch

```cpp
// 合成数据：y = 2.5x + 0.7 + ε, ε ~ N(0, 0.2)
std::mt19937 rng(42);
std::uniform_real_distribution<double> U(0.0, 1.0);
std::normal_distribution<double> N(0.0, 0.2);
for (int i = 0; i < n; ++i) {
    x[i] = U(rng);
    y[i] = 2.5 * x[i] + 0.7 + N(rng);
}

// 梯度下降训练
double w = 0.0, b = 0.0, lr = 0.1;
for (int epoch = 0; epoch < 1000; ++epoch) {
    for (int i = 0; i < n; ++i) {
        double y_hat = w * x[i] + b;       // 前向预测
        double resid = y_hat - y[i];        // 残差
        w -= lr * (2.0 * resid * x[i]);     // ∂MSE/∂w
        b -= lr * (2.0 * resid);            // ∂MSE/∂b
    }
}
```

### 第 2 步：逻辑回归 from Scratch

```cpp
// Sigmoid 激活
static inline double sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

// 二分类 BCE 训练
for (int epoch = 0; epoch < 3000; ++epoch) {
    double gw1 = 0, gw2 = 0, gb = 0;
    for (int i = 0; i < n; ++i) {
        double z = w1 * X[i][0] + w2 * X[i][1] + b;  // 线性得分
        double p = sigmoid(z);                         // 预测概率
        double diff = (p - y[i]) / n;                  // ∂BCE/∂z (取平均)
        gw1 += diff * X[i][0];
        gw2 += diff * X[i][1];
        gb  += diff;
    }
    w1 -= lr * gw1;  w2 -= lr * gw2;  b -= lr * gb;
}
// 决策边界: w1*x1 + w2*x2 + b = 0
```

### 第 3 步：LibTorch 神经元

```cpp
// 设备感知：优先 CUDA
torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

// 单层 Linear + ReLU 激活
torch::nn::Linear fc(5, 3);          // 5 输入特征 → 3 输出特征
torch::nn::init::kaiming_uniform_(fc->weight, std::sqrt(5.0));
torch::nn::init::zeros_(fc->bias);
fc->to(device);

auto x = torch::randn({4, 5}).to(device);   // batch=4, in_f=5
auto z = fc->forward(x);                     // [4, 3] logits
auto y = torch::relu(z);                     // ReLU 激活
auto ys = torch::sigmoid(z);                 // Sigmoid 激活
auto yt = torch::tanh(z);                    // Tanh 激活
```

### 第 4 步：Eigen MLP 完整实现

```cpp
class NeuralNetwork {
    MatrixXd W_ih;   // [hidden × input]  输入→隐藏层权重
    MatrixXd W_ho;   // [output × hidden] 隐藏→输出层权重
    VectorXd b_h;    // [hidden]           隐藏层偏置
    VectorXd b_o;    // [output]           输出层偏置
    double lr;

    // Sigmoid 及其导数
    VectorXd sigmoid(const VectorXd& x) { return 1.0/(1.0+(-x.array()).exp()); }
    VectorXd sigmoid_deriv(const VectorXd& y) { return y.array()*(1.0-y.array()); }

    VectorXd feedforward(const VectorXd& x) {
        VectorXd a1 = sigmoid(W_ih * x + b_h);    // 隐藏层
        return sigmoid(W_ho * a1 + b_o);           // 输出层
    }

    void train(const VectorXd& x, const VectorXd& t) {
        // Forward
        VectorXd a1 = sigmoid(W_ih * x + b_h);
        VectorXd y  = sigmoid(W_ho * a1 + b_o);
        // Backward (MSE + sigmoid)
        VectorXd d_o = sigmoid_deriv(y).cwiseProduct(t - y);  // output delta
        VectorXd e_h = W_ho.transpose() * d_o;                // hidden error
        VectorXd d_h = sigmoid_deriv(a1).cwiseProduct(e_h);   // hidden delta
        // SGD update
        W_ho += lr * (d_o * a1.transpose());   b_o += lr * d_o;
        W_ih += lr * (d_h * x.transpose());    b_h += lr * d_h;
    }
};
```

### 第 5 步：反向传播深度剖析

```cpp
// 单样本 (x, t) 训练——完整前向+反向+更新
double train_one_mse(const VectorXd& x, const VectorXd& t) {
    // ----- Forward -----
    VectorXd z1 = Wih * x + bh;      VectorXd a1 = sigmoid(z1);   // 隐藏层
    VectorXd z2 = Who * a1 + bo;     VectorXd y  = sigmoid(z2);   // 输出层

    // ----- Backprop deltas -----
    VectorXd delta_out = sigmoid_deriv(y).cwiseProduct(t - y);    // δ_out = σ'(y)·(t-y)
    VectorXd delta_hid = (Who.transpose() * delta_out).cwiseProduct(
                          sigmoid_deriv(a1));                      // δ_hid = (Wᵀδ_out)·σ'(a1)

    // ----- Gradients (outer products) -----
    MatrixXd dWho = delta_out * a1.transpose();   VectorXd dbo = delta_out;
    MatrixXd dWih = delta_hid * x.transpose();    VectorXd dbh = delta_hid;

    // ----- SGD update -----
    Who += lr * dWho;  bo += lr * dbo;  Wih += lr * dWih;  bh += lr * dbh;

    return 0.5 * (y - t).squaredNorm();  // 返回 MSE
}
```

---

## 编译与运行

### 环境要求

```bash
# 必需
C++17 编译器（GCC 11+ / Clang 14+）
CMake 3.22+
Eigen 3.4+          → apt install libeigen3-dev
LibTorch            → $HOME/Downloads/libtorch（仅 02_neuron_demo.cpp 需要）
```

### 编译

```bash
cd c++/deep_learning_cpp/build
cmake ..
cmake --build . --target <target_name> -j$(nproc)
```

### 运行示例

```bash
# 回归基础（无外部依赖，直接编译运行）
./build/chapter04/linear_regression
./build/chapter04/logistic_regression

# LibTorch 神经元（需要 LibTorch + CUDA 可选）
./build/chapter04/neuron_demo

# Eigen MLP（需要 Eigen）
./build/chapter04/mlp_eigen

# 反向传播 + SGD 变体（需要 Eigen）
./build/chapter04/backprop_training
```

---

## 技术速查

### LibTorch 常用 API（第 4 章涉及）

| API                                      | 用途                   |
| ---------------------------------------- | ---------------------- |
| `torch::Device(torch::kCUDA / torch::kCPU)` | 设备选择              |
| `torch::randn({batch, features})`        | 生成随机张量           |
| `torch::nn::Linear(in, out)`             | 全连接层               |
| `torch::nn::init::kaiming_uniform_(w, a)` | ReLU 适配初始化        |
| `torch::relu(z)`                         | ReLU 激活              |
| `torch::sigmoid(z)`                      | Sigmoid 激活           |
| `torch::tanh(z)`                         | Tanh 激活              |
| `tensor.to(device)`                      | 张量移到 GPU/CPU       |
| `tensor.sizes()`                         | 查看张量形状           |
| `tensor.slice(dim, start, end)`          | 切片操作               |

### Eigen 线性代数速查（第 4 章涉及）

| 操作                                       | 说明                 |
| ------------------------------------------ | -------------------- |
| `MatrixXd::Random(rows, cols)`             | 随机初始化矩阵       |
| `VectorXd::Random(size)`                   | 随机初始化向量       |
| `m1 * m2`                                  | 矩阵乘法             |
| `mat * vec`                                | 矩阵-向量乘法        |
| `mat.transpose()`                          | 矩阵转置             |
| `vec.array() * scalar`                     | 逐元素运算           |
| `v1.cwiseProduct(v2)`                      | 逐元素乘积（Hadamard）|
| `v1.dot(v2)`                               | 点积                 |
| `v.squaredNorm()`                          | L2 范数平方          |
| `1.0 / (1.0 + (-v.array()).exp())`         | 逐元素 sigmoid       |

### 常见激活函数导数

| 激活函数            | 导数的紧凑形式                    |
| ------------------- | --------------------------------- |
| Sigmoid: σ(z)       | σ'(z) = σ(z)·(1-σ(z))             |
| Tanh: tanh(z)       | tanh'(z) = 1 - tanh²(z)           |
| ReLU: max(0,z)      | ReLU'(z) = 1 if z>0 else 0        |

### 梯度下降核心公式

| 概念          | MSE + 线性输出                 | BCE + Sigmoid 输出              |
| ------------- | ------------------------------ | ------------------------------- |
| 损失 L         | (1/n)Σ(ŷ-y)²                  | -(y·log(ŷ)+(1-y)·log(1-ŷ))     |
| ∂L/∂ŷ        | 2(ŷ-y)/n                       | (ŷ-y)/(ŷ(1-ŷ))                |
| ∂L/∂z（合并后）| 2(ŷ-y)（无 σ' 因为激活是线性的） | (ŷ-y)（σ' 被分子分母抵消）      |
| 更新规则       | w -= lr · ∂L/∂w               | w -= lr · ∂L/∂w                |

> **BCE + Sigmoid 的优雅之处：** 将 `∂BCE/∂ŷ · ∂σ/∂z` 展开后，`ŷ(1-ŷ)` 的因子被抵消，最终得到简洁的 `∂L/∂z = (ŷ-y)`——与 MSE 加线性输出形式一致！

---

## PDF 完整内容对照

PDF 第 110–126 页（对应 PDF 文件第 142–160 页）的逐页纲要：

| 书本页   | 内容                                                                                          | 实现文件                        |
| -------- | --------------------------------------------------------------------------------------------- | ------------------------------- |
| 110      | 章节引入：感知机类比生物神经元、激活函数作用、学习流程                                        | --                              |
| 110-111  | 线性回归 vs 逻辑回归（图 4.1）、最小二乘 vs 交叉熵                                             | --                              |
| 111-112  | **线性回归 from Scratch**：合成数据 `y=2.5x+0.7+ε`、梯度下降、MSE 训练循环                     | `00_linear_regression.cpp`      |
| 112-115  | **逻辑回归 from Scratch**：sigmoid 函数、2D 高斯簇数据集、BCE 损失训练、准确率评估              | `01_logistic_regression.cpp`    |
| 115-116  | 深度网络层次特征学习（图 4.2）、从手工特征到自动特征学习                                      | --                              |
| 116-118  | **LibTorch 神经元实现**：`torch::nn::Linear`、Kaiming 初始化、CPU/CUDA 设备选择、ReLU/Sigmoid/Tanh 激活 | `02_neuron_demo.cpp`            |
| 118-119  | MLP 架构介绍（2→3→1，图 4.3）、Eigen 选择理由                                                  | --                              |
| 118-119  | `NeuralNetwork` 类声明：`weights_input_hidden`、`weights_hidden_output`、`sigmoid`、`feedforward`、`train` | `03_mlp_eigen.cpp`              |
| 119-120  | 构造函数（随机初始化打破对称性）+ `feedforward` 实现                                            | `03_mlp_eigen.cpp`              |
| 120-121  | `train()` 实现：前向 → 误差 → delta → 外积梯度 → SGD 更新                                      | `03_mlp_eigen.cpp`              |
| 121-122  | **反向传播深度剖析**：链式法则可视化（图 4.4）、δ_out 和 δ_hid 推导                           | `04_backprop_training.cpp`      |
| 122-123  | 单样本 backprop + SGD 代码（MSE + sigmoid）                                                    | `04_backprop_training.cpp`      |
| 123-124  | 梯度下降三变体对比（图 4.5）：Batch GD 平滑慢、SGD 抖动快、Mini-batch 平衡最优                 | `04_backprop_training.cpp`      |
| 124-125  | Mini-batch 最佳实践：shuffle、batch size 选择、学习率线性缩放、硬件效率                         | `04_backprop_training.cpp`      |
| 125-126  | 章节小结 + 课后问题 4 道                                                                       | --                              |
| 126      | 拓展阅读 + 参考答案                                                                           | --                              |

---

## 注意事项

### 外部库依赖

| 文件                      | 需要的库                  | 未安装时的行为                                  |
| ------------------------- | ------------------------- | ----------------------------------------------- |
| `00_linear_regression.cpp` | 无（纯 STL）              | 始终可编译运行                                  |
| `01_logistic_regression.cpp` | 无（纯 STL）            | 始终可编译运行                                  |
| `02_neuron_demo.cpp`       | LibTorch                  | CMake 找不到 Torch 时跳过                       |
| `03_mlp_eigen.cpp`         | Eigen 3.4+                | 需要 `libeigen3-dev`，否则编译失败              |
| `04_backprop_training.cpp` | Eigen 3.4+                | 需要 `libeigen3-dev`，否则编译失败              |

### 训练技巧速查（来自 PDF 本章）

| 技巧                   | 说明                                                             |
| ---------------------- | ---------------------------------------------------------------- |
| 随机初始化             | 用 `MatrixXd::Random` 或 Kaiming/Xavier 打破神经元对称性         |
| 学习率调优             | 太大→震荡/发散；太小→收敛极慢；典型范围 0.001–0.1                |
| 数据 shuffle           | 每个 epoch 前打乱顺序，减少梯度相关性                            |
| 数值稳定性             | BCE 中 log(0) 会 → -∞，用 `std::clamp(p, 1e-12, 1-1e-12)`        |
| batch size 经验法则    | 32、64、128、256、512、1024——选 GPU 显存能容纳的最大值           |
| 监控指标               | 不仅看训练 loss，也要看验证集 accuracy/loss 防过拟合              |

### 与后续章节的衔接

- **第 5 章 MLP**：将本章的 1 隐藏层扩展到多层 + 更多激活函数 + 更多优化器
- **第 6 章 CNN**：将 `nn::Linear` 换成 `nn::Conv2d`，本章的前向/反向逻辑完全复用
- **第 9 章 Transformers**：注意力机制是 `Linear` 层的组合，本章是基石
- **第 10 章 部署**：`torch::nn::Linear` → `torch.jit.trace` → ONNX 导出

---

## 拓展阅读

- **Deep Learning** (Goodfellow, Bengio, Courville): Ch. 6–8（MLP、优化、正则化）
- **Pattern Recognition and Machine Learning** (Bishop): Ch. 4–5（线性/逻辑模型、神经网络）
- **Stanford CS231n**: Backpropagation and Neural Network Optimization
- **PyTorch C++ API Tutorials**: https://pytorch.org/tutorials/advanced/cpp_frontend.html
- **A Gentle Introduction to Backpropagation**: 链式法则直觉 + 计算图可视化
- **Eigen 文档**: https://eigen.tuxfamily.org/dox/
