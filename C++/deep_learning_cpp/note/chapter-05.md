# 第 5 章：多层感知机（MLP）

基于 *Deep Learning with C++*（Packt，ISBN 9781835880036）第 5 章，第 128–167 页。本章是第 4 章的扩展——从单隐藏层 MLP 升级为**三种实现方式**（Eigen、CUDA、LibTorch）+ **完整的激活函数族**（Sigmoid + ReLU 两大族共 17 种）+ **六种优化器**（GD → RMSprop → Momentum → Adam → AdaGrad → AdaDelta）。

---

## 目录

1. [章节概述](#章节概述)
2. [三种实现方式对比](#三种实现方式对比)
3. [文件索引](#文件索引)
4. [核心代码演进](#核心代码演进)
5. [编译与运行](#编译与运行)
6. [技术速查](#技术速查)
7. [PDF 完整内容对照](#pdf-完整内容对照)
8. [注意事项](#注意事项)

---

## 章节概述

多层感知机（MLP）是深度学习的基石架构——多层互联神经元，每层应用加权变换和非线性激活，创建层次化表示以捕获数据中的复杂非线性模式。本章用 XOR 问题（线性不可分的经典案例）作为贯穿始终的示例，展示三种实现路径，并系统性地覆盖激活函数和优化器两大核心组件。

### 核心学习目标

| 目标                   | 说明                                                                         |
| ---------------------- | ---------------------------------------------------------------------------- |
| 三种实现方式比较       | Eigen（全手工 CPU）vs CUDA（GPU cuBLAS）vs LibTorch（自动求导），各有取舍     |
| Layer 类设计模式       | 封装权重/偏置/激活，`forward` → `backward`，链式堆叠构建 MLP                 |
| XOR 不可分性理解       | 单层感知机无法分离 XOR，MLP 通过隐藏层学习非线性决策边界                     |
| 激活函数全景           | Sigmoid 族（6 种）+ ReLU 族（7 种）+ 高级函数（4 种），共 17 种激活函数       |
| 优化器从浅到深         | 梯度下降 → RMSprop → Momentum → Adam → AdaGrad → AdaDelta，逐步构建           |

---

## 三种实现方式对比

本章用同一个 XOR 问题演示了三种 C++ MLP 实现，对应表 5.1（PDF 第 143 页）：

| 维度           | Eigen（原始 C++）            | CUDA（GPU 加速）              | LibTorch（PyTorch C++ API）   |
| -------------- | ---------------------------- | ----------------------------- | ----------------------------- |
| **库**         | Eigen 3.4+                   | cuBLAS, cuRAND               | LibTorch                      |
| **内存管理**   | 自动（C++ RAII）             | 显式（cudaMalloc/cudaFree）  | 自动                          |
| **处理单元**   | CPU                          | GPU                           | CPU / GPU（自动切换）         |
| **抽象级别**   | 中级（手动前向+反向）        | 低级（手动 kernel）           | 高级（只需写 forward）        |
| **矩阵运算**   | `Eigen::MatrixXd`            | `cublasSgemm`/`cublasSgemv`  | `torch::Tensor`               |
| **梯度计算**   | 手动实现反向传播             | 手动实现反向传播              | 自动求导（`loss.backward()`） |
| **复杂度**     | 中等                         | 复杂                          | 简单                          |
| **性能**       | 大计算慢                     | 大计算最快（10-100x）         | 良好平衡                      |
| **适用场景**   | 原型开发、嵌入式、可审计     | HPC、实时推理、自定义 kernel  | **生产部署默认推荐**          |
| **要求**       | C++/Eigen/8GB RAM            | CUDA Toolkit/NVIDIA GPU       | LibTorch/CMake/C++14+         |

### 选择策略（来自 PDF 第 141-143 页）

- **Eigen**: 当部署环境禁止外部框架依赖、嵌入式/IoT 严格内存预算、受监管领域需每行代码可审计
- **CUDA**: 当训练时间是主要瓶颈、需自定义 GPU kernel、实时大规模推理——但维护成本高
- **LibTorch**: **默认推荐**——自动求导、内置优化器、跨平台、PyTorch 生态。仅在框架无法满足特定可测量需求时偏离
- **混合模式**: LibTorch 为主 + 对特定瓶颈手写 CUDA kernel + 预处理用原始 C++

---

## 文件索引

### 一、MLP 实现（三种路径）— PDF 第 130–141 页

| 文件                        | PDF 页     | 涵盖知识点                                                    | 依赖           |
| --------------------------- | ---------- | ------------------------------------------------------------- | -------------- |
| `00_eigen_mlp_xor.cpp`      | 130–135    | `Layer` 类 + `MultilayerPerceptron` 类、手动反向传播、XOR 训练 | Eigen 3.4+     |
| `01_libtorch_mlp.cpp`       | 139–141    | `torch::nn::Linear` 三层 MLP、SGD 优化器、自动求导、CPU/CUDA   | LibTorch       |

### 二、激活函数全集 — PDF 第 144–152 页

| 文件                          | PDF 页     | 涵盖知识点                                                    | 依赖 |
| ----------------------------- | ---------- | ------------------------------------------------------------- | ---- |
| `02_activation_functions.cpp` | 144–152    | 17 种激活函数从零实现：Sigmoid 族（6）+ ReLU 族（7）+ 高级（4） | STL  |

### 三、优化器实现（六种）— PDF 第 152–164 页

| 文件                           | PDF 页     | 涵盖知识点                                                    | 依赖    |
| ------------------------------ | ---------- | ------------------------------------------------------------- | ------- |
| `03_optimizers_libtorch.cpp`   | 152–164    | GD / RMSprop / Momentum / Adam / AdaGrad / AdaDelta，LibTorch 实现 + 对比演示 | LibTorch |

---

## 核心代码演进

### 第 1 步：Eigen Layer 类（手动前向+反向）

```cpp
class Layer {
    Eigen::MatrixXd weights;     // [output_size × input_size]
    Eigen::VectorXd biases;      // [output_size]
    Eigen::MatrixXd activation;  // 前向中间值（用于反向传播的 gradient through ReLU）
    Eigen::MatrixXd input;       // 缓存输入（用于反向传播的外积梯度）

    Eigen::MatrixXd forward(const Eigen::MatrixXd& x) {
        input = x;
        activation = (weights * input + biases.replicate(1, input.cols()));
        output = relu(activation);
        return output;
    }

    Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_output, double lr) {
        // 通过 ReLU 的梯度: ∂L/∂z = grad_output * ReLU'(z)
        auto grad_activation = grad_output.array() * relu_derivative(activation).array();
        // 权重梯度 = δ_activation * input^T  (外积)
        auto grad_weights = grad_activation * input.transpose();
        auto grad_biases  = grad_activation.rowwise().sum();
        auto grad_input   = weights.transpose() * grad_activation; // 传给上一层
        // SGD 更新
        weights -= lr * grad_weights;
        biases  -= lr * grad_biases;
        return grad_input;
    }
};
```

### 第 2 步：MultilayerPerceptron 训练循环

```cpp
class MultilayerPerceptron {
    std::vector<Layer> layers;
    double learning_rate;

    void train(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            auto pred = forward(X);
            double loss = (pred - y).array().square().mean();  // MSE
            auto grad = 2.0 * (pred - y) / y.cols();          // dMSE/dpred
            for (int i = layers.size()-1; i >= 0; --i)        // 反向遍历
                grad = layers[i].backward(grad, learning_rate);
        }
    }
};

// XOR 训练: 网络 2→4→3→1, lr=0.01, 2000 epochs
// 输入: {{0,0},{0,1},{1,0},{1,1}} → 目标: {{0},{1},{1},{0}}
```

### 第 3 步：LibTorch MLP（只需写 forward！）

```cpp
struct MLP : torch::nn::Module {
    torch::nn::Linear layer1{nullptr}, layer2{nullptr}, layer3{nullptr};

    MLP() {
        layer1 = register_module("layer1", torch::nn::Linear(2, 4));
        layer2 = register_module("layer2", torch::nn::Linear(4, 3));
        layer3 = register_module("layer3", torch::nn::Linear(3, 1));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(layer1->forward(x));
        x = torch::relu(layer2->forward(x));
        x = layer3->forward(x);
        return x;
    }
};

// 训练循环——算法化的是标准三步骤：
for (epoch...) {
    optimizer.zero_grad();          // ① 清零旧梯度
    auto loss = torch::mse_loss(model->forward(X), y);
    loss.backward();                // ② 自动求导 + 反向传播
    optimizer.step();               // ③ 应用梯度更新
}
```

### 第 4 步：优化器核心公式速查

| 优化器    | 要点                                                                                     | 关键代码                                                        |
| --------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| GD        | `x -= lr * grad`                                                                         | 基准，只靠梯度方向                                             |
| RMSprop   | 用梯度平方的**指数移动平均**归一化                                                        | `cache=β*cache+(1-β)*g²; x-=lr*g/(√cache+ε)`                   |
| Momentum  | 类似物理速度：累积历史梯度方向                                                           | `v=β*v+lr*g; x-=v`                                              |
| Adam      | **RMSprop + Momentum 的结合体** + 偏置校正                                                 | `m=β1*m+(1-β1)*g; v=β2*v+(1-β2)*g²; m̂,v̂校正; x-=lr*m̂/(√v̂+ε)` |
| AdaGrad   | 累积全部历史梯度平方→学习率**单调递减**                                                   | `cache+=g²; x-=lr*g/(√cache+ε)`                                 |
| AdaDelta  | 无需手动设学习率，用更新量/梯度量的比值自调节                                            | `Δ=√(Δavg+ε)/√(gavg+ε)*g; x-=Δ`                                |

---

## 编译与运行

### 环境要求

```bash
# 基础
C++17 编译器
CMake 3.22+

# 00_eigen_mlp_xor.cpp
Eigen 3.4+  →  apt install libeigen3-dev

# 01_libtorch_mlp.cpp + 03_optimizers_libtorch.cpp
LibTorch     →  $HOME/Downloads/libtorch

# 02_activation_functions.cpp
无外部依赖，纯 STL
```

### 编译

```bash
cd c++/deep_learning_cpp/build
cmake ..
cmake --build . --target <target_name> -j$(nproc)
```

### 运行

```bash
./build/chapter05/eigen_mlp_xor        # Eigen 从零实现 MLP
./build/chapter05/libtorch_mlp         # LibTorch MLP（自动求导）
./build/chapter05/activation_functions  # 17 种激活函数表格对比
./build/chapter05/optimizers           # 6 种优化器在 f(x)=(x-3)² 上的收敛对比
```

---

## 技术速查

### 激活函数速查表（17 种全覆盖）

#### Sigmoid 族

| 函数              | 公式                                    | 输出范围  | 关键特性                   |
| ----------------- | --------------------------------------- | --------- | -------------------------- |
| Sigmoid           | 1/(1+e⁻ˣ)                               | (0, 1)    | 概率解释，梯度消失         |
| Tanh              | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)                      | (-1, 1)   | 零中心，比 sigmoid 梯度强  |
| Softplus          | ln(1+eˣ)                                | (0, ∞)    | ReLU 的平滑近似            |
| Softsign          | x/(1+|x|)                               | (-1, 1)   | 多项式饱和，比 tanh 简单   |

#### ReLU 族

| 函数       | 公式                                  | 输出范围  | 关键特性                    |
| ---------- | ------------------------------------- | --------- | --------------------------- |
| ReLU       | max(0, x)                             | [0, ∞)    | 默认隐藏层选择，计算快      |
| Leaky ReLU | max(αx, x), α=0.01                    | (-∞, ∞)   | 避免"死亡 ReLU"              |
| PReLU      | max(αx, x), α 可学习                  | (-∞, ∞)   | Leaky ReLU 的泛化           |
| ELU        | x>0?x:α(eˣ-1)                         | (-α, ∞)   | 平滑负值，自正则化          |
| SELU       | λx if x>0 else λα(eˣ-1)               | 自归一化  | λ≈1.0507, α≈1.6733          |
| GELU       | 0.5x(1+tanh(√(2/π)(x+0.044715x³)))   | (-∞, ∞)   | Transformer 默认（BERT/GPT） |
| RReLU      | max(αx, x), α~U(l,u) 训练中随机       | (-∞, ∞)   | 训练正则化                  |

#### 高级函数

| 函数         | 公式                                       | 特点                          |
| ------------ | ------------------------------------------ | ----------------------------- |
| Swish        | x·σ(βx)                                    | 自门控、非单调，比 ReLU 好    |
| Mish         | x·tanh(softplus(x))                        | 自正则化、梯度更平滑          |
| Hard Swish   | x·clamp((x+3)/6, 0, 1)                     | Swish 的快速近似（移动端）    |
| Hard Tanh    | clamp(x, -1, 1)                            | Tanh 的分段线性近似           |

### 优化器选择指南

| 场景                         | 推荐优化器     | 理由                                          |
| ---------------------------- | -------------- | --------------------------------------------- |
| **默认首选**                 | Adam           | 自适应学习率 + 动量，超参不敏感，大多数任务开箱即用 |
| 稀疏数据                     | AdaGrad        | 对低频特征自动放大学习率                      |
| 需要手动控制学习率衰减        | RMSprop        | 指数衰减窗口，适合非平稳目标                  |
| 逃离鞍点 / 浅区域             | Momentum       | 方向性加速，克服纯 GD 的停滞                  |
| 不理解如何设学习率            | AdaDelta        | 无需手动设置学习率                            |
| 最简单基线 / 教学用途         | Gradient Descent | 基础框架，后续优化器都由此衍生                |

---

## PDF 完整内容对照

| 书本页  | 内容                                                              | 实现文件                         |
| ------- | ----------------------------------------------------------------- | -------------------------------- |
| 128–129 | 章节引入、技术要求、MLP 架构回顾（图 5.1）                        | --                               |
| 129–130 | XOR 问题为什么需要隐藏层（图 5.2：AND/OR/NAND vs XOR）            | `00_eigen_mlp_xor.cpp`           |
| 131–132 | `Layer` 类实现：权重矩阵、ReLU 前向+后向                           | `00_eigen_mlp_xor.cpp`           |
| 132–134 | `MultilayerPerceptron` 类：链式 forward、MSE 训练循环              | `00_eigen_mlp_xor.cpp`           |
| 134–135 | 使用的库说明（Eigen、std::vector、cmath）                          | `00_eigen_mlp_xor.cpp`           |
| 135–139 | CUDA MLP：`CudaLayer`（cublasSgemm + curand）+ `CudaMLP`          | （对应 chapter03 CUDA 知识）     |
| 139–141 | LibTorch MLP：`MLP` 结构体 + SGD 三步训练循环                      | `01_libtorch_mlp.cpp`            |
| 141–143 | 三种方案对比指南（表 5.1）+ 混合模式建议                           | --                               |
| 144–146 | Sigmoid 族：Sigmoid, Tanh, Softplus, Softsign                     | `02_activation_functions.cpp`    |
| 146–149 | ReLU 族：ReLU, Leaky ReLU, PReLU, ELU, SELU, GELU, RReLU          | `02_activation_functions.cpp`    |
| 150–152 | 高级函数：Swish, Mish, Hard Swish, Hard Tanh                       | `02_activation_functions.cpp`    |
| 152–154 | 损失函数与优化问题（图 5.3 局部/全局最小）                         | --                               |
| 154–155 | 梯度下降（GD）— `gradient_descent()`                               | `03_optimizers_libtorch.cpp`     |
| 155–156 | RMSprop — `rmsprop_optimize()`                                     | `03_optimizers_libtorch.cpp`     |
| 156–158 | Momentum — `momentum_optimize()`                                   | `03_optimizers_libtorch.cpp`     |
| 158–160 | Adam — `adam_optimize()` + 偏置校正推导                            | `03_optimizers_libtorch.cpp`     |
| 160–162 | AdaGrad — `adagrad_optimize()`                                     | `03_optimizers_libtorch.cpp`     |
| 162–164 | AdaDelta — `adadelta_optimize()` + 无学习率机制                    | `03_optimizers_libtorch.cpp`     |
| 164–166 | 章节小结 + 训练关键概念（LR 策略、正则化、初始化、batch）          | --                               |
| 166–167 | 拓展阅读（Rumelhart, Glorot, Kingma, Eigen 文档等）               | --                               |

---

## 注意事项

### 外部库依赖

| 文件                          | 需要库             | 未安装时的行为                 |
| ----------------------------- | ------------------ | ------------------------------ |
| `00_eigen_mlp_xor.cpp`        | Eigen 3.4+         | CMake 找不到时跳过             |
| `01_libtorch_mlp.cpp`         | LibTorch           | CMake 找不到 Torch 时跳过      |
| `02_activation_functions.cpp` | **无（纯 STL）**   | 始终可编译                     |
| `03_optimizers_libtorch.cpp`  | LibTorch           | CMake 找不到 Torch 时跳过      |

### 激活函数选择实践建议

- **隐藏层默认：ReLU** — 简单、快速、大部分场景足够
- **如果遇到"死亡 ReLU"（输出恒为 0）：切换到 Leaky ReLU (α=0.01) 或 ELU**
- **Transformer / NLP：GELU**（BERT、GPT 标准）
- **输出层二分类：Sigmoid**；多分类：Softmax（本章未涉及）
- **移动端/边缘：Hard Swish** — 速度快，接近 Swish 精度

### 优化器超参默认值

| 优化器    | 推荐学习率 | 其他关键参数                 |
| --------- | ---------- | ---------------------------- |
| GD        | 0.01       | --                           |
| RMSprop   | 0.01       | decay=0.99, ε=1e-8           |
| Momentum  | 0.01       | momentum=0.9                 |
| **Adam**  | **0.001**  | β₁=0.9, β₂=0.999, ε=1e-8    |
| AdaGrad   | 0.01       | ε=1e-8                       |
| AdaDelta  | (无需设置) | ρ=0.95, ε=1e-8               |

### 本章与前后章节的关系

- **第 4 章**：建立单个神经元 + 手写反向传播的直觉基础
- **本章**：扩展到多层 + 三种实现方式 + 激活函数全景 + 优化器工具包
- **第 6 章 CNN**：将本章的 `Linear` 层换为 `Conv2d`，激活函数和优化器完全复用
- **第 3 章 CUDA**：本章的 `CudaLayer` 直接使用第 3 章学到的 kernel/cuBLAS 知识

---

## 拓展阅读

- **Eigen 文档**: https://eigen.tuxfamily.org/dox/
- **LibTorch C++ API**: https://pytorch.org/cppdocs/
- **Kingma & Ba (2014)** — Adam 原论文: arXiv:1412.6980
- **Zeiler (2012)** — AdaDelta: arXiv:1212.5701
- **Rumelhart, Hinton & Williams (1986)** — 反向传播原始论文: Nature 323
- **Glorot & Bengio (2010)** — 初始化策略与激活函数选择
- **Ramachandran et al. (2017)** — Swish 激活函数: arXiv:1710.05941
- **Misra (2019)** — Mish 激活函数: arXiv:1908.08681
- **Clevert et al. (2015)** — ELU: arXiv:1511.07289
- **CUDA Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **cuBLAS 文档**: https://docs.nvidia.com/cuda/cublas/
