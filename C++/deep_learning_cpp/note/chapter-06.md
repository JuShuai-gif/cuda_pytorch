# 第 6 章：卷积神经网络（CNN）

基于 *Deep Learning with C++*（Packt，ISBN 9781835880036）第 6 章，第 169–208 页。本章从卷积运算的数学基础出发，用三种实现（嵌套循环 CPU → Eigen 矩阵优化 → CUDA cuBLAS）逐步揭示 CNN 的底层原理，再上升到 VGG-16 图像分类和 U-Net 图像分割两大实战，最后覆盖 8 种图像增强技术。

---

## 目录

1. [章节概述](#章节概述)
2. [三种实现方式对比](#三种实现方式对比)
3. [文件索引](#文件索引)
4. [卷积运算：从数学到代码](#卷积运算从数学到代码)
5. [编译与运行](#编译与运行)
6. [技术速查](#技术速查)
7. [PDF 完整内容对照](#pdf-完整内容对照)
8. [注意事项](#注意事项)

---

## 章节概述

CNN 专为处理图像、视频等网格状数据设计。核心思想：一个小的滤波器（kernel）在输入上滑动，逐位置计算点积，生成特征图。这种「共享权重 + 局部连接」的设计天然保留了空间关系，同时大幅减少参数（与全连接层相比）。层次化的特征学习——浅层检测边缘、中层组合纹理、深层识别物体——使得 CNN 成为计算机视觉的基石

### 核心学习目标

| 目标                 | 说明                                                         |
| -------------------- | ------------------------------------------------------------ |
| 卷积数学基础         | `(f ∗ g)(x) = Σf(τ)·g(x-τ)`，核 + 步长 + 填充控制输出尺寸  |
| 三种卷积实现递进     | 嵌套循环（O(F·H·W·K²)）→ im2col + 矩阵乘法（5-6x 加速）→ cuBLAS GPU |
| VGG-16 架构          | 5 个卷积块（通道 64→128→256→512→512）+ 3 个全连接层          |
| U-Net 分割           | 编码器-解码器 + skip connections，sigmoid 输出概率图         |
| 8 种图像增强技术     | 旋转、平移、裁剪、缩放、变焦、翻转、填充、重采样              |
| CNN 术语             | 池化（Max/Avg）、步长、特征图、感受野                        |

### CNN vs MLP：关键区别

| 特性         | MLP（全连接）            | CNN（卷积）                         |
| ------------ | ------------------------ | ----------------------------------- |
| 连接方式     | 每个神经元连到所有输入   | 局部感受野（kernel 只连 K×K 区域）  |
| 参数共享     | 每个连接独立权重         | 同一 filter 在所有位置共享权重      |
| 空间感知     | 破坏（输入需展平）       | 保留二维空间结构                    |
| 参数数量     | 巨大（W×H×C × 下一层）  | 少（K² × C_in × C_out）             |
| 平移不变性   | 无                       | 池化层提供                          |

---

## 三种实现方式对比

| 维度         | 嵌套循环 CPU               | Eigen（im2col+矩阵乘法）     | CUDA（cuBLAS）                    | LibTorch（Conv2d）              |
| ------------ | -------------------------- | ---------------------------- | --------------------------------- | ------------------------------- |
| **代码复杂度** | 4 层嵌套循环               | 1 行 `filters * inputMatrix` | kernel + cublasSgemm + cudaMemcpy | 1 行 `conv->forward(x)`          |
| **抽象级别** | 低级（直接写数学）         | 中级（利用 BLAS）            | 低级（手动内存管理）              | 高级（框架封装）                |
| **性能**     | 基准                       | CPU 5-6x 加速                | GPU 10-100x 加速                  | 自动最优                        |
| **适用场景** | 理解底层原理、教学         | CPU 生产环境、嵌入式          | HPC、自定义 GPU kernel            | **生产默认推荐**                |
| **关键操作** | `sum += input[i+k][j+l] * filter[k][l]` | `filters * im2col(input)` | `cublasSgemm(handle, N, N, ...)`  | `torch::nn::Conv2d(in, out, k)` |

---

## 文件索引

### 一、卷积层实现（三种路径）— PDF 第 173–183 页

| 文件                          | PDF 页     | 涵盖知识点                                                    | 依赖          |
| ----------------------------- | ---------- | ------------------------------------------------------------- | ------------- |
| `00_convolution_cpu.cpp`      | 173–176    | `ConvolutionalLayer` 类、4 层嵌套循环、ReLU 激活              | STL           |
| `01_convolution_eigen.cpp`    | 176–178    | `im2col` 转换、`OptimizedConvolutionalLayer`、单行矩阵乘法替代所有循环 | Eigen 3.4+    |
| `02_convolution_libtorch.cpp` | 182–183    | `ConvNetImpl` 结构体、`torch::nn::Conv2d` + ReLU、padding 参数 | LibTorch      |

### 二、VGG-16 图像分类 — PDF 第 183–186 页

| 文件                  | PDF 页     | 涵盖知识点                                                    | 依赖     |
| --------------------- | ---------- | ------------------------------------------------------------- | -------- |
| `03_vgg16_mnist.cpp`  | 183–186    | VGG 风格网络（3 个卷积块 1→8→16→32）、flatten → Linear 分类头 | LibTorch |

### 三、图像增强（8 种技术）— PDF 第 191–202 页

| 文件                          | PDF 页     | 涵盖知识点                                                    | 依赖     |
| ----------------------------- | ---------- | ------------------------------------------------------------- | -------- |
| `04_image_augmentation.cpp`   | 191–202    | 旋转、平移、裁剪、缩放、变焦、翻转、填充、重采样 + 组合流水线 | LibTorch |

### 四、CNN 术语速查 — PDF 第 203–206 页

| 文件                       | PDF 页     | 涵盖知识点                                                    | 依赖     |
| -------------------------- | ---------- | ------------------------------------------------------------- | -------- |
| `05_cnn_terminology.cpp`   | 203–206    | Max/Avg Pooling、Stride 对比、特征图概念、感受野计算          | LibTorch |

---

## 卷积运算：从数学到代码

### 数学定义

```
(f ∗ g)(x) = Στ f(τ) · g(x - τ)
```

在 CNN 中：`f` = 输入图像，`g` = 卷积核（kernel/filter），`*` = 卷积操作。

### 第 1 步：嵌套循环实现

```cpp
// 4 层嵌套循环：filter → i(row) → j(col) → k,l(kernel element)
for (auto& filter : filters) {
    for (int i = 0; i <= H - K; i += stride) {
        for (int j = 0; j <= W - K; j += stride) {
            float sum = 0;
            for (int k = 0; k < K; k++)
                for (int l = 0; l < K; l++)
                    sum += input[i+k][j+l] * filter[k][l];
            featureMap.push_back(max(0.0f, sum));  // ReLU
        }
    }
}
// 时间复杂度: O(F × H_out × W_out × K²)
```

### 第 2 步：im2col + 矩阵乘法

```cpp
// im2col: 将每个 K×K 滑动窗口展平为一列
// 输入: [H, W] → 输出矩阵: [K², H_out × W_out]

Matrix im2col(const Matrix& input, int kSize, int stride) {
    int outRows = (input.rows() - kSize) / stride + 1;
    int outCols = (input.cols() - kSize) / stride + 1;
    Matrix result(kSize * kSize, outRows * outCols);
    int colIdx = 0;
    for (int i = 0; i <= H - kSize; i += stride)
        for (int j = 0; j <= W - kSize; j += stride) {
            int rowIdx = 0;
            for (int ki = 0; ki < kSize; ki++)
                for (int kj = 0; kj < kSize; kj++)
                    result(rowIdx++, colIdx) = input(i+ki, j+kj);
            colIdx++;
        }
    return result;
}

// 整个前向传播压缩为一行！
Matrix forward(const Matrix& input) {
    Matrix inputMatrix = im2col(input, kernelSize, stride);
    Matrix output = filters * inputMatrix;  // ← 一行替代所有嵌套循环
    return output.cwiseMax(0.0);  // ReLU
}
```

### 第 3 步：LibTorch Conv2d

```cpp
struct ConvNetImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::ReLU relu{nullptr};

    ConvNetImpl(int64_t in_c, int64_t out_c, int64_t ks)
        : conv1(torch::nn::Conv2dOptions(in_c, out_c, ks).padding(0)) {
        register_module("conv1", conv1);
        register_module("relu", relu);
    }

    torch::Tensor forward(torch::Tensor x) {
        return relu(conv1(x));
    }
};
TORCH_MODULE(ConvNet);
```

### 输出尺寸计算公式

```
out_height = (in_height - kernel_size + 2 × padding) / stride + 1
out_width  = (in_width  - kernel_size + 2 × padding) / stride + 1
```

---

## 编译与运行

### 环境要求

```bash
# 基础
C++17 编译器 + CMake 3.22+

# 00_convolution_cpu.cpp: 无外部依赖
# 01_convolution_eigen.cpp: Eigen 3.4+ (apt install libeigen3-dev)
# 02-05: LibTorch ($HOME/Downloads/libtorch)
```

### 编译

```bash
cd build && cmake .. && cmake --build . --target convolution_cpu -j$(nproc)
```

### 运行

```bash
./build/chapter06/convolution_cpu         # CPU 嵌套循环卷积
./build/chapter06/convolution_eigen       # Eigen im2col 矩阵卷积
./build/chapter06/convolution_libtorch    # LibTorch Conv2d 演示
./build/chapter06/vgg16_mnist            # VGG 风格 MNIST 网络
./build/chapter06/image_augmentation      # 8 种图像增强技术
./build/chapter06/cnn_terminology        # Pooling/Stride/特征图/感受野
```

---

## 技术速查

### 卷积参数速算

| 参数        | 含义                                | 典型值          |
| ----------- | ----------------------------------- | --------------- |
| kernel_size | 滤波器大小                          | 3, 5, 7         |
| stride      | 滤波器每次移动的像素数              | 1（保尺寸）, 2（减半） |
| padding     | 边框零填充数                        | kernel/2（保持尺寸不变） |
| dilation    | 空洞卷积（kernel 元素间隔）         | 1（普通）, 2+（扩大感受野） |

### VGG-16 架构速查

| Block | 层             | 输出尺寸       | 通道数 |
| ----- | -------------- | -------------- | ------ |
| 1     | Conv3×3 ×2     | 224×224 → 112×112 | 64     |
| 2     | Conv3×3 ×2     | 112×112 → 56×56   | 128    |
| 3     | Conv3×3 ×3     | 56×56 → 28×28     | 256    |
| 4     | Conv3×3 ×3     | 28×28 → 14×14     | 512    |
| 5     | Conv3×3 ×3     | 14×14 → 7×7       | 512    |
| Head  | FC 4096, 4096, 1000 | 1×1×1000    | -      |

**空间减半**通过 MaxPool2d(2, stride=2) 在各 block 末尾实现。

### U-Net 架构要点

| 组件          | 操作                             | 尺寸变化                  |
| ------------- | -------------------------------- | ------------------------- |
| Encoder       | Conv3×3(×2) + ReLU + MaxPool2d(2) | H×W → H/2×W/2, 通道翻倍   |
| Bottleneck    | Conv3×3(×2) + ReLU               | 最小空间，最深特征        |
| Decoder       | Up-conv2×2 + Conv3×3(×2) + ReLU  | H×W → 2H×2W, 通道减半     |
| Skip Connect  | 将 encoder 同层特征图拼接到 decoder | 恢复空间细节              |
| Output        | Conv1×1 → Sigmoid                | 输出与输入同尺寸的概率图  |

### 8 种图像增强技术

| 技术       | LibTorch 实现                                              | 效果                     |
| ---------- | ---------------------------------------------------------- | ------------------------ |
| 旋转       | `affine_grid` + `grid_sample`                              | 方向不变性               |
| 平移       | `pad` + `slice`                                            | 位置不变性               |
| 裁剪       | `tensor.index({Slice(), Slice(x,x+h), Slice(y,y+w)})`     | 部分可见性鲁棒性         |
| 缩放       | `interpolate(..., kBilinear)`                              | 尺度不变性               |
| 变焦       | 中心裁剪 + `interpolate` 恢复原尺寸                         | 远近适应性               |
| 翻转       | `tensor.flip(-1)`（水平） / `tensor.flip(-2)`（垂直）      | 镜面对称不变性           |
| 填充       | `functional::pad(..., kConstant/kReflect/kReplicate)`     | 边界信息保留             |
| 重采样     | `interpolate(..., nearest/bilinear/bicubic)`               | 多分辨率适应性           |

---

## PDF 完整内容对照

| 书本页  | 内容                                                              | 实现文件                       |
| ------- | ----------------------------------------------------------------- | ------------------------------ |
| 170–171 | CNN 发展史（Hubel & Wiesel → Fukushima → LeCun LeNet-5 → AlexNet） | --                             |
| 171–173 | 卷积数学基础（卷积公式、2D 卷积图解、RGB 多通道）                  | `00_convolution_cpu.cpp`       |
| 173–176 | 基本 CPU 卷积层（`ConvolutionalLayer`，4 层嵌套循环，时间复杂度）  | `00_convolution_cpu.cpp`       |
| 176–178 | Eigen 矩阵实现（`im2col`，`OptimizedConvolutionalLayer`，5-6x 加速） | `01_convolution_eigen.cpp`     |
| 178–181 | **CUDA 实现**（`CudaConvolutionalLayer`，cuBLAS：cublasSgemm/cublasSaxpy/cublasSscal） | （参考 chapter03 CUDA 知识）   |
| 182–183 | **LibTorch Conv2d**（`ConvNetImpl`，GPU 自动切换）                  | `02_convolution_libtorch.cpp`  |
| 183–186 | **VGG-16 图像分类**（5 卷积块 + classifier head，MNIST）             | `03_vgg16_mnist.cpp`           |
| 187–191 | **U-Net 图像分割**（编码器-解码器 + skip connections，sigmoid 输出） | （架构参考，见本书 GitHub）    |
| 191–192 | 旋转                                                                 | `04_image_augmentation.cpp`    |
| 192–193 | 平移                                                                 | `04_image_augmentation.cpp`    |
| 193–194 | 裁剪                                                                 | `04_image_augmentation.cpp`    |
| 194–196 | 缩放                                                                 | `04_image_augmentation.cpp`    |
| 196–198 | 变焦（Zoom）                                                         | `04_image_augmentation.cpp`    |
| 198–199 | 翻转（水平/垂直）                                                    | `04_image_augmentation.cpp`    |
| 199–200 | 填充（constant/reflect/replicate）                                   | `04_image_augmentation.cpp`    |
| 200–202 | 重采样（nearest/bilinear/bicubic）+ 组合变换流水线                   | `04_image_augmentation.cpp`    |
| 203–204 | 池化（Max/Avg Pooling）、步长                                        | `05_cnn_terminology.cpp`       |
| 204–206 | 特征图概念、感受野计算                                               | `05_cnn_terminology.cpp`       |
| 206–208 | 章节小结 + 拓展阅读                                                  | --                             |

---

## 注意事项

### 外部库依赖

| 文件                          | 依赖                    | 未安装时的行为                     |
| ----------------------------- | ----------------------- | ---------------------------------- |
| `00_convolution_cpu.cpp`      | **无（纯 STL）**        | 始终可编译                         |
| `01_convolution_eigen.cpp`    | Eigen 3.4+              | CMake 找不到时跳过                 |
| `02_convolution_libtorch.cpp` | LibTorch                | CMake 找不到时跳过                 |
| `03_vgg16_mnist.cpp`          | LibTorch                | CMake 找不到时跳过                 |
| `04_image_augmentation.cpp`   | LibTorch                | CMake 找不到时跳过                 |
| `05_cnn_terminology.cpp`      | LibTorch                | CMake 找不到时跳过                 |

### 输出尺寸速查表

以 28×28 MNIST 输入、3×3 kernel、stride=1、padding=0 为例：

| 层           | 参数                            | 输出尺寸           |
| ------------ | ------------------------------- | ------------------ |
| Conv2d       | kernel=3, stride=1, pad=0       | 26×26              |
| Conv2d       | kernel=3, stride=1, pad=1       | 28×28（保持不变）  |
| MaxPool2d    | kernel=2, stride=2              | 14×14（减半）      |
| Conv2d       | kernel=5, stride=2, pad=0       | 12×12              |

### VGG 风格 MNIST 架构（03_vgg16_mnist.cpp 实际使用的）

```
Input: [batch, 1, 28, 28]
  Block1: Conv2d(1→8, k=3, p=1) + ReLU + MaxPool2d(2) → [batch, 8, 14, 14]
  Block2: Conv2d(8→16, k=3, p=1) + ReLU + MaxPool2d(2) → [batch, 16, 7, 7]
  Block3: Conv2d(16→32, k=3, p=1) + ReLU + MaxPool2d(2) → [batch, 32, 3, 3]
  Flatten: [batch, 288]
  FC1: Linear(288→64) + ReLU + Dropout(0.5) → [batch, 64]
  FC2: Linear(64→10) → [batch, 10]  (10 类 logits)
```

### 增强技术注意事项

- **旋转**后的空白角落需要填充或裁剪
- **平移**会导致部分图像移出边界，需要 padding
- **缩放/变焦**后需用 `interpolate` 恢复原始尺寸
- **组合变换**的顺序很重要：平移→旋转→翻转→重采样 是推荐的确定性顺序
- LibTorch 的 `torchvision::transforms` 在部分版本不可用，备用方案用 `affine_grid` + `grid_sample`

---

## 拓展阅读

- **CS231n** (Stanford): https://cs231n.github.io/convolutional-networks/ — CNN 经典课程
- **NVIDIA CUDA C++ Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **Original U-Net paper** (Ronneberger et al., 2015): arXiv:1505.04597
- **VGG paper** (Simonyan & Zisserman, 2014): arXiv:1409.1556
- **AlexNet paper** (Krizhevsky et al., 2012): NIPS 2012
- **LeNet-5 paper** (LeCun et al., 1998): Gradient-based learning applied to document recognition
- **im2col 详解**: https://hackmd.io/@machine-learning/blog-post-cnnumpy-fast
- **PyTorch C++ API**: https://pytorch.org/cppdocs/
- **Semantic/Instance/Panoptic Segmentation survey**: arXiv:2111.10250
