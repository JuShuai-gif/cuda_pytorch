# Deep Learning with C++ — 学习知识库

基于 **《Deep Learning with C++》**（Packt，ISBN 9781835880036）构建的系统化 C++ 深度学习学习项目，集**学习笔记 + 示例代码**于一体。

---

## 目录结构

```text
c++/deep_learning_cpp/
├── note/                          # 学习笔记（对应全书 14 章）
│   ├── chapter-02.md              # 第 2 章：C++ 数据准备与预处理
│   ├── chapter-03.md              # 第 3 章：CUDA 与 GPU 加速
│   ├── chapter-04.md              # 第 4 章：基础神经网络
│   ├── chapter-05.md              # 第 5 章：多层感知机（MLP）
│   ├── chapter-06.md              # 第 6 章：卷积神经网络（CNN）
│   ├── chapter-07.md              # 第 7 章：RNN 与 LSTM
│   ├── chapter-08.md              # 第 8 章：生成模型（Autoencoder/VAE/GAN）
│   ├── chapter-09.md              # 第 9 章：Transformer 与注意力机制
│   ├── chapter-10.md              # 第 10 章：模型部署（TorchScript/ONNX/服务）
│   ├── chapter-11.md              # 第 11 章：部署模型的调试与重训练
│   ├── chapter-12.md              # 第 12 章：部署模型的监控
│   ├── chapter-13.md              # 第 13 章：可解释性与透明度
│   └── chapter-14.md              # 第 14 章：附录与推荐阅读
├── src/                           # 示例代码（每章独立可编译运行）
│   ├── chapter-01/                # 第 1 章：C++ 深度学习入门与环境设置
│   ├── chapter-02/                # 第 2 章：C++ 数据准备与预处理
│   ├── ...                        #
│   └── chapter-14/                # 第 14 章：（占位）
├── .clang-format                  # 代码格式化配置
├── .clangd                        # clangd LSP 配置
├── .gitignore                     # Git 忽略规则
├── CMakeLists.txt                 # 顶层构建文件
└── README.md                      # 本文档
```

---

## 环境要求

| 组件 | 最低版本 | 推荐版本 |
|------|----------|----------|
| C++ 标准 | C++17 | C++17 |
| 编译器 | GCC 8+ / Clang 7+ | GCC 11+ / Clang 14+ |
| CMake | 3.22+ | 3.22+ |
| CUDA Toolkit | 11.x+ | 12.x（第 3 章 CUDA 示例需要） |
| Eigen 3.4+ | apt install libeigen3-dev | —（第 4~7、13 章部分示例需要） |
| OpenCV 4.x | apt install libopencv-dev | —（第 2、13 章部分示例可选） |
| Armadillo 12+ | apt install libarmadillo-dev | —（第 2 章部分示例可选） |
| OS | Linux | Ubuntu 20.04+ / 22.04+ |

---

## 核心库安装

### 1. LibTorch（PyTorch C++ 前端）

LibTorch 是 PyTorch 的 C++ 发行版，提供 `torch::Tensor`、`torch::nn`、`torch::jit` 等 C++ API。本项目第 1、2、4~10、13 章的部分示例依赖 LibTorch。

#### 下载预编译包

从 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择对应的 C++/Java 分发版下载。本项目使用 CUDA 12.x 版本：

```bash
# 进入下载目录
cd ~/Downloads

# 下载 LibTorch（CUDA 12.1，CXX11 ABI 版本）
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu121.zip

# 解压
unzip libtorch-cxx11-abi-shared-with-deps-2.5.1+cu121.zip

# 解压后得到 ~/Downloads/libtorch/ 目录，包含：
#   libtorch/
#   ├── include/           # 头文件
#   ├── lib/               # 动态库（.so）
#   └── share/cmake/       # CMake 配置文件（TorchConfig.cmake）
```

> **版本选择说明：**
> - **CXX11 ABI** 版本：如果你的项目使用 `_GLIBCXX_USE_CXX11_ABI=1`（GCC 5+ 默认），选 cxx11-abi 版本
> - **Pre-cxx11 ABI** 版本：兼容旧 ABI（`_GLIBCXX_USE_CXX11_ABI=0`）
> - **CUDA 版本**：选与你系统 CUDA Toolkit 匹配的版本；若不需要 GPU 推理，可下载 CPU-only 版本

#### 指定库路径

本项目在顶层 `CMakeLists.txt` 中通过 CMake cache 变量管理 LibTorch 路径，默认为 `$HOME/Downloads/libtorch`：

```cmake
set(LIBTORCH_ROOT "$ENV{HOME}/Downloads/libtorch" CACHE PATH "Path to LibTorch installation")
```

各子章节通过以下方式引用：

```cmake
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_ROOT})
find_package(Torch REQUIRED)
```

> **自定义路径：** 若你的 LibTorch 安装在其他位置，通过 `-D` 覆盖即可：
> ```bash
> cmake -DLIBTORCH_ROOT=/your/path/to/libtorch ..
> ```

#### 运行时 LD_LIBRARY_PATH

编译后运行可执行文件时，需要将 LibTorch 动态库路径加入链接器搜索路径：

```bash
export LD_LIBRARY_PATH=$HOME/Downloads/libtorch/lib:$LD_LIBRARY_PATH
```

可写入 `~/.bashrc` 或 `~/.zshrc` 使其永久生效。

---

### 2. ONNX Runtime（GPU 版本）

ONNX Runtime 是微软开源的高性能推理引擎，支持多框架模型（PyTorch、TensorFlow、scikit-learn 等）通过 ONNX 中间表示格式进行跨平台推理。本项目第 2、10 章的部分示例使用 ONNX Runtime 加载和运行 ONNX 模型。

#### 下载预编译包

本项目使用 `onnxruntime-linux-x64-gpu-1.25.1`：

```bash
# 进入下载目录
cd ~/Downloads

# 下载 ONNX Runtime GPU 版本
wget https://github.com/microsoft/onnxruntime/releases/download/v1.25.1/onnxruntime-linux-x64-gpu-1.25.1.tgz

# 解压
tar -xzf onnxruntime-linux-x64-gpu-1.25.1.tgz

# 解压后得到 ~/Downloads/onnxruntime-linux-x64-gpu-1.25.1/ 目录，包含：
#   onnxruntime-linux-x64-gpu-1.25.1/
#   ├── include/           # 头文件（onnxruntime_c_api.h 等）
#   └── lib/               # 动态库（libonnxruntime.so）
```

> **版本说明：**
> - `gpu` 后缀版本包含 CUDA 执行提供器和 TensorRT 支持，需要安装 CUDA Toolkit 和 cuDNN
> - 若不需要 GPU 推理，可下载 `onnxruntime-linux-x64-1.25.1.tgz`（CPU only）
> - 建议 CUDA 11.8+ 和 cuDNN 8.x+ 以匹配 ONNX Runtime 1.25.1 的运行时要求

#### CUDA / cuDNN 前置依赖

ONNX Runtime GPU 版本需要以下前置依赖：

```bash
# 检查 CUDA 是否已安装
nvcc --version

# 检查 cuDNN 是否已安装
ls /usr/local/cuda/lib64/libcudnn*
# 或
dpkg -l | grep cudnn

# 如未安装 cuDNN，通过 NVIDIA 官网下载或 apt 安装：
# wget https://developer.download.nvidia.com/compute/cudnn/...
# 或
# sudo apt install libcudnn8 libcudnn8-dev
```

#### 指定库路径

本项目在顶层 `CMakeLists.txt` 中通过 CMake cache 变量管理 ONNX Runtime 路径，默认为 `$HOME/Downloads/onnxruntime-linux-x64-gpu-1.25.1`：

```cmake
set(ONNXRUNTIME_ROOT "$ENV{HOME}/Downloads/onnxruntime-linux-x64-gpu-1.25.1" CACHE PATH "Path to ONNX Runtime installation")
```

各子章节通过 IMPORTED 库方式引用：

```cmake
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
    INTERFACE_INCLUDE_DIRECTORIES ${ONNXRUNTIME_ROOT}/include
)
```

> **自定义路径：** 若你的 ONNX Runtime 安装在其他位置，通过 `-D` 覆盖即可：
> ```bash
> cmake -DONNXRUNTIME_ROOT=/your/path/to/onnxruntime ..
> ```

#### 运行时 LD_LIBRARY_PATH

编译后运行可执行文件时，需要将 ONNX Runtime 动态库路径加入链接器搜索路径：

```bash
export LD_LIBRARY_PATH=$HOME/Downloads/onnxruntime-linux-x64-gpu-1.25.1/lib:$LD_LIBRARY_PATH
```

---

### 3. Eigen 3.4+（仅头文件的线性代数库）

```bash
# Ubuntu / Debian
sudo apt install libeigen3-dev

# 安装后头文件位于 /usr/include/eigen3/
# CMakeLists.txt 通过 find_package(Eigen3 REQUIRED) 自动查找
```

### 4. OpenCV 4.x（图像处理库，可选）

```bash
# Ubuntu / Debian
sudo apt install libopencv-dev

# 第 2 章（图像/视频预处理）和第 13 章（Grad-CAM）需要
# 若未安装，相关 target 会被自动跳过
```

### 5. Armadillo（线性代数库，可选）

```bash
# Ubuntu / Debian
sudo apt install libarmadillo-dev

# 仅第 2 章 preprocessing_with_libraries 示例需要
# 若未安装，该 target 会被自动跳过
```

---

## 环境变量汇总

建议将以下内容添加到 `~/.bashrc` 或 `~/.zshrc`：

```bash
# LibTorch
export LD_LIBRARY_PATH=$HOME/Downloads/libtorch/lib:$LD_LIBRARY_PATH

# ONNX Runtime (GPU)
export LD_LIBRARY_PATH=$HOME/Downloads/onnxruntime-linux-x64-gpu-1.25.1/lib:$LD_LIBRARY_PATH

# CUDA（如果尚未添加）
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

执行 `source ~/.bashrc` 使其立即生效。

---

## 编译方法

```bash
# 进入项目目录
cd c++/deep_learning_cpp

# 创建构建目录
mkdir -p build && cd build

# 配置（启用所有章节，使用默认路径）
cmake ..

# 或指定自定义库路径
# cmake -DLIBTORCH_ROOT=/opt/libtorch -DONNXRUNTIME_ROOT=/opt/onnxruntime ..

# 编译所有 target（使用所有可用核心）
cmake --build . -j$(nproc)

# 或仅编译特定章节（例如第 3 章）
cmake --build . --target cuda_hello -j$(nproc)
```

---

## 运行示例

```bash
cd build

# 第 1 章：CUDA 向量加法
./src/chapter-01/vector_add

# 第 1 章：LibTorch 基本操作
./src/chapter-01/lib_torch

# 第 3 章：CUDA Hello World
./src/chapter-03/cuda_hello

# 第 5 章：Eigen MLP XOR 训练
./src/chapter-05/eigen_mlp_xor

# 第 9 章：Self-Attention
./src/chapter-09/self_attention

# 第 9 章：Multi-Head Attention
./src/chapter-09/multi_head_attention

# 第 10 章：TorchScript 导出
./src/chapter-10/torchscript_export

# 第 10 章：ONNX 推理（需 ONNX Runtime）
./src/chapter-10/onnx_inference

# 第 10 章：模型服务
./src/chapter-10/model_serving

# 第 12 章：Prometheus 风格指标
./src/chapter-12/prometheus_metrics
```

---

## 章节概览

| 章节 | 内容 | 关键依赖 |
|------|------|----------|
| 第 1 章 | C++ 深度学习入门（环境搭建、CUDA 向量加法、LibTorch 初探） | LibTorch, ONNX Runtime, CUDA |
| 第 2 章 | 数据预处理（缺失值/编码/标准化/降维/文本/图像/音频） | LibTorch, Eigen, OpenCV(可选), Armadillo(可选) |
| 第 3 章 | CUDA GPU 加速（线程模型、错误处理、性能分析） | CUDA Toolkit |
| 第 4 章 | 基础神经网络（线性/逻辑回归、MLP with Eigen、反向传播） | Eigen |
| 第 5 章 | 多层感知机（MLP with Eigen & LibTorch、激活函数、优化器） | LibTorch, Eigen |
| 第 6 章 | 卷积神经网络（CPU 卷积、Eigen 卷积、LibTorch CNN、VGG16） | LibTorch, Eigen |
| 第 7 章 | RNN/LSTM（Eigen RNN、BPTT、LSTM with LibTorch、Word2Vec） | LibTorch, Eigen |
| 第 8 章 | 生成模型（Autoencoder、VAE、GAN、采样策略） | LibTorch |
| 第 9 章 | Transformer（Self/Multi-Head Attention、位置编码、Encoder/Decoder、量化剪枝蒸馏） | LibTorch |
| 第 10 章 | 模型部署（TorchScript 导出、ONNX 推理、微批次调度、模型服务） | LibTorch, ONNX Runtime |
| 第 11 章 | 部署调试与重训练（漂移检测、可观测性、微批次、安全发布） | 纯 STL |
| 第 12 章 | 模型监控（百分位、直方图、ECE、Prometheus 指标、追踪、告警） | 纯 STL |
| 第 13 章 | 可解释性（LIME、KernelSHAP、Grad-CAM、模型卡片） | Eigen, LibTorch(可选), OpenCV(可选) |
| 第 14 章 | 附录与推荐阅读 | — |

---

## 验证安装

```bash
# 验证 LibTorch（根据你的实际路径调整）
find $HOME/Downloads/libtorch/lib -name "libtorch.so" 2>/dev/null && echo "LibTorch OK" || echo "LibTorch NOT FOUND"

# 验证 ONNX Runtime（根据你的实际路径调整）
find $HOME/Downloads/onnxruntime-linux-x64-gpu-1.25.1/lib -name "libonnxruntime.so*" 2>/dev/null && echo "ONNX Runtime OK" || echo "ONNX Runtime NOT FOUND"

# 验证 CUDA
nvcc --version 2>/dev/null && echo "CUDA OK" || echo "CUDA NOT FOUND"

# 验证 Eigen
find /usr/include -name "Dense" -path "*/Eigen/*" 2>/dev/null && echo "Eigen OK" || echo "Eigen NOT FOUND"
```

---

## 参考资料

| 资源 | 链接 |
|------|------|
| **原书** | Deep Learning with C++, Packt Publishing, ISBN 9781835880036 |
| **LibTorch 文档** | https://pytorch.org/cppdocs/ |
| **LibTorch 下载** | https://pytorch.org/get-started/locally/ |
| **ONNX Runtime 发布** | https://github.com/microsoft/onnxruntime/releases |
| **Eigen 文档** | https://eigen.tuxfamily.org/ |
| **CUDA 下载** | https://developer.nvidia.com/cuda-downloads |
