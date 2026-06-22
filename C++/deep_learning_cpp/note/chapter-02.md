# 第 2 章：C++ 数据准备与预处理

基于 *Deep Learning with C++*（Packt，ISBN 9781835880036）第 2 章，第 29–76 页。

---

## 目录

1. [章节概述](#章节概述)
2. [文件索引](#文件索引)
3. [编译与运行](#编译与运行)
4. [技术速查](#技术速查)
5. [PDF 完整内容对照](#pdf-完整内容对照)
6. [注意事项](#注意事项)

---

## 章节概述

数据准备是高效深度学习工作流的基础。原始数据很少能直接用于模型——往往不完整、含噪声、不一致，且形态多样（数值、文本、图像、时序、音频），预处理将它们转化为结构化、归一化的表示。

### 预处理六大价值

| 价值 | 说明 |
|------|------|
| 加速收敛 | 归一化输入减少训练早期的梯度震荡 |
| 降噪 | 去除无关特征和异常值 |
| 提升效率 | 紧凑表示降低内存和计算开销 |
| 增强泛化 | 数据增强扩大多样性 |
| 多模态支持 | 对文本、表格、图像、信号统一变换 |
| 领域适配 | 针对医疗、金融、CV 等场景定制流水线 |

### 五大挑战

| 挑战 | 说明 |
|------|------|
| 类别不平衡 | 少数类样本不足 |
| 跨特征归一化 | 异构数据尺度差异 |
| 内存/算力限制 | 大数据集无法全量加载 |
| 分布式扩展 | 多机多卡数据分发 |
| 实时性要求 | 在线推理需低延迟 |

---

## 文件索引

### 一、结构化数据（表格/时序）— PDF 第 33–50 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `01_handling_missing_values.cpp` | 33–37 | 均值替换、前向填充、反向填充、k-NN 插补、回归插补 | STL |
| `02_Encoding_categorical_features.cpp` | 37–41 | One-hot、频率、序数、二进制、嵌入编码 | STL |
| `03_feature_scaling_and_standardization.cpp` | 41–43 | Min-Max、Z-score、Robust Scaling、Log 变换、Power 变换 | STL |
| `04_Dimensionality_reduction.cpp` | 43–44, 55 | PCA（Eigen）+ 自编码器（LibTorch）+ t-SNE 概述 | Eigen3, LibTorch |
| `05_time_series_engineering.cpp` | 44–47 | 滑动窗口、指数平滑、差分、DFT、时间 sin/cos 特征 | STL |
| `06_feature_interation_engineering.cpp` | 47–50 | 多项式特征展开、成对交互项 | STL |

### 二、库驱动预处理 — PDF 第 51–58 页

第 51–54 页介绍了如何用现有 C++ 库重写相同技术，获得更简洁的代码、更好的数值精度和生产级性能：

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `07_preprocessing_with_libraries.cpp` | 51–56 | 用 Armadillo 重写：滑动窗口（`arma::conv`）、指数平滑、Z-score、Min-Max、PCA（`arma::princomp` SVD 方法） | Armadillo |

| 库 | 用途 | 实现位置 |
|----|------|---------|
| **Eigen 3.4+** | 线性代数、PCA、嵌入变换 | `04_Dimensionality_reduction.cpp`（PCA 部分） |
| **Armadillo** | 矩阵运算、时序平滑（`arma::conv` 卷积）、SVD-PCA | `07_preprocessing_with_libraries.cpp` |
| **mlpack** | 可扩展 ML 预处理（PCA、t-SNE、KNN） | `04_Dimensionality_reduction.cpp`（t-SNE 引用） |
| **LibTorch** | 张量操作、Dataset API、自编码器 | `08_libtorch.cpp`, `17_custom_dataset_demo.cpp` |
| **GPU 后端（可选）** | cuBLAS/cuFFT/cuML 加速 | 见第 51 页说明（硬件相关，未独立实现） |

### 三、非结构化数据 — 文本 — PDF 第 59–66 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `09_tokenization_stop_word_removal_stemming.cpp` | 59 | 分词、去停用词（`unordered_set`）、Porter 词干提取。PDF 中用 `libstemmer`（Snowball）做词干提取，我们提供了纯 C++ Porter 实现 + libstemmer 说明 | STL |
| `10_embedding_based_vectorization.cpp` | 60–62 | TF-IDF 向量化、玩具 Word2Vec 嵌入、句子嵌入（平均池化）。PDF 中还介绍了 GloVe + Eigen 加载预训练嵌入 | STL |
| `11_contextual_embedding.cpp` | 63 | BERT [CLS] 嵌入通过 ONNX Runtime（支持 mock/真实两种模式） | ONNX Runtime（可选） |
| `12_sequence_padding_and_truncation.cpp` | 64–66 | 固定长度填充/截断、注意力掩码生成、批次对齐 | STL |

### 四、非结构化数据 — 图像 / 音频 / 视频 — PDF 第 67–71 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `13_image_resol.cpp` | 67–69 | Resize→Center-Crop 224×224→直方图均衡化（Y 通道）→Canny 边缘检测。PDF 还提及翻转、旋转、透视变换、颜色归一化 | OpenCV 4.x |
| `14_video_frame_extraction.cpp` | 67 | 视频帧采样（均匀采样、关键帧提取、场景变化检测），合成测试视频生成 | OpenCV 4.x |
| `15_audio_video.cpp` | 67, 70 | STFT 语谱图（Hann 窗→DFT→dB 尺度）、峰值频率检测。PDF 还提及重采样 | STL（可移植 DFT） |

### 五、数据增强 — PDF 第 70–71 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `16_advanced_tech.cpp` | 70–71 | 高斯噪声增强（数值特征）、特征 Dropout、图像高斯噪声（OpenCV）、Token Dropout、分层采样、数据校验 | STL（图像增强需 OpenCV） |

### 六、LibTorch 集成 — PDF 第 55, 71–73 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `08_libtorch.cpp` | 55 | 自编码器训练（Adam+MSE）、压缩表示提取 | LibTorch |
| `17_custom_dataset_demo.cpp` | 71–73 | 自定义 Dataset、DataLoader 批处理/打乱/多线程 | LibTorch |

### 七、大数据集管理 — PDF 第 73–74 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `18_memory_mapping.cpp` | 73–74 | `mmap()` 零拷贝 I/O、惰性加载、多进程共享、`madvise` 优化 | Linux `mmap` |
| `19_large_scale_data.cpp` | 73–74 | 分片（Sharding）、流式加载（Streaming）、预处理缓存（Caching） | STL |

---

## 编译与运行

### 环境要求

```bash
# 必需
C++17 编译器（GCC 11+ / Clang 14+）
CMake 3.22+
LibTorch → $HOME/Downloads/libtorch
Eigen 3.4+ → apt install libeigen3-dev

# 可选（部分示例需要）
OpenCV 4.x          → apt install libopencv-dev
Armadillo 12+       → apt install libarmadillo-dev
ONNX Runtime        → 预编译包于 $HOME/Downloads/onnxruntime-linux-x64-gpu-1.25.1
libstemmer (Snowball) → apt install libstemmer-dev   # 生产环境词干提取
mlpack               → apt install libmlpack-dev      # t-SNE、KNN 等
SentencePiece        → 编译安装                        # 子词分词
```

### 编译

```bash
cd c++/deep_learning_cpp/build
cmake ..
cmake --build . --target <target_name> -j$(nproc)
```

### 运行示例

```bash
# 结构化数据
./build/chapter02/handling_missing_values
./build/chapter02/Encoding_categorical_features
./build/chapter02/feature_scaling
./build/chapter02/time_series_engineering
./build/chapter02/feature_interaction

# 文本
./build/chapter02/tokenization
./build/chapter02/embedding_vectorization
./build/chapter02/sequence_padding

# 图像/音频/视频
./build/chapter02/image_resol [input_image.jpg]   # 需 OpenCV
./build/chapter02/video_frame_extraction            # 需 OpenCV
./build/chapter02/audio_video

# 库驱动预处理
./build/chapter02/preprocessing_with_libraries      # 需 Armadillo

# LibTorch
./build/chapter02/libtorch_ae
./build/chapter02/custom_dataset
./build/chapter02/Dimensionality_reduction

# 高级
./build/chapter02/contextual_embedding
./build/chapter02/advanced_tech
./build/chapter02/memory_mapping
./build/chapter02/large_scale_data
```

---

## 技术速查

### 缺失值处理

| 方法 | 适用场景 | 注意事项 |
|------|----------|----------|
| 均值替换 | 快速基线、稀疏缺失 | 扭曲分布、压缩方差，非 MCAR 时有偏 |
| 前向填充 | 状态型信号（设备模式、传感器） | 长间隙时传播陈旧值，压平真实动态 |
| 反向填充 | 简单补缺 | 训练时可能泄露未来信息 |
| k-NN 插补 | 捕捉局部结构 | 需特征缩放，高维退化，对 k 敏感 |
| 回归插补 | 特征间存在线性相关 | 用交叉拟合避免泄露；可用 Ridge/Lasso/GBM |

### 类别特征编码

| 方法 | 适用场景 | 注意事项 |
|------|----------|----------|
| One-hot | 低基数、线性/NN 模型 | 高基数维度爆炸；线性模型应去掉一列消共线 |
| 频率编码 | 保留流行度、单数值特征 | 按折统计；对重尾类别考虑 log-频率或截断 |
| 序数编码 | 存在真实顺序（如尺寸 S<M<L） | 无真实顺序时注入虚假序关系 |
| 二进制编码 | 高基数、内存受限 | 引入伪序数邻近，不适用于距离模型 |
| 嵌入编码 | 学习稠密向量、语义相似 | 需训练；设置 `<UNK>` 令牌处理未见类别 |

### 特征缩放

| 方法 | 适用场景 | 注意事项 |
|------|----------|----------|
| Min-Max [0,1] | 有界输入（图像像素）、距离模型 | 对异常值敏感 |
| Z-score（μ=0,σ=1） | 近似高斯、线性模型、NN | 异常值会膨胀标准差 |
| Robust（中位数/IQR） | 重尾分布、含异常值 | 注意 IQR≈0 时需回退 |
| Log 变换 | 右偏数据、计数数据 | 用 `log1p(x)` 处理零值；负值用 Yeo-Johnson |
| Power（Box-Cox） | 异方差 | 要求正值输入 |

### 降维

| 方法 | 类型 | 适用场景 |
|------|------|----------|
| PCA | 线性 | 去冗余特征、加速训练、快速可解释；可用 SVD 或 EVD |
| 自编码器 | 非线性 | 复杂流形、异常检测、预训练 |
| t-SNE | 非线性 | 仅可视化（2D/3D），保留局部邻域 |

### 时序特征工程

| 方法 | 用途 |
|------|------|
| 滑动窗口 | 平滑噪声、揭示趋势 |
| 指数平滑 | 几何衰减加权，α 控制平滑度。扩展：Holt（趋势）、Holt-Winters（趋势+季节性） |
| 差分 | 去趋势/季节性、使序列平稳；过度差分会放大噪声 |
| DFT / FFT | 检测周期模式 |
| Sin/Cos 时间编码 | 保持循环结构（23:00 ≈ 01:00） |

### 数据增强

| 类型 | 方法 | 适用场景 |
|------|------|----------|
| 图像 — 几何 | 翻转、旋转、裁剪、缩放、透视变换 | CV 分类/检测/分割 |
| 图像 — 像素 | 高斯噪声、亮度/对比度调整、颜色抖动 | 提高鲁棒性 |
| 文本 | Token Dropout（随机丢弃词汇） | NLP 分类、检索 |
| 数值 | 高斯噪声、插值合成样本、Bootstrap | 表格数据 |
| 音频 | 时间拉伸、音高偏移、背景噪声混合 | 语音识别、音频分类 |

### 大数据集管理

| 技术 | 说明 |
|------|------|
| `mmap` | 零拷贝 I/O，惰性加载，多进程共享 |
| 分片（Sharding） | 将数据拆分为多个文件，支持并行读取 |
| 缓存 | 缓存预处理中间结果，避免重复计算 |
| 流式加载 | 边训练边加载，不占满内存 |
| 数据校验 | 检查特征维度、NaN/Inf、数值范围 |

---

## PDF 完整内容对照

以下是 PDF 第 29–76 页的完整纲要，标注了各节对应的实现文件：

| PDF 页 | 内容 | 实现文件 |
|--------|------|---------|
| 29–30 | 章节概述、技术要求（编译器/CUDA/库） | `note.md` |
| 31–32 | 预处理的必要性（病态损失曲面） | 所有文件注释 |
| 33 | 结构化 vs 非结构化数据 | — |
| 33–34 | 均值替换 | `01_handling_missing_values.cpp` |
| 34–35 | 前向填充 | `01_handling_missing_values.cpp` |
| 35–36 | 反向填充 | `01_handling_missing_values.cpp` |
| 36 | k-NN 插补 | `01_handling_missing_values.cpp` |
| 37 | 回归插补 | `01_handling_missing_values.cpp` |
| 37–38 | One-hot 编码 | `02_Encoding_categorical_features.cpp` |
| 38–39 | 频率编码 | `02_Encoding_categorical_features.cpp` |
| 39 | 序数编码 | `02_Encoding_categorical_features.cpp` |
| 39–40 | 二进制编码 | `02_Encoding_categorical_features.cpp` |
| 40–41 | 嵌入编码 | `02_Encoding_categorical_features.cpp` |
| 41–42 | Min-Max 缩放 | `03_feature_scaling_and_standardization.cpp` |
| 42 | Z-score 归一化 | `03_feature_scaling_and_standardization.cpp` |
| 43 | Robust Scaling | `03_feature_scaling_and_standardization.cpp` |
| 43–44 | Log 变换、Power 变换 | `03_feature_scaling_and_standardization.cpp` |
| 44 | PCA、t-SNE、自编码器概述 | `04_Dimensionality_reduction.cpp` |
| 44–45 | 滑动窗口聚合 | `05_time_series_engineering.cpp` |
| 45–47 | 指数平滑、差分、傅里叶变换 | `05_time_series_engineering.cpp` |
| 47 | 时间特征提取（sin/cos） | `05_time_series_engineering.cpp` |
| 47–50 | 特征交互工程（多项式、交互项） | `06_feature_interation_engineering.cpp` |
| 51 | **库驱动预处理**（Eigen/Armadillo/mlpack/GPU） | 见"库驱动预处理"节 |
| 52–54 | **mlpack/oneDAL PCA 示例** | `07_preprocessing_with_libraries.cpp`（Armadillo SVD-PCA） |
| 55 | **自编码器实现（LibTorch）** | `08_libtorch.cpp` |
| 55–56 | **时序预处理用 Armadillo + OpenCV** | `07_preprocessing_with_libraries.cpp`（`arma::conv` 滑动窗口等） |
| 57–58 | 从结构化数据过渡到非结构化数据 | — |
| 59 | **分词 + 去停用词（libstemmer/Snowball）** | `09_tokenization_stop_word_removal_stemming.cpp` |
| 60–62 | **基于嵌入的向量化（GloVe + Eigen）** | `10_embedding_based_vectorization.cpp`（含 GloVe 引用） |
| 63 | **上下文嵌入（BERT + ONNX Runtime）** | `11_contextual_embedding.cpp` |
| 64–66 | **序列填充与截断** | `12_sequence_padding_and_truncation.cpp` |
| 67–69 | **图像预处理**（Resize/Crop/Equalize/Canny） | `13_image_resol.cpp` |
| 67 | **视频帧采样** | `14_video_frame_extraction.cpp` |
| 67, 70 | **音频预处理**（STFT 语谱图）、视频帧采样 | `15_audio_video.cpp` |
| 70–71 | **数据增强**（图像高斯噪声、Token Dropout、数值增强） | `16_advanced_tech.cpp` |
| 71–73 | **PyTorch Dataset API 集成** | `17_custom_dataset_demo.cpp` |
| 73–74 | **内存映射（mmap）**、大数据集管理（分片/流式/缓存） | `18_memory_mapping.cpp`, `19_large_scale_data.cpp` |
| 74 | 数据版本管理与流水线可复现性 | `16_advanced_tech.cpp`（数据校验部分） |
| 75 | 章节问题 | — |
| 75–76 | 拓展阅读、参考答案 | — |

---

## 注意事项

### 外部库依赖

| 文件 | 需要的外部库 | 未安装时的行为 |
|------|-------------|---------------|
| `13_image_resol.cpp` | OpenCV 4.x | CMake 自动跳过 |
| `11_contextual_embedding.cpp` | ONNX Runtime | `#if USE_REAL_MODEL` 默认关闭，mock 模式可运行 |
| `04_Dimensionality_reduction.cpp` | Eigen3 + LibTorch | 必需，否则链接失败 |
| `08_libtorch.cpp` | LibTorch | 必需 |
| `17_custom_dataset_demo.cpp` | LibTorch | 必需 |
| `15_audio_video.cpp` | 无（可移植 DFT） | 始终可编译。生产环境建议 FFTW3 |

### PDF 中提及但未独立实现的库用法

以下知识点在 PDF 中有详细示例，但在本章代码中以引用说明的形式出现，因为需要额外安装对应库：

| 知识点 | PDF 页 | 说明 |
|--------|--------|------|
| mlpack PCA / t-SNE | 52–54 | `04_Dimensionality_reduction.cpp` 的 t-SNE 部分引用 mlpack API |
| Armadillo 时序处理 | 55–56 | `arma::conv()` 滑动窗口、矩阵指数平滑 |
| libstemmer (Snowball) 词干提取 | 59 | `09_tokenization_stop_word_removal_stemming.cpp` 提供了纯 C++ Porter 实现并注明了 libstemmer 用法 |
| GloVe + Eigen 嵌入加载 | 60–62 | `10_embedding_based_vectorization.cpp` 提供了 TF-IDF + 玩具嵌入，并注明 GloVe 预训练向量的加载方式 |
| OpenCV 图像高斯噪声增强 | 71 | `16_advanced_tech.cpp` 提供了数值噪声增强，并注明 OpenCV 图像增强用法 |
| Video 帧采样 | 67 | 未独立实现（需要 FFmpeg/OpenCV VideoCapture） |

### 其他注意事项

- `11_contextual_embedding.cpp` 设为 `USE_REAL_MODEL=0` 时以 mock 模式运行，输出使用说明。设为 1 后需要 BERT ONNX 模型（约 400 MB）和 WordPiece 分词器。
- `15_audio_video.cpp` 使用 O(N²) DFT，适合演示；生产环境建议安装 FFTW3 (`apt install libfftw3-dev`)。
- Eigen 3.4+ 位于 `/usr/include/eigen3/`（通过 `libeigen3-dev` 安装），`CMakeLists.txt` 使用 `find_package(Eigen3)` 自动查找。
- 所有 LibTorch 示例使用 C++17，LibTorch 路径为 `$HOME/Downloads/libtorch`。
- `18_memory_mapping.cpp` 使用 POSIX `mmap`，仅在 Linux/macOS 可用。Windows 需用 `CreateFileMapping` 等效 API。
