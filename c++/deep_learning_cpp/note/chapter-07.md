# 第 7 章：循环神经网络（RNN）与 LSTM 网络

基于 *Deep Learning with C++*（Packt，ISBN 9781835880036）第 7 章，第 210–264 页。本章处理深度学习中最核心的序列建模问题——从 RNN 的隐藏状态递归到 LSTM 的三门控机制，再到 BPTT 训练算法、文本预处理流水线和 Word2Vec 嵌入，最终实战文本预测和 Seq2Seq 机器翻译。

---

## 目录

1. [章节概述](#章节概述)
2. [核心概念：从 RNN 到 LSTM](#核心概念从-rnn-到-lstm)
3. [文件索引](#文件索引)
4. [代码演进：RNN → LSTM → 应用](#代码演进rnn--lstm--应用)
5. [编译与运行](#编译与运行)
6. [技术速查](#技术速查)
7. [PDF 完整内容对照](#pdf-完整内容对照)
8. [注意事项](#注意事项)

---

## 章节概述

序列数据——文本、语音、时间序列——的核心挑战在于：每个元素的意义依赖于其时间上下文。"bank"是银行还是河岸，取决于前后文。传统的全连接网络独立处理每个输入，丢弃了顺序信息。RNN 通过**隐藏状态（hidden state）递归**解决了这个问题：网络的输出成为下一时间步的输入之一，形成反馈回路。

但标准 RNN 有致命缺陷——**梯度消失/爆炸**。反向传播通过时间（BPTT）时，梯度经反复乘以权重矩阵和激活函数导数后呈指数衰减，导致早期时间步几乎收不到学习信号。LSTM 用**三门控机制 + 细胞状态**创建了一条"梯度高速公路"，使梯度能无损地流回数百甚至数千步之前。

### 核心学习目标

| 目标                   | 说明                                                                   |
| ---------------------- | ---------------------------------------------------------------------- |
| RNN 数学模型           | `h_t = tanh(W_xh·x_t + W_hh·h_{t-1} + b_h)`，递归隐藏状态              |
| BPTT 训练算法          | 沿时间步展开 RNN、逐步反向传播梯度、梯度裁剪/截断                       |
| LSTM 三门控机制        | Forget Gate（遗忘）/ Input Gate（输入）/ Output Gate（输出）+ Cell State |
| 三种实现递进           | Vector-based（理解）→ Eigen（矩阵优化）→ LibTorch（生产部署）           |
| 文本预处理             | 小写化、去标点、分词（词/字符/子词）、BPE、停用词                      |
| Word2Vec 嵌入          | Skip-gram 模型，CBOW，LibTorch 实现，语义向量空间                       |
| Seq2Seq 翻译           | 编码器-解码器架构 + Teacher Forcing + 自回归推理                        |

---

## 核心概念：从 RNN 到 LSTM

### RNN：递归隐藏状态

```
h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b_h)   ← 隐藏状态（网络记忆）
y_t = softmax(W_hy · h_t + b_y)                   ← 当前时刻输出
```

- 同一组权重（`W_xh`, `W_hh`, `W_hy`）在**所有时间步重复使用**——参数共享
- 4 种架构模式：One-to-One、One-to-Many（图像→描述）、Many-to-One（情感分类）、Many-to-Many（NER/翻译）

### BPTT：展开时间进行反向传播

RNN 沿时间轴展开成 T 层深的计算图。梯度必须流经所有 T 层：

```
∂L      ∂L   ∂L ∂h_{t+1}
---  =  --- + --- -------
∂h_t    ∂h_t  ∂h_{t+1} ∂h_t
```

**崩溃根源**：每次反向经过一个时间步，梯度乘以 `W_hh^T · tanh'(z)`——这个雅可比矩阵的特征值通常 < 1，连乘 T 次后指数趋零。

**两个应对策略**：
1. **梯度裁剪**：计算总梯度的 L2 范数，若 > threshold(5.0) 则等比例缩至 threshold
2. **截断 BPTT (Truncated BPTT)**：仅回传最近 K=10~50 步，超长序列时分段训练

### LSTM：三门控 + 细胞状态

LSTM 的核心创新是**细胞状态 `C_t`——一条独立于隐藏状态的"信息高速公路"**：

```
Forget Gate:  f_t = σ(W_f·x_t + U_f·h_{t-1} + b_f)     ← 决定遗忘什么 (0=全忘, 1=全留)
Input Gate:   i_t = σ(W_i·x_t + U_i·h_{t-1} + b_i)     ← 决定输入什么
Candidate:    C̃_t = tanh(W_c·x_t + U_c·h_{t-1} + b_c)  ← 新候选信息
Cell Update:  C_t = f_t⊙C_{t-1} + i_t⊙C̃_t               ← 选择性遗忘+选择性输入
Output Gate:  o_t = σ(W_o·x_t + U_o·h_{t-1} + b_o)     ← 决定暴露什么
Hidden:       h_t = o_t⊙tanh(C_t)                       ← 滤波后的输出
```

**为什么 LSTM 解决了梯度消失**：当 `f_t ≈ 1`（遗忘门打开），`C_{t-1}` 几乎无损地流向 `C_t`。反向传播时 `∂C_t/∂C_{t-1} = f_t ≈ 1`，梯度无需经历连乘衰减。

**关键初始化技巧**：遗忘门偏置 `b_f` 初始化为 **1.0**，让网络在训练初期"倾向于记住一切"，然后逐渐学会选择性遗忘。

---

## 文件索引

### 一、RNN + BPTT — PDF 第 214–221 页

| 文件                  | PDF 页    | 涵盖知识点                                           | 依赖     |
| --------------------- | --------- | ---------------------------------------------------- | -------- |
| `00_rnn_eigen.cpp`    | 214–216   | `RNNCell` 类、Xavier 初始化、tanh/sigmoid 激活、序列处理 | Eigen    |
| `01_bptt_demo.cpp`    | 216–221   | BPTT 反向传播、梯度裁剪、截断 BPTT、tanh 导数        | Eigen    |

### 二、LSTM 实现 — PDF 第 221–233 页

| 文件                     | PDF 页    | 涵盖知识点                                              | 依赖     |
| ------------------------ | --------- | ------------------------------------------------------- | -------- |
| `02_lstm_eigen.cpp`      | 227–230   | `EigenLSTMCell`（四门合一矩阵）、遗忘门偏置=1.0、LSTMNetwork | Eigen    |
| `03_lstm_libtorch.cpp`   | 231–233   | `ProductionLSTM`、多层堆叠、Dropout、模型序列化           | LibTorch |

### 三、文本处理 — PDF 第 237–247 页

| 文件                      | PDF 页    | 涵盖知识点                                               | 依赖 |
| ------------------------- | --------- | -------------------------------------------------------- | ---- |
| `04_text_processing.cpp`  | 237–247   | 小写化、去标点、句子分词、词/字符/子词级分词对比、BPE、停用词 | STL  |

### 四、Word2Vec — PDF 第 248–253 页

| 文件                        | PDF 页    | 涵盖知识点                                               | 依赖     |
| --------------------------- | --------- | -------------------------------------------------------- | -------- |
| `05_word2vec_libtorch.cpp`  | 248–253   | Skip-gram、词汇表构建、训练对生成、SGD 训练、嵌入语义空间 | LibTorch |

---

## 代码演进：RNN → LSTM → 应用

### 第 1 步：Eigen RNNCell

```cpp
class RNNCell {
    Eigen::MatrixXf W_xh, W_hh, W_hy;  // 三组权重矩阵
    Eigen::VectorXf b_h, b_y;

    // Xavier 初始化：稳定梯度流
    W_xh = MatrixXf::Random(hidden, input) * sqrt(2.0f/(input+hidden));

    std::pair<VectorXf, VectorXf> forward(const VectorXf& x_t, const VectorXf& h_prev) {
        VectorXf h_t = tanh(W_xh * x_t + W_hh * h_prev + b_h);  // 隐藏状态更新
        VectorXf y_t = sigmoid(W_hy * h_t + b_y);               // 输出生成
        return {h_t, y_t};  // 返回新隐藏状态 + 输出，h_t 传到下一时刻
    }
};
```

### 第 2 步：BPTT + 梯度裁剪

```cpp
// 沿时间反向遍历：t 从 T-1 到 0
for (int t = seq_len-1; t >= 0; --t) {
    output_error = hidden_states[t] - targets[t] + dh_future;
    pre_activation_grad = output_error * tanh_derivative(pre_activations[t]);
    // 累积外积梯度
    dW_xh += pre_activation_grad * inputs[t].transpose();
    dh_next = W_hh.transpose() * pre_activation_grad;  // 传给 t-1
}

// 梯度裁剪：防止梯度爆炸
void clip_gradients(float threshold=5.0) {
    float norm = sqrt(Σ g²);  // L2 范数
    if (norm > threshold)
        for (auto& g : grads) g *= threshold / norm;  // 等比例缩放
}
```

### 第 3 步：EigenLSTMCell（四门合一）

```cpp
class EigenLSTMCell {
    Eigen::MatrixXf W_combined;  // [4*hidden, input]  四门权重合一
    Eigen::MatrixXf U_combined;  // [4*hidden, hidden]
    Eigen::VectorXf b_combined;  // [4*hidden]
    // b_combined.segment(0, hidden).setOnes();  ← 遗忘门偏置=1.0

    auto forward(x_t, h_prev, c_prev) {
        // 一次矩阵乘法算出所有四个门！
        VectorXf gates = W_combined*x_t + U_combined*h_prev + b_combined;
        auto f = sigmoid(gates.segment(0, h));           // 遗忘门
        auto i = sigmoid(gates.segment(h, h));           // 输入门
        auto c = tanh(gates.segment(2*h, h));            // 候选
        auto o = sigmoid(gates.segment(3*h, h));         // 输出门

        VectorXf C_t = f.cwiseProduct(c_prev) + i.cwiseProduct(c);  // 细胞更新
        VectorXf h_t = o.cwiseProduct(tanh(C_t));                   // 隐藏输出
        return {h_t, C_t};  // 双状态返回
    }
};
```

### 第 4 步：LibTorch Production LSTM

```cpp
class ProductionLSTM : torch::nn::Module {
    torch::nn::LSTM lstm{nullptr};   // 内置 CUDA 优化
    torch::nn::Linear proj{nullptr}; // 输出投影
    torch::nn::Dropout dropout{nullptr};  // 防过拟合

    ProductionLSTM(in_sz, hidden_sz):
        lstm(LSTMOptions(in_sz, hidden_sz)
                .num_layers(2)       // 堆叠 2 层
                .dropout(0.3)        // 层间 Dropout
                .batch_first(true)), // 现代张量布局
        proj(Linear(hidden_sz, 1)) {
        register_module("lstm", lstm);
        register_module("dropout", dropout);
    }

    // AutoGrad 计算 BPTT——无需手写反向传播！
    auto forward(Tensor x) {
        auto [out, hidden] = lstm->forward(x);  // LSTM 前向
        out = dropout->forward(out);             // 正则化
        return proj->forward(out);              // 投影到目标维度
    }
};
```

### 第 5 步：文本处理流水线

```
原始文本 → 小写化 → 去标点 → 分词 → 词汇表映射 → 嵌入向量 → LSTM
```

3 种分词策略对比（以同一句为例）：

| 策略     | 分词结果                             | Token 数 | 特点                     |
| -------- | ------------------------------------ | -------- | ------------------------ |
| 词级     | [Machine, learning, ...]             | 8        | 保留语义，OOV 问题       |
| 字符级   | [M,a,c,h,i,n,e, ,...]                | 68       | 无 OOV，序列极长         |
| 子词(BPE)| [Mach,ine,learn,ing,...]            | 15       | 平衡语义和 OOV，生产常用 |

### 第 6 步：Skip-gram Word2Vec

```
P(context | center) = exp(v_c·v_w) / Σ exp(v_c·v_i)

训练后: vector("king") - vector("man") + vector("woman") ≈ vector("queen")
```

---

## 编译与运行

```bash
# 环境
C++17 + CMake 3.22+
Eigen 3.4+ (00-02) + LibTorch (03, 05)

# 编译
cd build && cmake .. && cmake --build . --target rnn_eigen -j$(nproc)
```

```bash
./build/chapter07/rnn_eigen          # RNNCell 隐藏状态演示
./build/chapter07/bptt_demo          # BPTT + 梯度裁剪
./build/chapter07/lstm_eigen         # 四门合一 LSTM 细胞
./build/chapter07/lstm_libtorch      # 生产级 LSTM + 模型序列化
./build/chapter07/text_processing     # 文本预处理 7 步流水线
./build/chapter07/word2vec           # Skip-gram 词嵌入训练
```

---

## 技术速查

### LSTM 四门速查

| 门        | 公式                      | 激活   | 作用                         | 关键初始化     |
| --------- | ------------------------- | ------ | ---------------------------- | -------------- |
| Forget    | σ(W_f·x + U_f·h + b_f)   | sigmoid | 0=遗忘旧信息, 1=保留         | **b_f = 1.0**  |
| Input     | σ(W_i·x + U_i·h + b_i)   | sigmoid | 0=忽略新信息, 1=完全吸收     | b_i ≈ 0        |
| Candidate | tanh(W_c·x + U_c·h + b_c) | tanh    | 生成新候选值 [-1, 1]         | 随机初始化     |
| Output    | σ(W_o·x + U_o·h + b_o)   | sigmoid | 0=隐藏细胞, 1=全部暴露       | 随机初始化     |

### 训练策略速查

| 技术               | 作用                            | 典型参数              |
| ------------------ | ------------------------------- | --------------------- |
| BPTT               | RNN/LSTM 训练算法               | 全序列或截断窗口      |
| 梯度裁剪           | 防止梯度爆炸                    | threshold = 5.0       |
| Teacher Forcing    | 用真实目标而非预测作为下一输入  | 100% 训练, 0% 推理    |
| 混合精度训练       | FP16 前向 + FP32 梯度累加        | autocast + scaler     |
| 自回归解码         | 逐 token 生成，<EOS> 终止       | argmax / beam search  |

### BPE 算法要点

1. 从字符级词汇表开始
2. 统计所有相邻 token 对的出现频率
3. 合并频率最高的那对为一个新 token
4. 重复 2-3 直到达到目标词汇量（8K~50K）

### Sigmoid/Tanh 导数

| 函数    | 导数                                | 简洁形式                |
| ------- | ----------------------------------- | ----------------------- |
| sigmoid | σ(x)·(1-σ(x))                        | 用输出值直接计算        |
| tanh    | 1 - tanh²(x)                        | 用输出值直接计算        |

---

## PDF 完整内容对照

| 书本页   | 内容                                                          | 实现文件                     |
| -------- | ------------------------------------------------------------- | ---------------------------- |
| 210–211  | 序列数据特性（因果、上下文、模式涌现、记忆需求）              | --                           |
| 211–213  | RNN 架构（4 种类型）、数学基础（`h_t = tanh(W·x + U·h + b)`） | `00_rnn_eigen.cpp`           |
| 214–216  | Eigen RNNCell 实现 (W_xh/W_hh/W_hy, Xavier init)              | `00_rnn_eigen.cpp`           |
| 216–220  | BPTT 数学 + 向量/Eigen/LibTorch 实现 + 梯度裁剪               | `01_bptt_demo.cpp`           |
| 221–227  | LSTM 三门控 + 细胞状态 + 数学公式 + Vec 实现                  | `02_lstm_eigen.cpp`          |
| 227–230  | EigenLSTMCell（四门合一矩阵，遗忘门偏置=1.0）                 | `02_lstm_eigen.cpp`          |
| 231–233  | LibTorch ProductionLSTM + 混合精度 + 序列化                   | `03_lstm_libtorch.cpp`       |
| 234–237  | LSTM 反向传播数学（四门梯度、细胞状态梯度）                   | --                           |
| 237–242  | C++23 字符串增强、文件 I/O、小写化、去标点                    | `04_text_processing.cpp`     |
| 242–247  | 分词（词/字符/子词）、BPE 算法、停用词处理                    | `04_text_processing.cpp`     |
| 248–250  | Word2Vec 原理、Skip-gram 数学公式、CBOW                        | `05_word2vec_libtorch.cpp`   |
| 250–253  | LibTorch WordEmbeddings 实现（词汇表/训练对/SGD）              | `05_word2vec_libtorch.cpp`   |
| 253–255  | 文本处理流水线总结、LSTM 集成                                 | --                           |
| 255–259  | **文本预测应用**：LSTM + Linear + CrossEntropy                 | （参考本书 GitHub）          |
| 259–263  | **Seq2Seq 翻译**：编码器-解码器 + Teacher Forcing + 自回归     | （参考本书 GitHub）          |
| 263–264  | 章节小结 + 拓展阅读                                           | --                           |

---

## 注意事项

### 外部库依赖

| 文件                      | 依赖           | 未安装时       |
| ------------------------- | -------------- | -------------- |
| `00_rnn_eigen.cpp`        | Eigen 3.4+     | CMake 跳过     |
| `01_bptt_demo.cpp`        | Eigen 3.4+     | CMake 跳过     |
| `02_lstm_eigen.cpp`       | Eigen 3.4+     | CMake 跳过     |
| `03_lstm_libtorch.cpp`    | LibTorch       | CMake 跳过     |
| `04_text_processing.cpp`  | **纯 STL**     | 始终可编译     |
| `05_word2vec_libtorch.cpp`| LibTorch       | CMake 跳过     |

### LSTM 训练关键技巧

- **遗忘门偏置 = 1.0**：让网络在训练初期保留所有信息，随后学会选择性遗忘
- **梯度裁剪 threshold = 5.0**：标准值，LSTM 训练几乎必用
- **截断 BPTT window = 50~200**：平衡内存和长程依赖学习
- **Teacher Forcing ratio**：训练 100% 强制教学，推理 0%（自回归）
- **sigmoid/tanh 需要输入钳制**：`max(-50, min(50, x))` 防止 exp 溢出

### 与前后章节的关系

- **第 6 章 CNN**：图像的空间局部性 vs 文本的时间依赖性
- **第 8 章**：Transformer 的注意力机制取代了 RNN/LSTM 的递归结构
- **本章的 Seq2Seq** 是 Transformer 出现前的主流翻译架构，注意力机制在此基础上发展而来

---

## 拓展阅读

- **Understanding LSTM Networks** (Colah's Blog): https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- **The Unreasonable Effectiveness of RNNs** (Karpathy): https://karpathy.github.io/2015/05/21/rnn-effectiveness/
- **Original LSTM** (Hochreiter & Schmidhuber, 1997): https://www.bioinf.jku.at/publications/older/2604.pdf
- **Seq2Seq** (Sutskever et al., 2014): arXiv:1409.3215
- **Word2Vec** (Mikolov et al., 2013): arXiv:1301.3781
- **CS224n** (Stanford NLP): https://web.stanford.edu/class/cs224n/
- **BPE Tokenization**: https://huggingface.co/learn/nlp-course/chapter6/5
