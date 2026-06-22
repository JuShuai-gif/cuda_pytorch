# Paper 17: QServe — W4A8KV4 Quantization and System Co-design for Efficient LLM Serving (Lin et al., MLSys 2025)

> 论文全称：**QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving**
> 发表会议：MLSys 2025
> 作者：Yujun Lin, Haotian Tang, Shang Yang, Zhekai Zhang, Guangxuan Xiao, Chuang Gan, Song Han（MIT HAN Lab）

---

## 1. 论文解决什么问题

LLM 推理服务的核心瓶颈是 GPU 显存带宽和容量：在 A100 (80GB) 上部署 LLaMA-70B 时，仅存储 FP16 权重就需要 ~140GB（超过单卡容量），而 KV Cache 在多请求并发时增长极快，4bit 权重（W4）需手写 CUDA kernel 才能高效 decode。QServe **联合设计权重量化方案（W4A8KV4）与推理系统**，在 LLaMA-70B 上实现比 vLLM 高 **2.7× 吞吐量**，并使 A100 能服务比 FP16 多 **3.5×** 的并发请求。

---

## 2. 核心方法

### W4A8KV4 量化方案

| 组件 | 精度 | 理由 |
|------|------|------|
| **W (权重)** | 4-bit | 权重占据模型大小主体，4-bit 最大程度减少显存 |
| **A (激活值)** | 8-bit | 激活值对量化更敏感（outliers），8-bit 保持精度 |
| **KV Cache** | 4-bit | KV Cache 是长序列时最大的显存消费者，4-bit 大幅压缩 |

这种非对称量化方案在精度和效率间取得最佳平衡：权重和 KV Cache 用最激进的精度（省显存），激活值保留较高精度（少损失）。

### SmoothAttention：平滑 KV Cache 量化

KV Cache 量化面临的核心问题是通道间的**异常值（outliers）**——某些通道的值远大于其他通道，导致量化步长被迫拉大，低值通道的信息被淹没。

设 $X \in \mathbb{R}^{N \times d}$ 为 Key 或 Value 矩阵，对逐 token 量化：

$$\hat{X} = \text{round}\left(\frac{X - \min X}{\max X - \min X} \cdot (2^{B} - 1)\right)$$

当某通道存在 outlier 时，$\max X - \min X$ 被 outlier 主导，正常通道的量化分辨率极低。

**SmoothAttention 方法**：在量化前对每通道乘以一个平滑因子 $s_c$，在后续的注意力计算中通过数学等价变换消除 $s_c$ 的影响：

$$QK^T = (Q \cdot \text{diag}(s)) \cdot (K \cdot \text{diag}(s^{-1}))^T$$

其中 $s_c = \max(|K_{:,c}|)^{\alpha}$（$\alpha \approx 0.5$ 为平滑强度超参）。

### GPU Kernel 协同设计

- **GEMM kernel**：针对 W4A8 矩阵乘法编写 optimized CUDA kernel，利用 Tensor Core 的 `mma.sp` 指令（Ampere+）
- **KV Cache 管理**：GPU kernel 直接对 4-bit KV Cache 进行 Attention 计算，避免 int4→fp16 的中间转换开销
- **Prefill-Decode 分离调度**：prefill 阶段密集计算 GPU bound，decode 阶段内存 bound，使用不同的 CUDA stream 异步并行

---

## 3. 关键公式

### 逐通道量化与反量化

对通道 $c$ 的张量 $X_c$，quantization scale 为：

$$\Delta_c = \frac{\max(X_c) - \min(X_c)}{2^B - 1}$$

量化值：
$$X_c^{\text{quant}} = \text{round}\left(\frac{X_c}{\Delta_c}\right)$$

### SmoothAttention 的等价变换

设平滑因子 $s \in \mathbb{R}^{d_k}$，平滑后的注意力计算：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{(Q \odot s)(K \oslash s)^T}{\sqrt{d_k}}\right)V$$

其中 $\odot$ 为逐元素乘，$\oslash$ 为逐元素除。由于 $s_c \cdot s_c^{-1} = 1$，结果不变。

### 吞吐量模型

设 batch size 为 $B$，序列长度为 $L$，总吞吐量（tokens/s）：

$$Throughput = \frac{B}{\frac{L_{\text{prefill}} \cdot B}{BW_{\text{compute}}} + \frac{B}{\text{memory bandwidth bound}}}$$

W4A8KV4 的带宽需求仅为 FP16 的：
$$\text{Bandwidth Ratio} = \frac{4 + 8 + 4}{16 + 16 + 16} = \frac{16}{48} = \frac{1}{3}$$

---

## 4. 实验结论

| 模型 | 方法 | A100 吞吐 (tokens/s) | 最大 Batch Size | WikiText-2 PPL |
|------|------|-----------------------|-----------------|----------------|
| LLaMA-2-7B | FP16 (vLLM) | 2048 | 64 | 5.47 |
| LLaMA-2-7B | QServe | **6154** | **256** | **5.68** |
| LLaMA-2-13B | FP16 (vLLM) | 1124 | 32 | 5.09 |
| LLaMA-2-13B | QServe | **3257** | **128** | **5.31** |
| LLaMA-2-70B | FP16 (vLLM) | 418 | OOM (>1) | — |
| LLaMA-2-70B | QServe | **1136** | **8** | **3.32** |

- **LLaMA-70B** 吞吐量 **2.7× 于 vLLM**，同时在 80GB A100 上能一次性加载模型（FP16 则需模型并行）
- SmoothAttention 使 4-bit KV Cache 的精度损失 <0.1 PPL
- QServe 在 LLaMA-3（8B/70B）和 Mistral-7B 上同样有效，通用性强
- 与 AWQ、GPTQ 等 W4 方案对比，QServe 的 W4A8KV4 在相同权重量化精度下，PPL 更低（由于保留了 A8 的激活精度）

---

## 5. 工业价值

- **降低 LLM 服务成本**：在 A100 上用 QServe 部署 LLaMA-70B 时，生成每个 token 的成本约为 FP16 的 1/3
- **更高并发**：KV Cache 4-bit 使单卡的请求容纳量提升 4×，在 API 服务场景中直接降低排队延迟
- **Edge LLM 部署**：在 Jetson Orin（32GB）等边缘设备上，W4A8KV4 使 7B 模型可以运行（FP16 会 OOM）
- **已被社区采纳**：QServe 的内核设计理念已被 llama.cpp（GGUF Q4_K_M）和 TensorRT-LLM 参考

---

## 6. 与课程 Lecture 的关系

- **Lecture 13（LLM Deployment）**：QServe 是 LLM 部署课程的核心论文，完整展示了从量化方案选择到 GPU kernel 实现的端到端优化流程
- **Lecture 4-6（Quantization）**：W4A8KV4 的量化方案设计和 SmoothAttention 的平滑技术是量化技术的直接应用
- **Lecture 1（Efficiency Metrics）**：论文的吞吐量（throughput）、延迟（latency）、显存占用三维评估是效率指标的综合实战案例
- **Lecture 7（Co-design）**：QServe 的核心哲学——量化方案与系统实现联合设计——贯穿课程始终

---

## 7. 我应该如何复现

1. **环境准备**：
   - GPU：NVIDIA A100 (80GB) 或 RTX 4090（小模型测试可用）
   - 软件：CUDA 12.1+，PyTorch 2.1+，安装 QServe（`pip install qserve` 或从 GitHub 源码编译）
2. **转换模型**：
   - 下载 LLaMA-2-7B / 13B / 70B 的 HuggingFace checkpoint
   - 运行 QServe 的量化脚本：`python quantize.py --model meta-llama/Llama-2-7b-hf --wbits 4 --abits 8 --kvbits 4`
   - 量化过程使用校准数据集（wiki 或 c4 的子集，128 samples）
3. **启动推理服务**：
   ```bash
   python -m qserve.server --model-path ./llama-2-7b-qserve \
       --max-num-seqs 256 --max-seq-len 8192
   ```
4. **Benchmark**：
   - 使用 ShareGPT 数据集模拟真实对话请求分布
   - 测量吞吐量（tokens/s）随 batch size 变化曲线
   - 对比 vLLM（FP16）在相同硬件上的吞吐量和延迟
5. **验证精度**：
   - 在 WikiText-2 上测 PPL
   - 在 MMLU、HellaSwag 等 benchmark 上验证零样本准确率
6. **关键注意事项**：
   - 编译 CUDA kernel 时需要匹配 GPU 架构（`TORCH_CUDA_ARCH_LIST="8.0"` for A100）
   - 对 70B 模型，单卡 A100 即可跑 QServe，但 FP16 baseline 需要 tensor parallel
