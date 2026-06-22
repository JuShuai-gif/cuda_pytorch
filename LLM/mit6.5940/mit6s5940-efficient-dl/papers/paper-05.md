# Paper 05: SmoothQuant (Xiao et al., ICML 2023)

> 论文全称：**SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models**
> 发表会议：ICML 2023
> 作者：Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han（MIT HAN Lab 与 NVIDIA 合作）

---

## 1. 论文解决什么问题

将 LLM 量化为 INT8 精度执行 W8A8（权重和激活都是 8 位整数）推理面临的核心问题：**激活值中存在异常大的离群值（outliers）**。

具体表现：
- LLM 的激活张量（activation）中，约 0.1% 的通道值比其余通道大 100 倍以上
- 这些 outliers 集中在固定的几个通道上（与 token 无关，只与通道位置有关）
- 如果对激活值做 per-tensor 量化，整个张量的量化粒度由这些离群值决定，导致 99.9% 的"正常"值量化精度极差
- Per-channel 量化可以解决激活的问题，但在推理时计算效率低（需要额外的乘加操作）
- Per-channel 量化权重 + per-tensor 量化激活是目前的主流方案（如 TensorRT-LLM），但激活的 outliers 问题仍然存在

**SmoothQuant** 提出将激活的量化难度"平滑转移"到权重上，实现 W8A8 高效推理。

---

## 2. 核心方法

### 核心思想：量化难度的迁移
激活 $X$ 有离群值（大值），导致量化困难。权重 $W$ 值分布相对均匀，容易量化。通过数学上等价的缩放，将量化难度从激活转移到权重：

$$Y = WX = (W \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot X) = \hat{W} \cdot \hat{X}$$

- **权重**除以 $s_j$（变小 → 更平滑，更容易量化）
- **激活**乘以 $s_j$（离群通道变小 → 所有通道归一到相近范围 → 更容易 per-tensor 量化）
- 因为权重分布本来就很均匀，除以 $s_j$ 引入的额外量化误差很小

### 迁移强度 $\alpha$ 的计算
对于每个通道 $j$：

$$s_j = \max(|X_j|)^{\alpha}$$

其中迁移强度 $\alpha$（migration strength）是可调超参数：
- $\alpha = 0$：全部迁移到激活 → 激活离群值不变（退化为不对激活做任何处理）
- $\alpha = 1$：全部迁移到权重 → 权重被过度扰动
- 实验：$\alpha = 0.5$ 对大多数 LLM 是最优的；对 GLM-130B 需要特定调优

### 实际执行流程
1. **校准阶段**：输入少量校准数据（128-256 个 token），收集每层的激活值
2. **计算平滑因子**：对每个输入通道计算 $s_j$
3. **缩放处理**：将 $W$ 除以 $s$，将 $X$ 乘以 $s$（在推理时 bake 到前一层）
4. **量化**：$W$ 做 per-channel INT8 量化，$X$ 做 per-tensor INT8 量化
5. **INT8 推理**：使用 INT8 GEMM kernel 执行推理

---

## 3. 关键公式

### 平滑变换目标
将离群值问题从激活转移到权重：

$$X_{\text{smooth}} = \text{diag}(s) \cdot X, \quad W_{\text{smooth}} = W \cdot \text{diag}(s)^{-1}$$

变换后 $X_{\text{smooth}}$ 的离群值被大幅抑制，而 $W_{\text{smooth}}$ 的分布仍然可控。

### Per-tensor 量化后的误差
变换前（激活离群值主导量化尺度导致误差大）：

$$\text{quant_error_before} \propto \max(|X|) \gg \text{mean}(|X|)$$

变换后（离群值被抑制，量化尺度更合理）：

$$\text{quant_error_after} \propto \max(|X_{\text{smooth}}|) \approx \text{mean}(|X_{\text{smooth}}|)$$

### SmoothQuant 在 transformer 中的应用
对 Q, K, V 和 FC1 输入使用 SmoothQuant：

$$\text{Attention}: \quad Y = (W_Q s_Q^{-1}) (s_Q X) \cdot (W_K s_K^{-1}) (s_K X)^T \cdot (W_V s_V^{-1}) (s_V X)$$
$$\text{FFN}: \quad Y = (W_{\text{FC2}} s_{\text{FC1}}^{-1}) (s_{\text{FC1}} \cdot \sigma((W_{\text{FC1}} s_{\text{FC1}}^{-1})(s_{\text{FC1}} X)))$$

---

## 4. 实验结论

### LLaMA 系列模型（W8A8）

| 模型 | FP16 PPL | SmoothQuant W8A8 PPL | 相对退化 |
|------|----------|---------------------|---------|
| LLaMA-7B | 5.68 | 5.71 | +0.5% |
| LLaMA-13B | 5.09 | 5.12 | +0.6% |
| LLaMA-30B | 4.10 | 4.12 | +0.5% |
| LLaMA-65B | 3.56 | 3.58 | +0.6% |
| OPT-13B | 10.13 | 10.18 | +0.5% |
| BLOOM-176B | 8.25 | 8.28 | +0.4% |

- **W8A8 精度几乎无损失**（PPL 退化 < 1%）
- **推理速度提升**：在 NVIDIA A100 上实现 1.56× 加速
- **内存节省**：权重和激活各节省 50% 显存
- **通用性**：在 LLaMA、OPT、BLOOM、GLM 等多个模型族上验证有效

---

## 5. 工业价值

- **成为 LLM W8A8 事实标准**：被 NVIDIA TensorRT-LLM、vLLM、MLC-LLM 等主流推理框架集成
- **实际产品落地**：NVIDIA 与 MIT HAN Lab 联合研发，已集成到 TensorRT-LLM
- **学术界高影响力**：ICML 2023 发表 + 被众多后续量化工作引用
- **产业启示**：W8A8 量化是 LLM 部署的"甜点"——精度几乎无损、硬件支持好、加速显著

---

## 6. 与课程 Lecture 的关系

- **Lecture 4 (Quantization)**：SmoothQuant 是量化技术从传统 CNN 到 LLM 扩展的代表性工作
- **Lecture 11 (LLM Efficiency)**：本论文是课程 LLM 效率专题的核心论文之一
- **Lecture 2 (Efficient Inference)**：W8A8 是当前 LLM 推理效率提升的重要技术
- **Lab 2 (Quantization)**：SmoothQuant 的平滑思想可以通过对 toy 张量做手工量化来验证

---

## 7. 我应该如何复现

1. **安装依赖**：`pip install transformers accelerate bitsandbytes`
2. **加载模型**：`from transformers import AutoModelForCausalLM`，加载 LLaMA-7B（或其他开源 LLM）
3. **实现 SmoothQuant**：
   - 收集校准数据：用 128-256 个 token 跑一次前向，hook 每层的激活值
   - 计算平滑因子：对每层的 `attention QKV` 和 `FFN FC1` 的输入激活，计算 `s_j = max(|X_j|)^α`
   - 执行平滑：`w.smooth = w / s`, `x.smooth = x * s`
   - 量化：用 PyTorch 的 `torch.quantize_per_channel` 量化权重，`torch.quantize_per_tensor` 量化激活
4. **验证**：在 Wikitext-2 上计算 perplexity，对比 FP16 的退化程度
5. **代码简化版**：
   ```python
   # 伪代码：对 attention QKV 层应用 SmoothQuant
   alpha = 0.5
   s = torch.pow(torch.max(torch.abs(x_calib), dim=0)[0], alpha)
   # Softmax 前后注意：scale factor 需要正确传递
   w_q_smooth = w_q / s.unsqueeze(-1)   # 各通道独立除以 s
   w_k_smooth = w_k / s.unsqueeze(-1)
   w_v_smooth = w_v / s.unsqueeze(-1)
   ```
6. **关键细节**：Attention 的 Q 和 K 共享同样的输入激活，所以 `s` 对 Q 和 K 是一致的，这很重要因为 softmax 需要保持一致性
7. **硬件加速**：如果有 GPU，用 `torch.amp.autocast('cuda')` 或 TensorRT-LLM 直接体验 W8A8 加速效果

