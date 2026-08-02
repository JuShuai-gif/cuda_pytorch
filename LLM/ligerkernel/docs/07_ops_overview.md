# Liger-Kernel 算子一览与学习路线

> 背景：`liger_kernel/ops/` 下平铺了所有 Triton 算子，`__init__.py` 只做统一导出，
> 本身没有难度梯度。本文给出按难易程度排序的学习路线，并评估它对大模型训练
> 常见算子的覆盖情况。

## 一、按难度排序的学习路线

### 第一阶段 · 入门（单算子、公式简单，适合第一个上手）

```
utils.py                      # 先读这个，全是通用工具（calculate_settings、
                              # device_context、ensure_contiguous、三种 casting 常量等）
softmax.py                    # 一个 kernel 搞定，理解 tl.load / sum / exp
rms_norm.py                   # 最佳范本：前向 + 反向、rstd 缓存、casting 模式
geglu.py / swiglu.py           # 激活函数，forward/backward 对称
relu_squared.py               # 更简单的激活
kl_div.py                     # 分布散度，练反向推导
```

### 第二阶段 · 进阶（数学稍复杂，含反向图）

```
layer_norm.py                 # 比 rms_norm 多 mean 和 bias，经典范本
cross_entropy.py              # 数值稳定版 softmax + CE
rope.py                       # 位置编码，涉及三角函数和 reshape
group_norm.py                 # 分组归一化，块结构更复杂
sparsemax.py / poly_norm.py   # 冷门算子，适合拓展视野
```

### 第三阶段 · 融合优化（理解"为什么 fuse 省显存"）

```
fused_add_rms_norm.py         # 归一化 + 残差加法合并
fused_linear_cross_entropy.py # 最后一层线性 + CE 融合，省大激活
fused_linear_jsd.py / jsd.py  # 同上思路的 JSD 变体
vocab_parallel_cross_entropy.py  # 张量并行场景
dyt.py / tvd.py / mhc.py      # 各种 loss 变体
grpo_loss.py                  # RL 训练 loss
```

### 第四阶段 · 高难度（大 kernel、并行、MoE、注意力变体）

```
fused_moe.py / fused_moe_kernels.py   # MoE：专家路由 + 分组 GEMM，最难之一
tiled_mlp.py                   # 分块 MLP，显存优化
fused_neighborhood_attention.py / multi_token_attention.py  # MTA / 邻域注意力
qwen2vl_mrope.py / llama4_rope.py     # 多模态 / 新一代位置编码
attn_res.py / modulated_rms_norm.py   # 注意力残差、调制归一化
```

> 建议主线：`utils.py` → `rms_norm.py` → `layer_norm.py` → `cross_entropy.py`，
> 其余按需插入。

## 二、对大模型训练算子的覆盖情况

Liger-Kernel 的设计目标是**把 LLM 训练中"值得用 Triton 手工优化"的算子都做一遍**，
覆盖面很广：

| 类别 | 算子 |
| --- | --- |
| 归一化 | RMSNorm、LayerNorm、GroupNorm、ModulatedRMSNorm、PolyNorm |
| 激活 | GeGLU、SwiGLU、ReLU² |
| Loss | CE、JSD、KL、TVD、GRPO、MHC、VocabParallelCE |
| 位置编码 | RoPE、Qwen2-VL MRoPE、Llama4 RoPE |
| 混合专家 | FusedMoE、FusedMoEKernels |
| 注意力 | MultiTokenAttention、FusedNeighborhoodAttention、AttnRes |
| 融合优化 | FusedAddRMSNorm、FusedLinearCrossEntropy、FusedLinearJSD、TiledMLP |

但它**不是"训练所需的全部"**，明显缺失的类别：

| 缺失类别 | 说明 |
| --- | --- |
| 优化器 kernel | 没有 AdamW/SGD 的融合 kernel（那是 titans、Apex 等库的活） |
| 标准 FlashAttention | 只有 MTA / 邻域注意力变体，经典 sdpa/flash-attn 交给 PyTorch / FlashAttention 库 |
| embedding / tokenizer | 最底层的查表、位置 embedding 不在范围 |
| 部分激活变体 | 如 GeGLU 之外的 LReLU、Swish 等小算子 |

取舍原则：**只做"融合后能显著省显存/提速度"的算子**，标准库（PyTorch、
FlashAttention、Triton 教程）已做好的不重复造轮子。所以更准确的说法是：
**常见的高价值算子基本都覆盖了，但"所有"谈不上**。
