# LoRA 微调

> 知识点扩展：LoRA 原理、哪些层加 LoRA、视频模型 vs LLM 微调异同，回扣 FastVideo。

## 1. LoRA 原理

全量微调更新所有权重 `W (d×d)`，参数量 `d²`。LoRA 冻结 `W`，只训练低秩分解 `ΔW = B·A`：
```
W' = W + ΔW = W + B·A     A: (d×r), B: (r×d), r≪d
前向：y = Wx + (BA)x
```
参数量从 `d²` 降到 `2dr`。r=16 时省 99% 以上参数。

优点：省显存、快、可插拔、多个 LoRA 可切换。

**简单代码示例（教学用，最小 LoRA 线性层）**：
```python
import torch, torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank=16, alpha=32):
        super().__init__()
        self.base = base                              # 原始层，冻结
        self.base.requires_grad_(False)
        d_in, d_out = base.in_features, base.out_features
        self.A = nn.Parameter(torch.randn(rank, d_in) * 0.01)  # A 随机小值
        self.B = nn.Parameter(torch.zeros(d_out, rank))         # B 零初始化 → 初始 ΔW=0
        self.scaling = alpha / rank                    # 有效缩放
    def forward(self, x):
        return self.base(x) + self.scaling * (x @ self.A.T @ self.B.T)  # Wx + (BA)x

# 用法：把模型里的 nn.Linear 替换成 LoRALinear（FastVideo 的 replace_submodule 做的事）
layer = nn.Linear(1536, 1536)
lora = LoRALinear(layer, rank=16, alpha=32)
# 只有 A/B 可训练
trainable = [p for p in lora.parameters() if p.requires_grad]   # [A, B]
print(sum(p.numel() for p in trainable))   # 2*16*1536 ≈ 49K，远小于 1536² ≈ 2.4M
```
关键点：`B` 零初始化保证训练开始时输出 == 原模型；只有 `A`/`B` 进优化器。FastVideo 的 `enable_lora_training`（`train/utils/lora.py:192`）就是自动做这个替换 + 冻结。

### 1.1 为什么低秩有效

假设：微调对权重的改动 `ΔW` 本身是**低秩**的（在原模型基础上做小幅适配，不需要全秩变化）。实践验证 r=8~64 就能捕捉大部分微调收益。

### 1.2 初始化技巧

- `A` 用随机高斯初始化，`B` 用**零初始化** → 训练开始时 `ΔW = B·A = 0`，即初始等于原模型，训练稳定不破坏预训练权重。
- FastVideo `_replicate_lora_parameters` 后 `B` 仍应保证初始输出为 0（见 `layers/lora/linear.py`）。

### 1.3 显存节省来源

LoRA 省显存不只是参数少，更重要是**优化器状态少**：
- 全量：AdamW 需存 `全部参数×3`（参数+一阶矩+二阶矩）。
- LoRA：只存 `LoRA 参数×3`（冻结层无梯度、无优化器状态）。
对 14B 模型，这是几十 GB 的差别。

## 2. 哪些层适合加 LoRA

通常加在 attention 的投影层（信息交互的关键）：
```python
# FastVideo 默认（train/utils/lora.py:DEFAULT_LORA_TARGET_MODULES）
["q_proj", "k_proj", "v_proj", "o_proj",
 "to_q", "to_k", "to_v", "to_out", "to_qkv", "to_gate_compress"]
```
FFN 层也可加但收益递减。视频模型还可加 `to_gate_compress`（VSA 相关）。

### 2.1 rank / alpha 调参经验

| 场景 | rank | alpha |
|------|------|-------|
| 风格微调（轻量） | 8-16 | = rank 或 2×rank |
| 内容适配（中等） | 32-64 | 2×rank |
| 大幅改动 | 128+ | 2×rank |

- `alpha/rank` 是有效缩放。固定 alpha/rank 比值时，改 rank 主要改容量。
- rank 太大接近全量微调，失去 LoRA 优势且易过拟合。

## 3. FastVideo LoRA 注入

```
源码：train/utils/lora.py:enable_lora_training (L192)
```
```python
transformer.requires_grad_(False)             # 冻结
for name, module in named_modules():
    if _is_target_layer(name, target_modules):
        replace_submodule(transformer, name, get_lora_layer(module, rank, alpha))
_replicate_lora_parameters(transformer)        # DTensor Replicate
```

`alpha` 是缩放因子：`ΔW = (alpha/rank)·B·A`。alpha 越大 LoRA 影响越强。

## 4. 与 FSDP 配合

LoRA 参数很小，用 `DTensor.from_local(param, mesh, [Replicate()])` 复制（不分片），保证与 FSDP 拓扑兼容。

## 5. 只训练 LoRA 参数

```python
params = [p for p in transformer.parameters() if p.requires_grad]  # 只有 lora_A/B
optimizer = AdamW(params, ...)
```

## 6. 视频模型 vs LLM 微调异同

| 维度 | LLM LoRA | 视频 DiT LoRA |
|------|----------|--------------|
| 目标层 | attention q/k/v/o + MLP | attention 投影 + gate |
| 数据 | 文本 | 视频 latent + text embedding（Parquet） |
| loss | 交叉熵 | flow matching MSE |
| 显存瓶颈 | KV cache | 激活（长序列）+ VAE |
| 并行 | TP/PP | SP + FSDP |
| 步数 | 多 epoch | 常几千步 |

相同点：都冻结主干、只训低秩、可插拔。

## 7. LoRA 提取（反向）

`scripts/lora_extraction/extract_lora.py`：从全量微调模型反推 LoRA：
```python
delta = FT_weights - base_weights
U, S, V = SVD(delta)
A, B = top_r(U, S, V)    # 取 top-r 奇异值
```
把已发布的全量微调模型压成轻量 LoRA。

## 8. 推理时加载 LoRA

```
源码：pipelines/lora_pipeline.py:set_lora_adapter (L296)
```
加载 `.safetensors` LoRA 权重，`merge_lora_weights` 可物理合并到基础层（推理更快）或保持分离（可切换）。

## 9. 回扣源码
| 概念 | 源码 |
|------|------|
| 注入 | `train/utils/lora.py:enable_lora_training` |
| LoRA 层 | `layers/lora/linear.py` |
| 推理加载 | `pipelines/lora_pipeline.py` |
| 提取 | `scripts/lora_extraction/extract_lora.py` |

## 10. 延伸
- LoRA 流程：[`../03_core_flows/08_lora_finetune_flow.md`](../03_core_flows/08_lora_finetune_flow.md)
- 训练 LoRA 实践：[`../06_practical_guides/05_how_to_train_lora.md`](../06_practical_guides/05_how_to_train_lora.md)
