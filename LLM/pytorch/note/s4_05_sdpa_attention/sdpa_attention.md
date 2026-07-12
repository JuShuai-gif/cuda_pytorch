# SDPA / FlashAttention 源码分析

> 源码路径: `torch/nn/attention/` — SDPA 后端选择与实现
> FlashAttention: `torch/nn/attention/_fa3.py`, `_fa4.py` — FA3/FA4 kernel 绑定
> 入口: `torch/nn/functional.py` — `scaled_dot_product_attention()`
> C++ 后端: `aten/src/ATen/native/transformers/` — CUDA attention kernels

## 0. 一句话总览

`torch.nn.functional.scaled_dot_product_attention` (SDPA) 是一个**后端自动选择的融合 attention API**。根据输入 shape/dtype/device，自动在 FlashAttention v2/3、Memory-Efficient Attention、C++ Math 实现之间选择最优后端，用户完全无感知。

---

## 一、SDPA 后端自动选择机制

### 1.1 后端优先级

```
if GPU + fp16/bf16 + causal=False + seq_len ≤ max:
    → FlashAttention v2 (forward, dropout, causal)
elif GPU + fp16/bf16 + causal 条件满足:
    → Memory-Efficient Attention (xformers 提供)
else:
    → C++ Math (naive matmul + softmax, 无融合)
```

### 1.2 用 `sdpa_kernel` 强制后端

```python
with torch.nn.attention.sdpa_kernel(
    torch.nn.attention.SDPBackend.FLASH_ATTENTION  # 强制 Flash
):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

可用的后端:
- `SDPBackend.FLASH_ATTENTION` — FlashAttention v2/v3
- `SDPBackend.EFFICIENT_ATTENTION` — Memory-efficient (xformers/ck)
- `SDPBackend.MATH` — C++ naive implementation
- `SDPBackend.CUDNN_ATTENTION` — cuDNN attention (torch >= 2.5)

---

## 二、FlashAttention 源码核心

### 2.1 为什么 FlashAttention 省显存

标准 attention:
```
S = Q @ K^T      [B, H, S, S] — O(S^2) 显存!
P = softmax(S)   [B, H, S, S] — O(S^2) 显存!
O = P @ V        [B, H, S, D]
```

FlashAttention 用 **tiling + online softmax** 消除 S 和 P 的完整物化:
```
for each tile of Q:
    for each tile of K, V:
        S_tile = Q_tile @ K_tile^T   ← 在 shared memory 中, 不写回 HBM
        P_tile = online_softmax(S_tile) ← 也在 shared memory 中
        O_tile += P_tile @ V_tile
output O
```

显存: O(S^2) → O(S)!

### 2.2 PyTorch 如何调用 FlashAttention

`torch/nn/attention/_fa3.py` (FlashAttention v3):

```python
# 通过 torch.ops.aten._flash_attention_forward 调用
# 底层是 cuDNN 或 NVIDIA 的 flash_attn library
torch.ops.aten._flash_attention_forward(
    query, key, value,
    dropout_p, softmax_scale, causal, ...
)
```

### 2.3 `sdpa_kernel` 上下文管理器源码

```python
# torch/nn/attention/__init__.py
class SDPBackend(enum.Enum):
    FLASH_ATTENTION = 0
    EFFICIENT_ATTENTION = 1
    MATH = 2
    CUDNN_ATTENTION = 3

@contextmanager
def sdpa_kernel(backends):
    # 设置 TLS 中的允许后端 → dispatch 时检查
    old = torch._C._get_sdpa_kernel_backends()
    torch._C._set_sdpa_kernel_backends(backends)
    try:
        yield
    finally:
        torch._C._set_sdpa_kernel_backends(old)
```

---

## 三、关键 API 对比

| API | 何时用 | 特点 |
|-----|--------|------|
| `F.scaled_dot_product_attention` | 通用场景 | 自动后端选择 |
| `nn.MultiheadAttention` | 经典 Transformer | 基于 SDPA，自带 proj |
| `sdpa_kernel(FLASH)` | 强制用 Flash | 调试/性能对比 |
| `F.attention(Q, K, V, causal=True)` | torch >= 2.2 | 更简洁的 SDPA alias |

---

## 四、关键源码位置速查

| 机制 | 文件 | 行号 |
|------|------|------|
| `scaled_dot_product_attention` | `torch/nn/functional.py` | — |
| SDPBackend 枚举 | `torch/nn/attention/__init__.py` | — |
| `sdpa_kernel` 上下文 | `torch/nn/attention/__init__.py` | — |
| FlashAttention v3 kernel | `torch/nn/attention/_fa3.py` | — |
| FlashAttention v4 kernel | `torch/nn/attention/_fa4.py` | — |
| Memory-efficient backend | `torch/nn/attention/experimental/` | — |
| CuDNN attention | `aten/src/ATen/native/transformers/cuda/` | — |

---

## 五、实战常见坑点

### 1. FlashAttention 没有触发
**排查**: `torch.backends.cuda.flash_sdp_enabled()` 返回 `True`? 检查 `q/k/v` dtype 是否为 fp16/bf16, device 是否 cuda, seq_len 是否不太大。

### 2. causal mask 与 FlashAttention 的不兼容
FlashAttention v2 支持 causal 但某些组合可能回退。查看日志:
```python
torch.backends.cuda.enable_flash_sdp(True)
# 若 sdpa_kernel(FLASH) 抛异常 → 当前输入不兼容
```

### 3. 显存比预期大
SDPA 只省 attention 部分 — QKV 矩阵 + output projection 仍然存在。如果 attention 只占显存 20%，FlashAttention 收益有限。
