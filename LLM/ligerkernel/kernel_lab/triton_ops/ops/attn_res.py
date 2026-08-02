"""
Attention Residuals (AttnRes) - Kimi/Moonshot AI

Replaces standard residual connections h_l = h_{l-1} + f_l(RMSNorm(h_{l-1}))
with softmax attention over depth for dynamic weighting:

  V = stack(blocks)           # [N, B, T, D]
  K = RMSNorm(V)              # per-block normalize
  scores = einsum(w, K)       # [N, B, T] — w is [D] learned query
  alpha = softmax(scores, 0)  # over block dim
  h = einsum(alpha, V)        # [B, T, D] — weighted sum

Solves PreNorm dilution: deep layer contributions being diluted.
Paper: https://arxiv.org/abs/2603.15031

Triton optimizations:
1. Single kernel fuses RMSNorm + dot + softmax + weighted sum
2. Each program handles one (batch, token) position
3. N is small (≤16 blocks), scores fit in registers
"""

import torch
import triton
import triton.language as tl

from ligerkernel.ops.utils import ensure_contiguous

# ============================================================================
# Forward Kernel
# ============================================================================


@triton.jit
def _attn_res_fwd_kernel(
    # [N, B*T, D] 堆叠的块值（各注意力残差块的输出）
    V_ptr,
    # [D] 可学习的伪查询向量
    W_query_ptr,
    # [D] keys 的 RMSNorm 权重
    W_norm_ptr,
    # [B*T, D] 输出（加权求和结果）
    Out_ptr,
    # [B*T, N] 注意力权重（softmax 结果，供反向传播保存）
    Alpha_ptr,
    # [B*T, N] 每个 (token, block) 的 RMSNorm 逆标准差
    RSTD_ptr,
    # N 块数量
    n_blocks,
    # B*T token 总数
    n_tokens,
    # 隐藏维度
    D,
    eps,
    # 编译期常量：D 的块大小
    BLOCK_D: tl.constexpr,
    # 编译期常量：最大块数（决定寄存器中 scores/alpha 数组大小）
    MAX_BLOCKS: tl.constexpr,
):
    """
    Forward: one program per token position.

    tok = tl.program_id(0) 是当前 kernel program 在 grid 里的编号。
    输入张量被压扁成 [N, B*T, D]：原 batch 和序列两个维度（B 和 T）被合并成一维，
    总共 B*T 个 token，每个 token 只对应一个线性下标。
    program_id(0) 就返回这个下标：第 0 号 program 处理第 0 个 token，
    第 tok 号 program 处理第 tok 个 token。
    例如 B=2, T=3 时 B*T=6，tok=4 表示第 1 个 batch、第 1 个位置（1*3+1）的那个 token。

    这样每个 token 的计算完全独立（AttnRes 的定义中注意力权重只对单个 token
    的块维度做 softmax），天然适合 GPU 大规模并行。
    """
    # tok 就是从 0 到 B*T-1 的 token 编号，表示当前 program 处理第几个 token
    tok = tl.program_id(0)
    # 隐藏维度的块索引 [0, BLOCK_D)，用于向量化加载/计算
    cols = tl.arange(0, BLOCK_D)
    # 当 D 不是 BLOCK_D 的整数倍时，屏蔽越界的维度
    d_mask = cols < D

    # 加载共享向量：将 [D] 的 w_query / w_norm 从全局显存（HBM）加载到寄存器/SRAM。
    # 两者被所有 token、所有块共用，只需一次加载即可全程复用，避免重复访问显存。
    # w_query：可学习伪查询向量，用于下方计算 score = dot(w_query, k)
    w_query = tl.load(W_query_ptr + cols, mask=d_mask, other=0.0).to(tl.float32)
    # w_norm：keys 的 RMSNorm 权重，用于下方做仿射缩放 k = v * rstd * w_norm
    w_norm = tl.load(W_norm_ptr + cols, mask=d_mask, other=0.0)

    # Pass 1：对每个块计算 score = dot(w_query, RMSNorm(v_i))
    # scores[i] 通过 tl.where 存入寄存器的第 i 个位置（绕过 Triton 不支持对
    # 寄存器向量做真正标量索引的限制；该模式编译为带条件选择的搬移指令，
    # 全程保持在寄存器中，无需溢出到全局内存）。
    # 初始化为 -inf：真实块会被覆盖，填充位保持 -inf，softmax 时 exp(-inf)=0 自然被排除
    scores = tl.zeros((MAX_BLOCKS,), dtype=tl.float32) + float("-inf")
    # score_max 为标量，记录所有块中的最大 score，用于 softmax 数值稳定
    score_max = tl.full((), float("-inf"), dtype=tl.float32)

    for i in tl.static_range(0, MAX_BLOCKS):
        if i < n_blocks:
            # 第 i 个块中当前 token 在 [N, B*T, D] 张量中的行偏移，再按 D 维展开
            v_off = i * n_tokens * D + tok * D
            # 加载当前 token 在第 i 个块的 D 维向量
            v = tl.load(V_ptr + v_off + cols, mask=d_mask, other=0.0).to(tl.float32)

            # RMSNorm：k = v * rstd * w_norm
            # 先算均方根 ms = mean(v^2)，再取逆标准差 rstd = 1/sqrt(ms + eps)
            ms = tl.sum(v * v, axis=0) / D
            rstd = tl.rsqrt(ms + eps)
            # 存储 rstd 供反向传播复用；布局 [B*T, N]，每个 token 的 N 个元素连续排列
            tl.store(RSTD_ptr + tok * n_blocks + i, rstd)

            # 归一化并做仿射缩放（逐维度乘上可学习权重 w_norm）
            k = (v * rstd).to(w_norm.dtype) * w_norm

            # score = dot(w_query, k)，得到当前块的点积分数（标量）
            sc = tl.sum(w_query * k.to(tl.float32), axis=0)
            # 用 tl.where 把 sc 写入 scores 向量的第 i 个寄存器位置
            scores = tl.where(tl.arange(0, MAX_BLOCKS) == i, sc, scores)
            # 更新历史最大分数，用于 softmax 数值稳定
            score_max = tl.maximum(score_max, sc)

    # 对块维度做 softmax（数值稳定版）
    # 真实块的分数先减去 score_max 再取 exp；填充位（i >= n_blocks）直接置 0
    exp_scores = tl.where(
        tl.arange(0, MAX_BLOCKS) < n_blocks,
        tl.exp(scores - score_max),
        0.0,
    )

    # 归一化：alpha = exp(scores) / sum(exp(scores))，得到 [MAX_BLOCKS] 的注意力权重向量
    sum_exp = tl.sum(exp_scores, axis=0)
    alpha = exp_scores / sum_exp  # [MAX_BLOCKS]

    # 存储 alpha 供反向传播复用；布局 [B*T, N]，每个 token 的 N 个元素连续排列
    # tl.where 从 alpha 向量中提取第 i 个标量（与写入 scores 相同的寄存器索引技巧）
    for i in tl.static_range(0, MAX_BLOCKS):
        if i < n_blocks:
            a_i = tl.sum(tl.where(tl.arange(0, MAX_BLOCKS) == i, alpha, 0.0))
            tl.store(Alpha_ptr + tok * n_blocks + i, a_i)

    # Pass 2：加权求和 h = sum(alpha_i * v_i)，即最终残差输出
    h = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for i in tl.static_range(0, MAX_BLOCKS):
        if i < n_blocks:
            # 重新加载第 i 个块的 v（避免为存储 alpha 结果而额外保留全部块）
            v_off = i * n_tokens * D + tok * D
            v = tl.load(V_ptr + v_off + cols, mask=d_mask, other=0.0).to(tl.float32)
            # 提取第 i 个块的注意力权重标量
            a_i = tl.sum(tl.where(tl.arange(0, MAX_BLOCKS) == i, alpha, 0.0))
            # 累加 alpha_i * v_i
            h += a_i * v

    # tl.store 会隐式处理 dtype 转换，将 h 写入输出张量
    tl.store(Out_ptr + tok * D + cols, h, mask=d_mask)


# ============================================================================
# Backward Kernel
# ============================================================================


@triton.jit
def _attn_res_bwd_kernel(
    dOut_ptr,  # [B*T, D] upstream gradient
    V_ptr,  # [N, B*T, D]
    W_query_ptr,  # [D]
    W_norm_ptr,  # [D]
    Alpha_ptr,  # [B*T, N] saved from forward
    RSTD_ptr,  # [B*T, N] saved from forward
    dV_ptr,  # [N, B*T, D] output gradients
    dW_query_ptr,  # [D] atomic accumulate
    dW_norm_ptr,  # [D] atomic accumulate
    n_blocks,
    n_tokens,
    D,
    eps,
    BLOCK_D: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
):
    """Backward: one program per token."""
    tok = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    d_mask = cols < D

    dh = tl.load(dOut_ptr + tok * D + cols, mask=d_mask, other=0.0).to(tl.float32)
    w_query = tl.load(W_query_ptr + cols, mask=d_mask, other=0.0).to(tl.float32)
    w_norm = tl.load(W_norm_ptr + cols, mask=d_mask, other=0.0).to(tl.float32)

    # Load alpha for all blocks — layout [B*T, N], contiguous load
    d_alpha = tl.zeros((MAX_BLOCKS,), dtype=tl.float32)
    alpha = tl.zeros((MAX_BLOCKS,), dtype=tl.float32)

    for i in tl.static_range(0, MAX_BLOCKS):
        if i < n_blocks:
            v_off = i * n_tokens * D + tok * D
            v = tl.load(V_ptr + v_off + cols, mask=d_mask, other=0.0).to(tl.float32)
            a_i = tl.load(Alpha_ptr + tok * n_blocks + i)

            da_i = tl.sum(dh * v, axis=0)
            d_alpha = tl.where(tl.arange(0, MAX_BLOCKS) == i, da_i, d_alpha)
            alpha = tl.where(tl.arange(0, MAX_BLOCKS) == i, a_i, alpha)

    # Softmax backward: d_score_i = alpha_i * (d_alpha_i - sum_j(alpha_j * d_alpha_j))
    sum_a_da = tl.sum(alpha * d_alpha, axis=0)
    d_scores = alpha * (d_alpha - sum_a_da)

    # For each block: compute dV_i and accumulate dW_query, dW_norm
    dw_query_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    dw_norm_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for i in tl.static_range(0, MAX_BLOCKS):
        if i < n_blocks:
            v_off = i * n_tokens * D + tok * D
            v = tl.load(V_ptr + v_off + cols, mask=d_mask, other=0.0).to(tl.float32)
            a_i = tl.sum(tl.where(tl.arange(0, MAX_BLOCKS) == i, alpha, 0.0))
            ds_i = tl.sum(tl.where(tl.arange(0, MAX_BLOCKS) == i, d_scores, 0.0))
            rstd = tl.load(RSTD_ptr + tok * n_blocks + i)

            # dV_i from weighted sum: alpha_i * dh
            dv_from_sum = a_i * dh

            # dV_i from score path: d_score_i * d(score_i)/d(v_i)
            # score_i = dot(w_query, RMSNorm(v_i) * w_norm)
            # d(score_i)/d(v_i) = d(score_i)/d(k_i) * d(k_i)/d(v_i)
            # where k_i = RMSNorm(v_i) * w_norm
            # d(score_i)/d(k_i) = w_query
            # d(k_i)/d(v_i) = w_norm * d(RMSNorm)/d(v_i)

            # RMSNorm backward: d(v*rstd)/dv = rstd * (I - (1/D) * rstd^2 * v * v^T)
            v_norm = v * rstd
            dk = ds_i * w_query * w_norm  # [D]
            sum_dk_v = tl.sum(dk * v, axis=0)
            dv_from_score = rstd * dk - rstd * rstd * rstd * (sum_dk_v / D) * v

            dv_total = dv_from_sum + dv_from_score
            # tl.store handles implicit dtype conversion
            tl.store(dV_ptr + v_off + cols, dv_total, mask=d_mask)

            # dW_query += d_score_i * k_i
            k_i = v_norm * w_norm
            dw_query_acc += ds_i * k_i

            # dW_norm += d_score_i * w_query * v_norm (element-wise)
            dw_norm_acc += ds_i * w_query * v_norm

    tl.atomic_add(dW_query_ptr + cols, dw_query_acc, mask=d_mask)
    tl.atomic_add(dW_norm_ptr + cols, dw_norm_acc, mask=d_mask)


# ============================================================================
# Python wrappers
# ============================================================================


def _next_pos2(n):
    return triton.next_power_of_2(n)


def _get_max_blocks(n_blocks):
    """Round up to constexpr-friendly value."""
    for mb in [4, 8, 16, 32]:
        if n_blocks <= mb:
            return mb
    return 32


def attn_res_forward(blocks, w_query, w_norm, eps=1e-6):
    """
    Args:
        blocks: list of N tensors [B, T, D] or stacked [N, B, T, D]
        w_query: [D] learned pseudo-query
        w_norm: [D] RMSNorm weight for keys
    Returns:
        h: [B, T, D] weighted output
        V: [N, B*T, D] stacked (saved for bwd)
        alpha: [B*T, N] attention weights
        rstd: [B*T, N] per-token rstd
    """
    if isinstance(blocks, (list, tuple)):
        V = torch.stack(blocks)  # [N, B, T, D]
    else:
        V = blocks
    orig_shape = V.shape  # [N, B, T, D] or [N, B*T, D]
    N = V.shape[0]
    D = V.shape[-1]

    # Flatten to [N, B*T, D]
    V_3d = V.reshape(N, -1, D).contiguous()
    n_tokens = V_3d.shape[1]

    w_query = w_query.contiguous()
    w_norm = w_norm.contiguous()

    Out = torch.empty(n_tokens, D, device=V.device, dtype=V.dtype)
    # Layout [B*T, N] for coalesced access per token
    Alpha = torch.empty(n_tokens, N, device=V.device, dtype=torch.float32)
    RSTD = torch.empty(n_tokens, N, device=V.device, dtype=torch.float32)

    BLOCK_D = _next_pow2(D)
    MAX_BLOCKS = _get_max_blocks(N)
    nw = 4
    if BLOCK_D >= 2048:
        nw = 8
    if BLOCK_D >= 8192:
        nw = 16

    _attn_res_fwd_kernel[(n_tokens,)](
        V_3d,
        w_query,
        w_norm,
        Out,
        Alpha,
        RSTD,
        N,
        n_tokens,
        D,
        eps,
        BLOCK_D=BLOCK_D,
        MAX_BLOCKS=MAX_BLOCKS,
        num_warps=nw,
    )

    # Reshape output to match input spatial dims
    out_shape = list(orig_shape[1:])  # [B, T, D] or [B*T, D]
    return Out.view(out_shape), V_3d, Alpha, RSTD


def attn_res_backward(dh, V_3d, w_query, w_norm, Alpha, RSTD, eps=1e-6):
    """
    Returns: dV [N, B*T, D], dW_query [D], dW_norm [D]
    """
    dh = dh.contiguous()
    N, n_tokens, D = V_3d.shape
    dh_2d = dh.reshape(n_tokens, D)

    dV = torch.empty_like(V_3d)
    dW_query = torch.zeros(D, dtype=torch.float32, device=dh.device)
    dW_norm = torch.zeros(D, dtype=torch.float32, device=dh.device)

    BLOCK_D = _next_pow2(D)
    MAX_BLOCKS = _get_max_blocks(N)
    nw = 4
    if BLOCK_D >= 2048:
        nw = 8
    if BLOCK_D >= 8192:
        nw = 16

    _attn_res_bwd_kernel[(n_tokens,)](
        dh_2d,
        V_3d,
        w_query,
        w_norm,
        Alpha,
        RSTD,
        dV,
        dW_query,
        dW_norm,
        N,
        n_tokens,
        D,
        eps,
        BLOCK_D=BLOCK_D,
        MAX_BLOCKS=MAX_BLOCKS,
        num_warps=nw,
    )

    return dV, dW_query.to(w_query.dtype), dW_norm.to(w_norm.dtype)


# ============================================================================
# PyTorch Autograd Function
# ============================================================================


class LigerAttnResFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, V_stacked, w_query, w_norm, eps):
        # 保存原始输入形状 [N, B, T, D] 或 [N, B*T, D]，反向时用于还原 dV 形状
        ctx.orig_shape = V_stacked.shape
        h, V_3d, Alpha, RSTD = attn_res_forward(V_stacked, w_query, w_norm, eps)
        # 保存反向传播所需的中间张量：V_3d（三维块）、w_query（查询向量）、w_norm（归一化权重）、
        # Alpha（注意力权重）、RSTD（RMSNorm 逆标准差）
        ctx.save_for_backward(V_3d, w_query, w_norm, Alpha, RSTD)
        # 保存 eps 供反向复用
        ctx.eps = eps
        return h

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dh):
        # 取回前向保存的张量
        V_3d, w_query, w_norm, Alpha, RSTD = ctx.saved_tensors
        # 计算三个输入的梯度：块梯度 dV、查询向量梯度 dW_query、归一化权重梯度 dW_norm
        dV, dW_query, dW_norm = attn_res_backward(
            dh, V_3d, w_query, w_norm, Alpha, RSTD, ctx.eps
        )
        # 将 dV 还原为原始输入形状 [N, B, T, D]
        dV = dV.view(ctx.orig_shape)
        # 与 forward 的四个入参一一对应返回梯度；eps 是标量无需梯度，返回 None
        return dV, dW_query, dW_norm, None
