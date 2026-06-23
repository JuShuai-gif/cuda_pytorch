import torch


# 对称量化 / 反量化（per-tensor，整张量共用一个 scale）
# 对称量化假设数据以 0 为中心，量化区间为 [-qmax, qmax]，zero_point 恒为 0。
# scale 由整张量绝对值的最大值决定 —— 这正是它对 outlier（离群值）极度敏感的根源。
def symmetric_quant_dequant(x: torch.Tensor, bits=8):
    qmax = 2 ** (bits - 1) - 1
    # scale = 最大绝对值 / 量化上限；clamp 防止全 0 张量导致除零
    scale = x.abs().max().clamp(min=1e-8) / qmax
    # round 到整数并截断到合法区间，存为 int8
    q = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int8)
    # 反量化回浮点
    x_hat = q.float() * scale
    return x_hat, scale, q


# 百分位截断版对称量化（per-tensor + clipping）
# 思路：用绝对值的某个百分位（如 99.9%）代替 max 作为量化上限，
# 主动“牺牲”极少数离群值，换取绝大多数正常值更小的量化步长（scale）。
def symmetric_quant_clip(x: torch.Tensor, bits=8, percentile=99.9):
    qmax = 2 ** (bits - 1) - 1
    # 用百分位阈值替代 max，避免 outlier 撑大 scale
    thresh = torch.quantile(x.abs(), percentile / 100.0).clamp(min=1e-8)
    scale = thresh / qmax
    q = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int8)
    x_hat = q.float() * scale
    return x_hat, scale, q


# 逐通道对称量化（per-channel）
# 每个通道（这里沿 dim 划分）独立计算 scale。
# 离群值被“隔离”在它所在的那一个通道里，不会污染其他通道的量化精度。
def per_channel_symmetric(x: torch.Tensor, bits=8, dim=0):
    qmax = 2 ** (bits - 1) - 1
    # 对“非 dim”维度求每个通道的绝对值最大，得到每通道独立 scale
    other_dims = [d for d in range(x.dim()) if d != dim]
    scale = x.abs().amax(dim=other_dims, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int8)
    x_hat = q.float() * scale
    return x_hat, scale, q


# 信号量化噪声比（SQNR，单位 dB），越大表示量化误差相对信号越小、精度越高
def sqnr_db(x, x_hat):
    signal = torch.mean(x.float() ** 2)
    noise = torch.mean((x.float() - x_hat.float()) ** 2).clamp(min=1e-12)
    return 10 * torch.log10(signal / noise)


# ---------------- 对比实验：一个 outlier 如何摧毁 per-tensor 量化 ----------------
# 构造 1024 个标准差为 0.1 的“正常权重”，再人为塞入一个 8.0 的离群值。
# 正常值的量级约 0.1，而 outlier 是它的 ~80 倍，会把 per-tensor 的 scale 拉爆。
w = torch.randn(1024) * 0.1
w[0] = 8.0  # outlier（离群值）

# 用一个布尔掩码区分“离群值”和“正常值”，以便单独评估正常值的量化质量
mask_bulk = torch.ones_like(w, dtype=torch.bool)
mask_bulk[0] = False  # 0 号位是 outlier，其余为正常值（bulk）


# 统一打印工具：同时报告“整体 SQNR”和“仅正常值(bulk) SQNR”
# 关键观察点在 bulk SQNR —— 它代表绝大多数权重的真实量化精度。
def report(name, x, x_hat, scale):
    s = float(scale.mean()) if torch.is_tensor(scale) else float(scale)
    print(f"[{name}]")
    print(f"  scale(mean)      = {s:.6f}")
    print(f"  SQNR 整体(dB)     = {float(sqnr_db(x, x_hat)):.2f}")
    print(f"  SQNR 仅正常值(dB) = {float(sqnr_db(x[mask_bulk], x_hat[mask_bulk])):.2f}")


# 方案 A：朴素 per-tensor。scale 被 outlier 撑大，正常值被压进极少数量化格 → bulk 精度崩塌
w_hat_a, scale_a, _ = symmetric_quant_dequant(w, bits=8)
report("A. per-tensor (max)", w, w_hat_a, scale_a)

# 方案 B：per-tensor + 99.9% 百分位截断。outlier 被截断（自身误差变大），但正常值 scale 大幅减小 → bulk 精度显著回升
w_hat_b, scale_b, _ = symmetric_quant_clip(w, bits=8, percentile=99.9)
report("B. per-tensor + clip 99.9%", w, w_hat_b, scale_b)

# 方案 C：per-channel。把 1024 个权重重排成 8 通道 x 128，outlier 被隔离到 0 号通道
# 其余 7 个通道拿到各自的小 scale → 这些通道的量化精度接近“无 outlier”的理想情况
w2d = w.view(8, 128)
w_hat_c2d, scale_c, _ = per_channel_symmetric(w2d, bits=8, dim=0)
w_hat_c = w_hat_c2d.reshape(-1)
report("C. per-channel (8 x 128)", w, w_hat_c, scale_c)

print()
print("结论：单个 outlier 会通过 max-scale 污染整张量的量化精度；")
print(
    "     截断(clip) 和 逐通道(per-channel) 都能把损失限制在局部，是工业界常用的抗 outlier 手段。"
)
