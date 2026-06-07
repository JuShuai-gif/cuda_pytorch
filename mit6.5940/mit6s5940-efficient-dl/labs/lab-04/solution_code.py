"""
实验 4：LLM 量化实验 (AWQ) - 完整参考实现
实现 AWQ 的核心组件：伪量化、显著性通道识别、缩放搜索

所有注释和文档均使用中文
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import copy
from typing import Tuple, List, Dict, Optional
import time
import matplotlib.pyplot as plt

# ============ 设备配置 ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============ 微型 Transformer 模型 ============
class SimpleAttention(nn.Module):
    """简化的单头注意力层"""

    def __init__(self, d_model: int = 64, d_head: int = 64):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_head, bias=False)
        self.k_proj = nn.Linear(d_model, d_head, bias=False)
        self.v_proj = nn.Linear(d_model, d_head, bias=False)
        self.o_proj = nn.Linear(d_head, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scale = math.sqrt(D)
        attn = (q @ k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        return self.o_proj(out)


class SimpleFFN(nn.Module):
    """简化的 FFN（SwiGLU 风格）"""

    def __init__(self, d_model: int = 64, d_ff: int = 256):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class MiniLLMBlock(nn.Module):
    """微型 Transformer 块"""

    def __init__(self, d_model: int = 64, d_ff: int = 256):
        super().__init__()
        self.attn = SimpleAttention(d_model, d_model)
        self.ffn = SimpleFFN(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniLLM(nn.Module):
    """微型大语言模型，用于量化实验"""

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 64,
        d_ff: int = 256,
        num_layers: int = 4,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        self.layers = nn.ModuleList(
            [MiniLLMBlock(d_model, d_ff) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.token_embed(input_ids) + self.pos_embed[:, :T, :]
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.lm_head(x)


# ============ 激活值收集 ============
def collect_activation_stats(
    model: nn.Module, calibration_data: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    收集模型各层 Linear 输出的激活值幅度

    利用前向钩子记录每个 Linear 层输出的按通道 L2 范数，
    取多个校准样本的平均值
    """
    activation_stats = {}
    sample_count = {}

    def make_hook(name):
        def hook(module, input, output):
            if output.dim() >= 2:
                # (B, T, D) 或 (B, D) -> 沿 batch/seq 维度取 L2 范数
                if output.dim() == 3:
                    act_mag = output.norm(dim=(0, 1))
                elif output.dim() == 2:
                    act_mag = output.norm(dim=0)
                else:
                    return

                if name not in activation_stats:
                    activation_stats[name] = act_mag.detach().cpu()
                    sample_count[name] = 1
                else:
                    activation_stats[name] += act_mag.detach().cpu()
                    sample_count[name] += 1

        return hook

    # 注册钩子
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # 运行校准数据
    model.eval()
    with torch.no_grad():
        batch_size = 8
        for i in range(0, calibration_data.size(0), batch_size):
            batch = calibration_data[i : i + batch_size].to(device)
            _ = model(batch)

    # 移除钩子
    for h in hooks:
        h.remove()

    # 取平均
    for name in activation_stats:
        activation_stats[name] /= sample_count[name]

    return activation_stats


# ============ 伪量化实现 ============
def pseudo_quantize_weight(
    weight: torch.Tensor, bits: int = 4, group_size: int = 128
) -> torch.Tensor:
    """
    Per-group 伪量化

    将权重按 group_size 沿 in_features 维度分组，
    每组独立计算 min/max 和量化参数

    参数:
        weight: (out_features, in_features)
        bits: 量化位宽
        group_size: 分组大小

    返回:
        quantized: 伪量化后的权重（FP32 格式，值被量化到离散集合）
    """
    out_feat, in_feat = weight.shape
    result = torch.zeros_like(weight)

    qmax = 2 ** (bits - 1) - 1  # 有符号整数最大值

    for oc in range(out_feat):
        for g_start in range(0, in_feat, group_size):
            g_end = min(g_start + group_size, in_feat)
            group = weight[oc, g_start:g_end]

            # 计算 scale（对称量化）
            max_abs = group.abs().max().item()
            if max_abs < 1e-8:
                scale = 1.0
            else:
                scale = max_abs / qmax

            # 量化 + 反量化
            q_val = torch.clamp(torch.round(group / scale), -qmax, qmax)
            result[oc, g_start:g_end] = q_val * scale

    return result


# ============ 显著性通道识别 ============
def identify_salient_channels(
    activation_stats: Dict[str, torch.Tensor],
    linear_layers: Dict[str, nn.Linear],
    top_k_ratio: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """
    根据激活值幅度识别显著性通道

    对每个 Linear 层的输出通道，计算其激活幅度（L2 范数），
    选择 top_k_ratio 比例的最大幅度通道作为显著性通道

    参数:
        activation_stats: {layer_name: activation_magnitudes}
        linear_layers: {layer_name: nn.Linear module}
        top_k_ratio: 显著性通道的比例

    返回:
        salient_masks: {layer_name: boolean mask (out_features,)}
    """
    salient_masks = {}

    # 收集所有可用的 Linear 层
    linear_dict = {name: mod for name, mod in linear_layers.items()}

    for name, act_mag in activation_stats.items():
        if name not in linear_dict:
            continue

        out_features = linear_dict[name].out_features
        # 确保激活统计维度与输出特征数一致
        if len(act_mag) != out_features:
            continue

        k = max(1, int(out_features * top_k_ratio))
        # 选择幅度最大的 k 个通道
        _, top_indices = torch.topk(act_mag, k)
        mask = torch.zeros(out_features, dtype=torch.bool)
        mask[top_indices] = True
        salient_masks[name] = mask

    return salient_masks


# ============ 保护性量化 ============
def quantize_with_salient_protection(
    weight: torch.Tensor,
    salient_mask: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
) -> torch.Tensor:
    """
    对非显著性通道进行量化，显著性通道保持 FP16/FP32

    参数:
        weight: (out_features, in_features)
        salient_mask: (out_features,) 布尔掩码
        bits: 量化位宽
        group_size: 分组大小

    返回:
        quantized: 混合精度权重
    """
    quantized = weight.clone()
    nonsalient_idx = torch.where(~salient_mask)[0]

    if len(nonsalient_idx) > 0:
        # 仅对非显著性通道进行量化
        for oc in nonsalient_idx:
            oc_int = oc.item()
            row = quantized[oc_int]
            quantized[oc_int] = pseudo_quantize_weight(
                row.unsqueeze(0), bits, group_size
            ).squeeze(0)

    # 显著性通道（salient_mask == True）保持原始值
    return quantized


# ============ 缩放操作 ============
def scale_weight_and_activation(
    weight: torch.Tensor,
    scale_factors: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对权重进行通道级缩放

    对第 i 个输出通道的权重乘以 scale_factors[i]，
    返回缩放后的权重和对应的激活缩放因子（1/scale）

    参数:
        weight: (out_features, in_features)
        scale_factors: (out_features,)

    返回:
        scaled_weight: 缩放后的权重
        activation_inv_scale: 激活缩放因子 (out_features,)
    """
    scaled_weight = weight * scale_factors.unsqueeze(1)
    activation_inv_scale = 1.0 / scale_factors
    return scaled_weight, activation_inv_scale


# ============ 自动缩放搜索 ============
def auto_scale_search(
    weight: torch.Tensor,
    act_magnitudes: torch.Tensor,
    salient_mask: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    n_grid: int = 20,
    alpha_range: Tuple[float, float] = (0.5, 2.0),
) -> torch.Tensor:
    """
    自动搜索最优的通道级缩放因子

    AWQ 核心思想：通过缩放将量化难度从显著通道转移，
    只对显著性通道进行搜索

    对每个显著性通道，尝试不同的缩放因子 α，
    计算缩放后权重的量化误差，选择误差最小的 α
    """
    out_features = weight.shape[0]
    best_scales = torch.ones(out_features, device=weight.device)

    # 均匀采样候选缩放因子
    candidates = torch.linspace(
        alpha_range[0], alpha_range[1], n_grid, device=weight.device
    )

    salient_idx = torch.where(salient_mask)[0]

    if len(salient_idx) == 0:
        return best_scales

    for ch in salient_idx:
        ch_int = ch.item()
        original_row = weight[ch_int].clone()

        best_error = float("inf")
        best_alpha = 1.0

        for alpha in candidates:
            alpha_val = alpha.item()
            # 缩放权重
            scaled_row = original_row * alpha_val
            # 伪量化
            q_row = pseudo_quantize_weight(
                scaled_row.unsqueeze(0), bits, group_size
            ).squeeze(0)
            # 反缩放（模拟量化后再除以 α）
            recovered_row = q_row / alpha_val
            # 计算与原始权重的误差
            error = F.mse_loss(recovered_row, original_row).item()

            if error < best_error:
                best_error = error
                best_alpha = alpha_val

        best_scales[ch_int] = best_alpha

    return best_scales


# ============ 困惑度计算 ============
def compute_perplexity(model: nn.Module, data: torch.Tensor) -> float:
    """计算模型在数据上的困惑度（越低越好）"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        batch_size = 8
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size].to(device)
            logits = model(batch[:, :-1])  # (B, T-1, V)
            targets = batch[:, 1:]  # (B, T-1)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


# ============ AWQ 完整流程 ============
def apply_awq_to_model(
    model: nn.Module,
    activation_stats: Dict[str, torch.Tensor],
    calib_data: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    top_k_ratio: float = 0.01,
    use_scale_search: bool = True,
) -> nn.Module:
    """
    对模型应用完整的 AWQ 量化流程

    流程：
    1. 识别显著性通道
    2. （可选）搜索缩放因子并应用缩放
    3. 量化非显著性通道
    """
    # 构建 layer_name -> module 的映射
    linear_layers = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            linear_layers[name] = mod

    # 识别显著性通道
    print("  识别显著性通道...")
    salient_masks = identify_salient_channels(
        activation_stats, linear_layers, top_k_ratio
    )
    total_salient = sum(mask.sum().item() for mask in salient_masks.values())
    print(f"  显著性通道总数: {total_salient}")

    # 对每个 Linear 层应用 AWQ
    q_model = copy.deepcopy(model)

    for name, module in q_model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name not in salient_masks:
            continue

        weight = module.weight.data
        mask = salient_masks[name].to(weight.device)

        # 自动缩放搜索
        if use_scale_search and name in activation_stats:
            scales = auto_scale_search(
                weight, activation_stats[name].to(weight.device), mask, bits, group_size
            )
            scaled_w, _ = scale_weight_and_activation(weight, scales)
            # 对缩放后的权重进行保护性量化
            module.weight.data = quantize_with_salient_protection(
                scaled_w, mask, bits, group_size
            )
        else:
            # 直接保护性量化
            module.weight.data = quantize_with_salient_protection(
                weight, mask, bits, group_size
            )

    return q_model


# ============ 生成校准和评估数据 ============
def generate_data(
    vocab_size: int = 1000, seq_len: int = 64, num_sequences: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成模拟的校准和评估数据

    在实际 AWQ 中使用 WikiText-2 或 C4 数据集
    """
    all_data = torch.randint(0, vocab_size, (num_sequences, seq_len))
    split = num_sequences // 2
    calib_data = all_data[:split]
    eval_data = all_data[split:]
    return calib_data, eval_data


# ============ 绘图函数 ============
def plot_perplexity_comparison(results: Dict[str, float]):
    """绘制各方法的困惑度对比条形图"""
    plt.figure(figsize=(10, 5))
    methods = list(results.keys())
    perplexities = list(results.values())

    colors = ["#2ecc71", "#e74c3c", "#f39c12", "#3498db", "#9b59b6"]
    bars = plt.bar(range(len(methods)), perplexities, color=colors[: len(methods)])

    plt.xticks(range(len(methods)), methods, rotation=20, ha="right")
    plt.ylabel("困惑度")
    plt.title("不同量化方法的困惑度对比")

    for bar, ppl in zip(bars, perplexities):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{ppl:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig("awq_perplexity_comparison.png", dpi=150)
    print("困惑度对比图已保存为 awq_perplexity_comparison.png")


def plot_activation_distribution(activation_stats: Dict[str, torch.Tensor]):
    """可视化部分层的激活幅度分布"""
    # 选择前 6 个层进行可视化
    layers_to_plot = list(activation_stats.keys())[:6]
    n_layers = len(layers_to_plot)
    if n_layers == 0:
        return

    cols = 3
    rows = (n_layers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3 * rows))
    axes = axes.flatten() if n_layers > 1 else [axes]

    for idx, name in enumerate(layers_to_plot):
        ax = axes[idx]
        act_mag = activation_stats[name].numpy()
        ax.bar(range(len(act_mag)), act_mag, alpha=0.7)
        ax.set_title(name[:30], fontsize=9)
        ax.set_xlabel("通道索引")
        ax.set_ylabel("激活幅度")

    # 隐藏多余的子图
    for idx in range(n_layers, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("各层激活幅度分布")
    plt.tight_layout()
    plt.savefig("activation_distribution.png", dpi=150)
    print("激活分布图已保存为 activation_distribution.png")


# ============ 主程序 ============
if __name__ == "__main__":
    print("=" * 60)
    print("实验 4：LLM 量化实验 (AWQ) - 完整实现")
    print("=" * 60)

    # 超参数
    VOCAB_SIZE = 1000
    D_MODEL = 64
    D_FF = 256
    NUM_LAYERS = 4
    SEQ_LEN = 64

    # 1. 创建模型和数据
    print("\n[步骤 1] 创建微型 LLM...")
    model = MiniLLM(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        max_seq_len=SEQ_LEN,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")

    print("\n[步骤 2] 生成数据...")
    calib_data, eval_data = generate_data(
        vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, num_sequences=64
    )

    # 2. 评估 FP32 基线
    print("\n[步骤 3] 评估 FP32 基线困惑度...")
    fp32_ppl = compute_perplexity(model, eval_data)
    print(f"  FP32 基线困惑度: {fp32_ppl:.2f}")

    # 3. 收集激活值统计
    print("\n[步骤 4] 收集激活值统计...")
    act_stats = collect_activation_stats(model, calib_data)
    print(f"  收集到 {len(act_stats)} 层的激活统计")

    # 显示统计摘要
    print("  激活幅度统计 (前 5 层):")
    for i, (name, mag) in enumerate(act_stats.items()):
        if i >= 5:
            break
        print(
            f"    {name}: mean={mag.mean():.4f}, max={mag.max():.4f}, "
            f"min={mag.min():.4f}"
        )

    # 4. Naive INT4 量化
    print("\n[步骤 5] Naive INT4 量化...")
    naive_q_model = copy.deepcopy(model)
    for name, module in naive_q_model.named_modules():
        if isinstance(module, nn.Linear):
            module.weight.data = pseudo_quantize_weight(
                module.weight.data, bits=4, group_size=128
            )
    naive_ppl = compute_perplexity(naive_q_model, eval_data)
    print(f"  Naive INT4 困惑度: {naive_ppl:.2f} (退化: {naive_ppl - fp32_ppl:.2f})")

    # 5. AWQ 不同显著性比例
    print("\n[步骤 6] AWQ 量化（不同显著性比例）...")
    salient_ratios = [0.005, 0.01, 0.02, 0.05]
    awq_results = {}

    for ratio in salient_ratios:
        print(f"  显著性比例: {ratio * 100:.1f}%")
        q_model = apply_awq_to_model(
            model,
            act_stats,
            calib_data,
            bits=4,
            group_size=128,
            top_k_ratio=ratio,
            use_scale_search=False,
        )
        ppl = compute_perplexity(q_model, eval_data)
        awq_results[f"AWQ ({ratio * 100:.1f}%)"] = ppl
        print(f"    困惑度: {ppl:.2f} (退化: {ppl - fp32_ppl:.2f})")

    # 6. AWQ + 缩放搜索
    print("\n[步骤 7] AWQ + 缩放搜索...")
    q_model_ss = apply_awq_to_model(
        model,
        act_stats,
        calib_data,
        bits=4,
        group_size=128,
        top_k_ratio=0.01,
        use_scale_search=True,
    )
    awq_ss_ppl = compute_perplexity(q_model_ss, eval_data)
    awq_results["AWQ (1% + scale)"] = awq_ss_ppl
    print(
        f"  AWQ (1% + scale search) 困惑度: {awq_ss_ppl:.2f} "
        f"(退化: {awq_ss_ppl - fp32_ppl:.2f})"
    )

    # 7. 汇总报告
    print("\n" + "=" * 60)
    print("实验报告汇总")
    print("=" * 60)

    all_results = {"FP32 基线": fp32_ppl, "Naive INT4": naive_ppl}
    all_results.update(awq_results)

    print(f"\n{'方法':<25} {'困惑度':<10} {'退化':<10}")
    print("-" * 45)
    for method, ppl in all_results.items():
        degradation = ppl - fp32_ppl
        print(f"{method:<25} {ppl:<10.2f} {degradation:<10.2f}")

    # 8. 消融分析
    print(f"\n  消融分析:")
    print(f"    显著性保护贡献: {(naive_ppl - awq_results['AWQ (1.0%)']):.2f} ppl")
    print(f"    缩放搜索贡献: {(awq_results['AWQ (1.0%)'] - awq_ss_ppl):.2f} ppl")
    print(f"    总改进: {(naive_ppl - awq_ss_ppl):.2f} ppl")

    # 9. 绘图
    print("\n[步骤 8] 绘制结果...")
    plot_perplexity_comparison(all_results)
    plot_activation_distribution(act_stats)

    print("\n实验完成！")
