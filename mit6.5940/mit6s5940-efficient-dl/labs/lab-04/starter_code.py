"""
实验 4：LLM 量化实验 (AWQ) - 起始代码
学生需要完成所有标记为 TODO 的部分

本实验模拟 AWQ (Activation-aware Weight Quantization) 的核心思想：
1. 伪量化 (Pseudo-quantization)
2. 显著性通道识别
3. 显著性通道保护 (FP16)
4. 缩放操作 (Scale-up/Scale-down)
5. 自动缩放搜索
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import copy
from typing import Tuple, List, Dict
import time


# ============ 设备配置 ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ============ 微型 Transformer 模型 ============
class SimpleAttention(nn.Module):
    """简化的单头注意力层，用于模拟 LLM 中的线性层量化"""

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
    """简化的前馈网络，模拟 LLM 中的 FFN 层"""

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
    """微型 Transformer 块，模拟 LLM 中的一个解码器层"""

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
def collect_activation_stats(model: nn.Module, calibration_data: torch.Tensor):
    """
    收集模型各层的激活值统计信息

    通过前向钩子记录每个 Linear 层输出的 L2 范数，
    用于后续识别显著性通道

    参数:
        model: 待分析的模型
        calibration_data: 校准数据 (B, T)，整数 token IDs

    返回:
        activation_stats: 字典 {layer_name: activation_magnitudes}
    """
    activation_stats = {}

    # TODO: 实现激活值统计收集
    # 1. 为每个 Linear 层注册前向钩子
    # 2. 运行校准数据的前向传播
    # 3. 记录每层输出的按通道平均的 L2 范数
    # 4. 返回统计信息

    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            # output shape: (B, T, out_features) 或 (B, out_features)
            # 计算每个输出通道的激活幅度（L2 范数）
            if output.dim() == 3:  # (B, T, D)
                # 沿 batch 和 seq 维度取平均的 L2 范数
                act_mag = output.norm(dim=(0, 1))
            elif output.dim() == 2:  # (B, D)
                act_mag = output.norm(dim=0)
            else:
                act_mag = output.abs().mean(dim=0)

            if name not in activation_stats:
                activation_stats[name] = act_mag
            else:
                activation_stats[name] += act_mag

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # 运行校准
    model.eval()
    with torch.no_grad():
        batch_size = min(32, calibration_data.size(0))
        for i in range(0, calibration_data.size(0), batch_size):
            batch = calibration_data[i : i + batch_size].to(device)
            _ = model(batch)

    for h in hooks:
        h.remove()

    return activation_stats


# ============ TODO 1: 实现伪量化 ============
def pseudo_quantize_weight(
    weight: torch.Tensor, bits: int = 4, group_size: int = 128
) -> torch.Tensor:
    """
    伪量化权重张量（per-group 量化）

    AWQ 使用 per-group (per-channel group) 量化：
    将权重分成大小为 group_size 的组，每组独立计算量化参数

    参数:
        weight: FP16 权重张量，形状为 (out_features, in_features)
        bits: 量化位宽
        group_size: 每组的大小（沿 in_features 维度）

    返回:
        quantized_weight: 伪量化后的权重（FP16 格式，但值被量化到有限集合）
    """
    # TODO: 实现 per-group 伪量化
    # 步骤：
    # 1. 获取 out_features, in_features 维度
    # 2. 对每个 out_channel，沿 in_features 维度按 group_size 分组
    # 3. 每组内计算 min/max，确定量化 scale
    # 4. 量化-反量化该组权重
    # 5. 返回所有组拼接的结果

    out_features, in_features = weight.shape
    quantized = torch.zeros_like(weight)

    # 计算量化范围
    qmax = 2 ** (bits - 1) - 1

    pass  # TODO: 完成实现


# ============ TODO 2: 识别显著性通道 ============
def identify_salient_channels(
    activation_stats: Dict[str, torch.Tensor],
    linear_layers: Dict[str, nn.Linear],
    top_k_ratio: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """
    根据激活值统计识别显著性通道

    显著性通道 = 激活值幅度最大的通道，对模型输出影响最大，
    量化时需要特殊保护

    参数:
        activation_stats: {layer_name: activation_magnitudes}
        linear_layers: {layer_name: nn.Linear module}
        top_k_ratio: 保留为显著通道的比例

    返回:
        salient_masks: {layer_name: boolean mask} 标记每层中哪些通道是显著的
    """
    # TODO: 实现显著性通道识别
    # 步骤：
    # 1. 对每层的激活幅度进行排序
    # 2. 选择 top_k_ratio 比例的通道作为显著性通道
    # 3. 为每层创建布尔掩码
    # 4. 返回掩码字典

    salient_masks = {}

    for name, act_mag in activation_stats.items():
        if name not in linear_layers:
            continue

        # TODO: 选择激活幅度最大的前 k 个通道
        k = max(1, int(len(act_mag) * top_k_ratio))
        pass

    return salient_masks


# ============ TODO 3: 保护显著性通道 ============
def quantize_with_salient_protection(
    weight: torch.Tensor,
    salient_mask: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
) -> torch.Tensor:
    """
    对权重进行量化，但保护显著性通道（保持在 FP16）

    参数:
        weight: FP16 权重张量 (out_features, in_features)
        salient_mask: 布尔掩码 (out_features,)，True 表示显著性通道
        bits: 量化位宽
        group_size: 分组大小

    返回:
        quantized_weight: 量化后的权重（显著性通道保持 FP16）
    """
    # TODO: 实现保护性量化
    # 步骤：
    # 1. 对非显著性通道执行伪量化
    # 2. 显著性通道保持原始 FP16 值不变
    # 3. 返回混合精度权重

    quantized = weight.clone()

    # 对非显著性通道进行量化
    nonsalient_indices = torch.where(~salient_mask)[0]
    if len(nonsalient_indices) > 0:
        pass  # TODO: 对非显著性通道量化

    return quantized


# ============ TODO 4: 实现缩放操作 ============
def scale_weight_and_activation(
    weight: torch.Tensor, scale_factors: torch.Tensor, in_features: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对权重进行通道级缩放，并生成对应的激活缩放因子

    核心思想：对权重放大 α，对上一层的输出相应缩小 1/α
    线性层的数学等效性保证输出不变

    参数:
        weight: 权重张量 (out_features, in_features)
        scale_factors: 缩放因子 (out_features,) 或 (in_features,)
        in_features: 输入特征数

    返回:
        scaled_weight: 缩放后的权重
        activation_scale: 需要对前一层激活应用的缩放因子
    """
    # TODO: 实现缩放操作
    # 步骤：
    # 1. 对权重沿 out_features 维度乘以 scale_factors
    # 2. 激活缩放因子应为 scale_factors 的倒数（应用于前一层输出）
    # 3. 返回缩放后的权重和激活缩放因子

    pass


# ============ TODO 5: 自动缩放搜索 ============
def auto_scale_search(
    weight: torch.Tensor,
    act_magnitudes: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    n_grid: int = 20,
    alpha_range: Tuple[float, float] = (0.5, 2.0),
):
    """
    自动搜索最优的通道级缩放因子

    AWQ 的核心思想：通过缩放将量化难度从显著通道转移到非显著通道

    参数:
        weight: 权重张量 (out_features, in_features)
        act_magnitudes: 激活值幅度 (out_features,)
        bits: 量化位宽
        group_size: 分组大小
        n_grid: 搜索网格大小
        alpha_range: 缩放因子搜索范围

    返回:
        best_scales: 最优缩放因子 (out_features,)
    """
    # TODO: 实现自动缩放搜索
    # 步骤：
    # 1. 为每个输出通道初始化缩放候选值（在 alpha_range 范围内均匀采样）
    # 2. 对每个通道，尝试不同的缩放因子
    # 3. 对缩放后的权重进行伪量化
    # 4. 计算量化误差（原始权重与量化后权重的 MSE）
    # 5. 选择使量化误差最小的缩放因子
    # 6. 返回所有通道的最优缩放因子

    out_features = weight.shape[0]
    best_scales = torch.ones(out_features, device=weight.device)

    # 均匀采样候选缩放因子
    candidates = torch.linspace(alpha_range[0], alpha_range[1], n_grid).to(
        weight.device
    )

    for ch in range(out_features):
        pass  # TODO: 为每个通道搜索最优缩放因子

    return best_scales


# ============ 评估函数 ============
def compute_perplexity(model: nn.Module, data: torch.Tensor):
    """
    计算模型在给定数据上的困惑度（简化的交叉熵）

    参数:
        model: 语言模型
        data: token IDs (B, T)

    返回:
        perplexity: 困惑度
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        batch_size = 8
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size].to(device)
            logits = model(batch[:, :-1])
            targets = batch[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity


# ============ 量化模型构建 ============
def apply_awq_to_model(
    model: nn.Module,
    activation_stats: dict,
    bits: int = 4,
    group_size: int = 128,
    top_k_ratio: float = 0.01,
):
    """
    对模型应用 AWQ 量化

    完整流程：
    1. 收集激活值统计
    2. 识别显著性通道
    3. 搜索缩放因子
    4. 应用缩放
    5. 量化非显著性通道
    """
    # TODO: 串联所有步骤，完成 AWQ 量化
    pass


# ============ 生成校准数据 ============
def generate_calibration_data(
    vocab_size: int = 1000, seq_len: int = 128, num_sequences: int = 64
):
    """
    生成模拟的校准数据（在实际 AWQ 中使用 WikiText-2 等数据集）
    """
    return torch.randint(0, vocab_size, (num_sequences, seq_len))


# ============ 主程序 ============
if __name__ == "__main__":
    print("=" * 60)
    print("实验 4：LLM 量化实验 (AWQ)")
    print("=" * 60)

    # 超参数
    VOCAB_SIZE = 1000
    D_MODEL = 64
    D_FF = 256
    NUM_LAYERS = 4
    SEQ_LEN = 64

    # 1. 创建微型 LLM
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

    # 2. 生成校准数据
    print("\n[步骤 2] 生成校准数据...")
    calib_data = generate_calibration_data(
        vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, num_sequences=32
    )

    # 3. 评估基线困惑度
    print("\n[步骤 3] 评估 FP32 基线困惑度...")
    # TODO: 计算并打印基线困惑度

    # 4. 收集激活值统计
    print("\n[步骤 4] 收集激活值统计...")
    # TODO: 调用 collect_activation_stats

    # 5. 识别显著性通道
    print("\n[步骤 5] 识别显著性通道...")
    # TODO: 调用 identify_salient_channels

    # 6. 测试伪量化
    print("\n[步骤 6] 测试伪量化...")
    # TODO: 对模型权重进行伪量化，比较困惑度变化

    # 7. 测试显著性通道保护
    print("\n[步骤 7] 测试显著性通道保护...")
    # TODO: 实现并测试保护性量化

    # 8. 自动缩放搜索
    print("\n[步骤 8] 自动缩放搜索...")
    # TODO: 实现并测试自动缩放搜索

    # 9. 汇总报告
    print("\n[步骤 9] 汇总实验报告...")
    # TODO: 打印困惑度对比表，包括：
    # FP32 基线、Naive INT4、AWQ (1% salient)、AWQ (no scale search)、
    # AWQ (with scale search)

    print("\n实验完成！请将结果填入 report_template.md。")
