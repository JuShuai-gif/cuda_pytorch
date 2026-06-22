"""
实验 5：LLM 边缘部署实验 - 完整参考实现
模拟 ONNX 导出、INT8 量化部署、CPU 基准测试和部署报告生成

所有注释和文档均使用中文
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import copy
import json
import os
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, field, asdict
import matplotlib.pyplot as plt


# ============ 设备配置 ============
device = torch.device("cpu")  # 强制使用 CPU 模拟边缘设备
# 模拟低功耗边缘设备的 CPU 限制
torch.set_num_threads(2)
print(f"使用设备: {device}（模拟边缘设备 CPU，限制为 {torch.get_num_threads()} 线程）")

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============ 数据类 ============
@dataclass
class DeploymentMetrics:
    """部署指标数据类"""

    model_name: str = ""
    precision: str = "FP32"
    model_size_mb: float = 0.0
    params_count: int = 0
    latency_ms: float = 0.0
    throughput_qps: float = 0.0
    accuracy: float = 0.0
    compression_ratio: float = 1.0
    speedup_ratio: float = 1.0
    memory_peak_mb: float = 0.0
    notes: str = ""


# ============ 边缘 Transformer 模型 ============
class EdgeTransformer(nn.Module):
    """
    适合边缘部署的微型 Transformer

    设计原则：
    - 参数量小：< 200K 参数
    - 浅层：只需 2 层
    - 小隐藏维度：适合 INT8 量化
    - 无复杂操作：仅使用标准 PyTorch 算子
    """

    def __init__(
        self,
        vocab_size: int = 512,
        d_model: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_layers = num_layers

        # 词嵌入 + 位置嵌入
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        # QKV 投影（合并为单个矩阵以提高计算效率）
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # FFN
        self.ffn1 = nn.Linear(d_model, 4 * d_model, bias=False)
        self.ffn2 = nn.Linear(4 * d_model, d_model, bias=False)

        # LayerNorm
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # 输出投影
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def _transformer_block(self, x):
        """单层 Transformer 解码器块"""
        B, T, D = x.shape

        # === 自注意力 ===
        residual = x
        x_norm = self.ln1(x)
        qkv = self.qkv_proj(x_norm)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 缩放点积注意力
        scale = math.sqrt(self.head_dim)
        attn_weights = (q @ k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = attn_weights @ v  # (B, H, T, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B, T, D)
        attn_output = self.out_proj(attn_output)
        x = residual + attn_output

        # === FFN ===
        residual = x
        x_norm = self.ln2(x)
        ffn_hidden = F.relu(self.ffn1(x_norm))
        ffn_output = self.ffn2(ffn_hidden)
        x = residual + ffn_output

        return x

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.token_embed(input_ids) + self.pos_embed[:, :T, :]

        for _ in range(self.num_layers):
            x = self._transformer_block(x)

        return self.lm_head(x)


# ============ ONNX 导出模拟 ============
def simulate_onnx_export(
    model: nn.Module, input_shape: Tuple, export_path: str = "model.onnx"
) -> Dict:
    """
    模拟 ONNX 导出流程

    在实际部署中，使用 torch.onnx.export()。
    本函数模拟该过程，提取模型的计算图信息和元数据
    """
    model_info = {
        "framework": "PyTorch",
        "export_format": "ONNX (simulated)",
        "input_shape": list(input_shape),
        "output_shape": [],
        "layers": [],
        "total_params": 0,
        "model_size_bytes": 0,
        "export_path": export_path,
        "onnx_opset": 17,
        "dynamic_axes": False,
    }

    # 遍历所有层，收集信息
    layer_stats = {}
    for name, module in model.named_modules():
        if name == "":
            continue
        layer_type = module.__class__.__name__
        params = sum(p.numel() for p in module.parameters())
        model_info["total_params"] += params

        if layer_type not in layer_stats:
            layer_stats[layer_type] = {"count": 0, "params": 0}
        layer_stats[layer_type]["count"] += 1
        layer_stats[layer_type]["params"] += params

    # 汇总层信息
    for ltype, stats in layer_stats.items():
        model_info["layers"].append(
            {
                "type": ltype,
                "count": stats["count"],
                "params": stats["params"],
            }
        )

    # 计算模型大小（FP32 参数）
    model_info["model_size_bytes"] = model_info["total_params"] * 4
    model_info["model_size_mb"] = model_info["model_size_bytes"] / (1024 * 1024)

    # 模拟计算输出形状
    dummy = torch.randint(0, 100, (1, input_shape[0]))
    model.eval()
    with torch.no_grad():
        out = model(dummy)
    model_info["output_shape"] = list(out.shape)

    print(
        f"  模型分析完成：{len(model_info['layers'])} 种层类型, "
        f"{model_info['total_params']:,} 参数, "
        f"{model_info['model_size_mb']:.2f} MB (FP32)"
    )

    return model_info


# ============ INT8 量化部署 ============
def quantize_model_to_int8(model: nn.Module) -> nn.Module:
    """
    将 FP32 模型量化为 INT8

    实现逐通道对称量化：
    - 对每个 nn.Linear 层的每个输出通道独立计算 scale
    - nn.Embedding 和 nn.LayerNorm 保持 FP32
    """
    q_model = copy.deepcopy(model)

    for name, module in q_model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data  # (out_features, in_features)

            # 逐通道 INT8 对称量化
            deq_weight = torch.zeros_like(weight)
            for out_ch in range(weight.shape[0]):
                ch_weight = weight[out_ch]
                max_abs = ch_weight.abs().max().item()
                if max_abs < 1e-8:
                    scale = 1.0
                else:
                    scale = max_abs / 127.0

                # 量化
                q_val = torch.clamp(torch.round(ch_weight / scale), -128, 127)
                # 反量化（模拟推理时的 dequantize）
                deq_weight[out_ch] = q_val.float() * scale

            module.weight.data = deq_weight

        elif isinstance(module, nn.Embedding):
            # Embedding 层保持 FP32（查表操作，量化收益小且损失大）
            pass

        elif isinstance(module, nn.LayerNorm):
            # LayerNorm 参数少，保持 FP32
            pass

    return q_model


def get_model_size_bytes(
    model: nn.Module, weight_bits: int = 32, quantized_layers: set = None
) -> int:
    """
    计算模型的估计存储大小

    参数:
        model: PyTorch 模型
        weight_bits: 默认位宽
        quantized_layers: 已量化的层类型集合

    返回:
        size_bytes: 估计字节数
    """
    if quantized_layers is None:
        quantized_layers = set()

    total_bytes = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if "Linear" in quantized_layers:
                bits = 8
            else:
                bits = weight_bits
            params = sum(p.numel() for p in module.parameters())
            total_bytes += params * (bits // 8)
        elif isinstance(module, nn.Embedding):
            # Embedding 通常保持 FP32
            params = sum(p.numel() for p in module.parameters())
            total_bytes += params * 4
        elif isinstance(module, nn.LayerNorm):
            params = sum(p.numel() for p in module.parameters())
            total_bytes += params * 4
        else:
            params = sum(p.numel() for p in module.parameters())
            total_bytes += params * (weight_bits // 8)

    return total_bytes


# ============ CPU 推理基准测试 ============
def benchmark_inference(
    model: nn.Module,
    input_shape: Tuple,
    num_warmup: int = 20,
    num_runs: int = 100,
    batch_sizes: List[int] = None,
) -> Dict:
    """
    在 CPU 上进行推理基准测试

    测量每个 batch_size 的延迟和吞吐量
    """
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16]

    model.eval()
    results = {}

    for bs in batch_sizes:
        # 生成输入
        seq_len = input_shape[0]
        dummy_input = torch.randint(0, 512, (bs, seq_len)).to(device)

        # 预热
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy_input)

        # 计时
        torch.cpu.synchronize() if hasattr(torch.cpu, "synchronize") else None

        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
        torch.cpu.synchronize() if hasattr(torch.cpu, "synchronize") else None

        end_time = time.time()

        total_time = end_time - start_time
        avg_latency_ms = (total_time / num_runs) * 1000
        throughput_qps = num_runs * bs / total_time

        results[f"batch_{bs}"] = {
            "batch_size": bs,
            "avg_latency_ms": avg_latency_ms,
            "throughput_qps": throughput_qps,
            "total_time_s": total_time,
            "num_runs": num_runs,
        }

        print(
            f"    Batch={bs:>3}: Latency={avg_latency_ms:>8.2f} ms, "
            f"Throughput={throughput_qps:>8.1f} samples/s"
        )

    return results


# ============ FP32 vs INT8 对比 ============
def compare_fp32_int8(
    fp32_model: nn.Module,
    int8_model: nn.Module,
    input_shape: Tuple,
    test_data: torch.Tensor = None,
) -> Dict:
    """
    全面对比 FP32 和 INT8 模型的性能指标
    """
    comparison = {}

    # 模型大小
    fp32_size_bytes = get_model_size_bytes(fp32_model, weight_bits=32)
    int8_size_bytes = get_model_size_bytes(
        int8_model, weight_bits=8, quantized_layers={"Linear"}
    )
    fp32_size_mb = fp32_size_bytes / (1024 * 1024)
    int8_size_mb = int8_size_bytes / (1024 * 1024)
    compression_ratio = fp32_size_bytes / int8_size_bytes

    print(f"\n  模型大小对比:")
    print(f"    FP32: {fp32_size_mb:.2f} MB")
    print(f"    INT8: {int8_size_mb:.2f} MB")
    print(f"    压缩率: {compression_ratio:.2f}×")

    # 推理性能
    print(f"\n  FP32 推理基准测试:")
    fp32_bench = benchmark_inference(fp32_model, input_shape)

    print(f"\n  INT8 推理基准测试:")
    int8_bench = benchmark_inference(int8_model, input_shape)

    # 加速比（以 batch=1 为准）
    speedup = (
        fp32_bench["batch_1"]["avg_latency_ms"]
        / int8_bench["batch_1"]["avg_latency_ms"]
    )

    comparison = {
        "fp32": {
            "model_size_mb": fp32_size_mb,
            "model_size_bytes": fp32_size_bytes,
            "benchmark": fp32_bench,
            "precision": "FP32",
        },
        "int8": {
            "model_size_mb": int8_size_mb,
            "model_size_bytes": int8_size_bytes,
            "benchmark": int8_bench,
            "precision": "INT8",
        },
        "compression_ratio": compression_ratio,
        "speedup_ratio": speedup,
    }

    # 精度损失
    if test_data is not None:
        print(f"\n  评估精度损失...")
        mse = evaluate_accuracy_loss(fp32_model, int8_model, test_data)
        comparison["accuracy_mse"] = mse
        print(f"    FP32 vs INT8 输出 MSE: {mse:.6f}")
    else:
        comparison["accuracy_mse"] = 0.0

    return comparison


# ============ 精度损失评估 ============
def evaluate_accuracy_loss(
    fp32_model: nn.Module, int8_model: nn.Module, test_data: torch.Tensor
) -> float:
    """
    评估量化引入的输出差异
    """
    fp32_model.eval()
    int8_model.eval()

    total_mse = 0.0
    total_max_error = 0.0
    count = 0

    with torch.no_grad():
        batch_size = 8
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i : i + batch_size].to(device)
            fp32_out = fp32_model(batch)
            int8_out = int8_model(batch)

            mse = F.mse_loss(fp32_out, int8_out).item()
            max_err = (fp32_out - int8_out).abs().max().item()

            total_mse += mse
            total_max_error = max(total_max_error, max_err)
            count += 1

    avg_mse = total_mse / count
    return avg_mse


# ============ 部署报告生成 ============
def generate_deployment_report(
    comparison: Dict, model_info: Dict, output_path: str = "deployment_report.json"
) -> str:
    """
    生成完整的部署报告
    """
    fp32 = comparison["fp32"]
    int8 = comparison["int8"]

    fp32_latency = fp32["benchmark"]["batch_1"]["avg_latency_ms"]
    int8_latency = int8["benchmark"]["batch_1"]["avg_latency_ms"]

    lines = []
    lines.append("=" * 65)
    lines.append("            LLM 边缘部署报告")
    lines.append("=" * 65)
    lines.append("")
    lines.append(f"模型: EdgeTransformer")
    lines.append(f"参数量: {model_info['total_params']:,}")
    lines.append(f"输入形状: {model_info['input_shape']}")
    lines.append("")
    lines.append("-" * 65)
    lines.append("  模型大小")
    lines.append("-" * 65)
    lines.append(f"  FP32 模型大小:  {fp32['model_size_mb']:.2f} MB")
    lines.append(f"  INT8 模型大小:  {int8['model_size_mb']:.2f} MB")
    lines.append(f"  压缩率:         {comparison['compression_ratio']:.2f}×")
    lines.append(
        f"  节省空间:       {fp32['model_size_mb'] - int8['model_size_mb']:.2f} MB"
    )
    lines.append("")
    lines.append("-" * 65)
    lines.append("  推理性能 (batch_size=1)")
    lines.append("-" * 65)
    lines.append(f"  FP32 延迟:      {fp32_latency:.2f} ms")
    lines.append(f"  INT8 延迟:      {int8_latency:.2f} ms")
    lines.append(f"  加速比:         {comparison['speedup_ratio']:.2f}×")
    lines.append("")
    lines.append("-" * 65)
    lines.append("  吞吐量对比")
    lines.append("-" * 65)
    for key in fp32["benchmark"]:
        bs = fp32["benchmark"][key]["batch_size"]
        fp32_tp = fp32["benchmark"][key]["throughput_qps"]
        int8_tp = int8["benchmark"][key]["throughput_qps"]
        lines.append(
            f"  Batch={bs:>3}: FP32={fp32_tp:>8.1f} samples/s, "
            f"INT8={int8_tp:>8.1f} samples/s"
        )
    lines.append("")
    lines.append("-" * 65)
    lines.append("  精度评估")
    lines.append("-" * 65)
    lines.append(f"  FP32 vs INT8 输出 MSE: {comparison.get('accuracy_mse', 0):.6f}")
    lines.append("")
    lines.append("-" * 65)
    lines.append("  部署建议")
    lines.append("-" * 65)

    # 自动化建议
    recommendations = []
    if comparison["compression_ratio"] > 3.5:
        recommendations.append("✓ INT8 量化显著减小了模型大小（>3.5× 压缩率）")
    if comparison["speedup_ratio"] > 1.5:
        recommendations.append("✓ INT8 推理速度有明显提升")
    if fp32_latency < 100:
        recommendations.append("✓ 延迟满足实时性要求（< 100ms）")
    else:
        recommendations.append("⚠ 延迟较高，建议进一步优化（如剪枝、蒸馏）")
    if int8["model_size_mb"] < 50:
        recommendations.append("✓ INT8 模型大小适合移动端部署（< 50MB）")
    if comparison.get("accuracy_mse", 0) < 0.01:
        recommendations.append("✓ 量化精度损失在可接受范围内（MSE < 0.01）")

    for rec in recommendations:
        lines.append(f"  {rec}")

    lines.append("")
    lines.append("=" * 65)

    report_text = "\n".join(lines)

    # 保存 JSON 报告
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_info": model_info,
                "comparison": {
                    "fp32": {k: v for k, v in fp32.items() if k != "benchmark"},
                    "int8": {k: v for k, v in int8.items() if k != "benchmark"},
                    "fp32_benchmark": fp32["benchmark"],
                    "int8_benchmark": int8["benchmark"],
                    "compression_ratio": comparison["compression_ratio"],
                    "speedup_ratio": comparison["speedup_ratio"],
                    "accuracy_mse": comparison.get("accuracy_mse", 0),
                },
                "recommendations": recommendations,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"  JSON 报告已保存到 {output_path}")

    return report_text


# ============ 测试数据生成 ============
def generate_test_data(
    vocab_size: int = 512, seq_len: int = 32, num_samples: int = 100
) -> torch.Tensor:
    """生成模拟测试数据"""
    return torch.randint(0, vocab_size, (num_samples, seq_len))


# ============ 绘图函数 ============
def plot_deployment_metrics(comparison: Dict):
    """绘制部署指标对比图"""
    fp32 = comparison["fp32"]
    int8 = comparison["int8"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 左图：模型大小
    ax = axes[0]
    models = ["FP32", "INT8"]
    sizes = [fp32["model_size_mb"], int8["model_size_mb"]]
    bars = ax.bar(models, sizes, color=["#3498db", "#2ecc71"])
    ax.set_ylabel("模型大小 (MB)")
    ax.set_title("模型大小对比")
    for bar, s in zip(bars, sizes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{s:.2f} MB",
            ha="center",
            va="bottom",
        )

    # 中图：延迟
    ax = axes[1]
    batch_labels = []
    fp32_lats = []
    int8_lats = []
    for key in fp32["benchmark"]:
        bs = fp32["benchmark"][key]["batch_size"]
        batch_labels.append(f"batch={bs}")
        fp32_lats.append(fp32["benchmark"][key]["avg_latency_ms"])
        int8_lats.append(int8["benchmark"][key]["avg_latency_ms"])

    x = np.arange(len(batch_labels))
    width = 0.35
    ax.bar(x - width / 2, fp32_lats, width, label="FP32", color="#3498db")
    ax.bar(x + width / 2, int8_lats, width, label="INT8", color="#2ecc71")
    ax.set_ylabel("延迟 (ms)")
    ax.set_title("推理延迟对比")
    ax.set_xticks(x)
    ax.set_xticklabels(batch_labels)
    ax.legend()

    # 右图：吞吐量
    ax = axes[2]
    fp32_tps = [fp32["benchmark"][key]["throughput_qps"] for key in fp32["benchmark"]]
    int8_tps = [int8["benchmark"][key]["throughput_qps"] for key in int8["benchmark"]]
    ax.bar(x - width / 2, fp32_tps, width, label="FP32", color="#3498db")
    ax.bar(x + width / 2, int8_tps, width, label="INT8", color="#2ecc71")
    ax.set_ylabel("吞吐量 (samples/s)")
    ax.set_title("吞吐量对比")
    ax.set_xticks(x)
    ax.set_xticklabels(batch_labels)
    ax.legend()

    plt.suptitle("LLM 边缘部署指标对比")
    plt.tight_layout()
    plt.savefig("deployment_metrics.png", dpi=150)
    print("部署指标图已保存为 deployment_metrics.png")


# ============ 主程序 ============
if __name__ == "__main__":
    print("=" * 60)
    print("实验 5：LLM 边缘部署实验 - 完整实现")
    print("=" * 60)

    # 超参数
    VOCAB_SIZE = 512
    D_MODEL = 32
    NUM_HEADS = 4
    NUM_LAYERS = 2
    SEQ_LEN = 32

    # 1. 创建 FP32 模型
    print("\n[步骤 1] 创建 FP32 边缘 Transformer...")
    model = EdgeTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=SEQ_LEN,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params:,}")
    print(f"  估计模型大小 (FP32): {total_params * 4 / (1024 * 1024):.2f} MB")

    # 2. 模拟 ONNX 导出
    print("\n[步骤 2] 模拟 ONNX 导出...")
    model_info = simulate_onnx_export(model, (SEQ_LEN,), "edge_transformer.onnx")

    # 3. INT8 量化
    print("\n[步骤 3] INT8 量化...")
    int8_model = quantize_model_to_int8(model)
    int8_params = sum(p.numel() for p in int8_model.parameters())
    print(f"  INT8 模型参数量: {int8_params:,}")

    # 4. 生成测试数据
    print("\n[步骤 4] 生成测试数据...")
    test_data = generate_test_data(VOCAB_SIZE, SEQ_LEN, num_samples=50)

    # 5. FP32 vs INT8 全面对比
    print("\n[步骤 5] FP32 vs INT8 全面对比...")
    comparison = compare_fp32_int8(model, int8_model, (SEQ_LEN,), test_data)

    # 6. 生成部署报告
    print("\n[步骤 6] 生成部署报告...")
    report = generate_deployment_report(
        comparison, model_info, "deployment_report.json"
    )

    # 7. 打印报告
    print("\n" + report)

    # 8. 绘图
    print("\n[步骤 7] 绘制部署指标图...")
    plot_deployment_metrics(comparison)

    print("\n实验完成！")
