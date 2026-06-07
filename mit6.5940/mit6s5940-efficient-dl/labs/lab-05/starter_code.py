"""
实验 5：LLM 边缘部署实验 - 起始代码
学生需要完成所有标记为 TODO 的部分

本实验模拟 LLM 在边缘设备上的部署流程：
1. 模拟 ONNX 导出
2. INT8 量化部署
3. CPU 推理基准测试
4. FP32 vs INT8 对比
5. 生成部署报告
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import copy
import json
import struct
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, field


# ============ 设备配置 ============
device = torch.device("cpu")  # 强制使用 CPU 模拟边缘设备
print(f"使用设备: {device}（模拟边缘设备 CPU）")


# ============ 数据类定义 ============
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


# ============ 微型 Transformer 模型 ============
class EdgeTransformer(nn.Module):
    """
    适合边缘部署的微型 Transformer 模型

    设计考虑：
    - 参数量小（< 1M 参数）
    - 层数少（适合低延迟推理）
    - 隐藏维度适中（适合 INT8 量化）
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

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        # 简化的 Transformer 层
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ffn1 = nn.Linear(d_model, 4 * d_model, bias=False)
        self.ffn2 = nn.Linear(4 * d_model, d_model, bias=False)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.num_layers = num_layers
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def _transformer_block(self, x):
        """单层 Transformer 块"""
        B, T, D = x.shape

        # 自注意力
        residual = x
        x = self.ln1(x)
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)
        x = residual + out

        # FFN
        residual = x
        x = self.ln2(x)
        x = F.relu(self.ffn1(x))
        x = self.ffn2(x)
        x = residual + x

        return x

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.token_embed(input_ids) + self.pos_embed[:, :T, :]

        for _ in range(self.num_layers):
            x = self._transformer_block(x)

        return self.lm_head(x)


# ============ TODO 1: 模拟 ONNX 导出 ============
def simulate_onnx_export(
    model: nn.Module, input_shape: Tuple, export_path: str = "model.onnx"
) -> Dict:
    """
    模拟 ONNX 导出流程

    在实际部署中，使用 torch.onnx.export() 导出模型。
    本实验模拟该过程：提取模型的计算图信息、参数和元数据

    参数:
        model: PyTorch 模型
        input_shape: 输入形状
        export_path: 导出路径（模拟）

    返回:
        model_info: 包含模型信息的字典
    """
    # TODO: 实现模拟导出
    # 步骤：
    # 1. 使用 torch.jit.trace 或直接遍历模块提取计算图信息
    # 2. 统计各层的类型和参数量
    # 3. 计算模型的总大小（参数 × 数据类型字节数）
    # 4. 记录输入/输出形状
    # 5. 将模型信息序列化为字典
    # 6. 模拟写入文件（可选）

    model_info = {
        "framework": "PyTorch",
        "export_format": "ONNX (simulated)",
        "input_shape": input_shape,
        "layers": [],
        "total_params": 0,
        "model_size_bytes": 0,
        "export_path": export_path,
    }

    # TODO: 填充 model_info
    pass

    return model_info


# ============ TODO 2: INT8 量化部署 ============
def quantize_model_to_int8(model: nn.Module) -> nn.Module:
    """
    将 FP32 模型量化为 INT8（模拟部署量化）

    在实际部署中，使用 ONNX Runtime 或 TensorRT 进行 INT8 量化。
    本实验模拟该过程：对权重进行 INT8 量化，激活保持 FP32

    参数:
        model: FP32 模型

    返回:
        quantized_model: INT8 量化后的模型（权重为 INT8，激活为 FP32）
    """
    # TODO: 实现 INT8 量化
    # 步骤：
    # 1. 深拷贝原始模型
    # 2. 遍历所有 nn.Linear 和 nn.Embedding 层
    # 3. 对每层的权重进行对称 INT8 量化
    #    - 计算 scale = max(abs(weight)) / 127
    #    - 量化: q_w = round(weight / scale), clamp to [-128, 127]
    #    - 反量化: deq_w = q_w * scale
    # 4. 用反量化的权重替换原始权重（模拟推理时的 dequantize）
    # 5. 返回量化后的模型

    q_model = copy.deepcopy(model)
    pass  # TODO: 完成实现

    return q_model


def get_model_size(model: nn.Module, weight_bits: int = 32) -> float:
    """
    计算模型的大小（MB）

    参数:
        model: PyTorch 模型
        weight_bits: 权重的位宽

    返回:
        size_mb: 模型大小（MB）
    """
    # TODO: 统计所有参数并计算字节数
    # 参数总数 × (weight_bits / 8) 字节
    pass


# ============ TODO 3: CPU 推理基准测试 ============
def benchmark_inference(
    model: nn.Module,
    input_shape: Tuple,
    num_warmup: int = 10,
    num_runs: int = 100,
    batch_sizes: List[int] = None,
) -> Dict:
    """
    在 CPU 上进行推理基准测试（模拟边缘设备）

    测量指标：
    - 延迟 (latency)：单个样本的推理时间
    - 吞吐量 (throughput)：每秒处理的样本数
    - 峰值内存使用

    参数:
        model: 待测试的模型
        input_shape: 输入形状 (seq_len,)
        num_warmup: 预热次数
        num_runs: 测量次数
        batch_sizes: 要测试的批次大小列表

    返回:
        benchmark_results: 包含延迟、吞吐量等指标的字典
    """
    if batch_sizes is None:
        batch_sizes = [1, 4, 8]

    # TODO: 实现基准测试
    # 步骤：
    # 1. 将模型设为 eval 模式
    # 2. 对每个 batch_size：
    #    a. 生成随机输入
    #    b. 预热 num_warmup 次
    #    c. 计时 num_runs 次推理
    #    d. 计算平均延迟和吞吐量
    # 3. 返回结果

    model.eval()
    results = {}

    for bs in batch_sizes:
        # TODO: 为每个 batch_size 进行基准测试
        pass

    return results


# ============ TODO 4: FP32 vs INT8 对比 ============
def compare_fp32_int8(
    fp32_model: nn.Module,
    int8_model: nn.Module,
    input_shape: Tuple,
    test_data: torch.Tensor = None,
) -> Dict:
    """
    全面对比 FP32 和 INT8 模型的性能

    参数:
        fp32_model: FP32 基线模型
        int8_model: INT8 量化模型
        input_shape: 输入形状
        test_data: 测试数据（可选）

    返回:
        comparison: 对比结果字典
    """
    # TODO: 实现全面对比
    # 步骤：
    # 1. 计算 FP32 和 INT8 的模型大小
    # 2. 进行推理基准测试
    # 3. 计算压缩率和加速比
    # 4. 如果有测试数据，计算精度差异
    # 5. 返回对比结果

    comparison = {
        "fp32": {},
        "int8": {},
        "compression_ratio": 0.0,
        "speedup_ratio": 0.0,
        "accuracy_drop": 0.0,
    }

    # TODO: 填充对比结果
    pass

    return comparison


# ============ TODO 5: 生成部署报告 ============
def generate_deployment_report(
    comparison: Dict, output_path: str = "deployment_report.json"
) -> str:
    """
    生成部署报告，包含所有指标和建议

    参数:
        comparison: compare_fp32_int8 的输出
        output_path: 报告输出路径

    返回:
        report_text: 部署报告的文本内容
    """
    # TODO: 实现部署报告生成
    # 步骤：
    # 1. 从 comparison 中提取所有指标
    # 2. 格式化为可读的文本报告
    # 3. 包含部署建议：
    #    - 是否建议使用 INT8 部署
    #    - 延迟是否满足实时性要求（< 100ms）
    #    - 内存使用是否在边缘设备可接受范围内
    # 4. 将报告写入 JSON 文件
    # 5. 返回报告文本

    report = []
    report.append("=" * 60)
    report.append("LLM 边缘部署报告")
    report.append("=" * 60)

    # TODO: 填充报告内容

    report_text = "\n".join(report)

    # 保存 JSON 报告
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    return report_text


# ============ 测试数据生成 ============
def generate_test_data(
    vocab_size: int = 512, seq_len: int = 32, num_samples: int = 100
) -> torch.Tensor:
    """生成测试数据"""
    return torch.randint(0, vocab_size, (num_samples, seq_len))


# ============ 评估精度损失 ============
def evaluate_accuracy_loss(
    fp32_model: nn.Module, int8_model: nn.Module, test_data: torch.Tensor
) -> float:
    """
    评估量化前后的输出差异（使用 MSE）

    参数:
        fp32_model: FP32 模型
        int8_model: INT8 模型
        test_data: 测试数据

    返回:
        mse: 均方误差
    """
    fp32_model.eval()
    int8_model.eval()

    total_mse = 0.0
    count = 0

    with torch.no_grad():
        batch_size = 8
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i : i + batch_size]
            fp32_out = fp32_model(batch)
            int8_out = int8_model(batch)
            total_mse += F.mse_loss(fp32_out, int8_out).item()
            count += 1

    return total_mse / count


# ============ 主程序 ============
if __name__ == "__main__":
    print("=" * 60)
    print("实验 5：LLM 边缘部署实验")
    print("=" * 60)

    # 超参数
    VOCAB_SIZE = 512
    D_MODEL = 32
    NUM_HEADS = 4
    NUM_LAYERS = 2
    SEQ_LEN = 32

    # 1. 创建模型
    print("\n[步骤 1] 创建 FP32 边缘 Transformer...")
    model = EdgeTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=SEQ_LEN,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")

    # 2. 模拟 ONNX 导出
    print("\n[步骤 2] 模拟 ONNX 导出...")
    # TODO: 调用 simulate_onnx_export

    # 3. INT8 量化
    print("\n[步骤 3] INT8 量化...")
    # TODO: 调用 quantize_model_to_int8

    # 4. 生成测试数据
    print("\n[步骤 4] 生成测试数据...")
    test_data = generate_test_data(VOCAB_SIZE, SEQ_LEN, num_samples=50)

    # 5. CPU 基准测试
    print("\n[步骤 5] CPU 推理基准测试...")
    # TODO: 对 FP32 和 INT8 模型分别进行基准测试

    # 6. FP32 vs INT8 对比
    print("\n[步骤 6] FP32 vs INT8 全面对比...")
    # TODO: 调用 compare_fp32_int8

    # 7. 精度损失评估
    print("\n[步骤 7] 评估量化精度损失...")
    # TODO: 调用 evaluate_accuracy_loss

    # 8. 生成部署报告
    print("\n[步骤 8] 生成部署报告...")
    # TODO: 调用 generate_deployment_report

    # 9. 打印报告
    print("\n[步骤 9] 打印部署报告...")
    # TODO: 打印报告内容

    print("\n实验完成！请将结果填入 report_template.md。")
