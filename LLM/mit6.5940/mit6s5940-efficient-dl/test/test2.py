from typing import Any
import copy

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
import time
import numpy as np


def count_parameters(model):
    """统计参数量和模型大小(按实际 dtype 计算存储)"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    size_mb = size_bytes / (1024**2)
    size_gb = size_bytes / (1024**3)
    return total, trainable, size_mb, size_gb


def measure_latency(model, input_shape, device="cpu", warmup=10, repeat=100):
    """测量推理延迟"""
    model = model.to(device).eval()
    dummy = torch.randn(*input_shape).to(device)

    # Warmup
    for _ in range(warmup):
        _ = model(dummy)

    # 如果是 GPU，同步
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        _ = model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
    end = time.perf_counter()

    avg_latency = (end - start) / repeat * 1000  # ms
    return avg_latency


# ============ 测 MACs / FLOPs 的三种方式 ============
# 统一口径: 1 MAC(乘加) = 2 FLOPs, 各函数都返回 (macs, flops)。
# 注意: 不同工具原生口径不同, 已在各函数内统一换算。


def count_macs_profiler(model, input_shape, device="cpu"):
    """方式 A: torch.profiler 内置(无需额外依赖)。
    profiler 的 e.flops 估计的是 FLOPs, 故 MACs = flops / 2。
    缺点: 只覆盖部分算子(conv/matmul 等), 统计可能不全。
    """
    model = copy.deepcopy(model).to(device).eval()
    dummy = torch.randn(*input_shape, device=device)
    with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
        model(dummy)
    flops = sum(e.flops for e in prof.key_averages())
    return flops / 2, flops


def count_macs_thop(model, input_shape, device="cpu"):
    """方式 B: thop(最常用)。thop.profile 返回的是 MACs, 故 FLOPs = 2 * macs。
    未安装时返回 None, 安装: pip install thop
    """
    try:
        from thop import profile as thop_profile
    except ImportError:
        print("[thop 未安装] pip install thop")
        return None
    model = copy.deepcopy(model).to(device).eval()
    dummy = torch.randn(*input_shape, device=device)
    macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
    return macs, 2 * macs


def count_macs_fvcore(model, input_shape, device="cpu"):
    """方式 C: fvcore(统计更细)。fvcore 的 total() 实际是 MAC 计数, 故 FLOPs = 2 * total。
    未安装时返回 None, 安装: pip install fvcore
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        print("[fvcore 未安装] pip install fvcore")
        return None
    model = copy.deepcopy(model).to(device).eval()
    dummy = torch.randn(*input_shape, device=device)
    macs = FlopCountAnalysis(model, dummy).total()
    return macs, 2 * macs


def compute_profile(model, input_shape, latency_ms, device="cpu"):
    """结合 MACs 与延迟, 估算实测算力(achieved GFLOPS)与算术强度(roofline 判据)。
    算术强度 = FLOPs / Bytes; 高 -> compute-bound, 低 -> memory-bound。
    """
    macs, flops = count_macs_profiler(model, input_shape, device)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    achieved_gflops = flops / (latency_ms / 1000) / 1e9 if latency_ms > 0 else 0.0
    intensity = flops / param_bytes if param_bytes > 0 else 0.0
    return {
        "GMACs": macs / 1e9,
        "GFLOPs": flops / 1e9,
        "achieved_GFLOPS": achieved_gflops,
        "arithmetic_intensity": intensity,
    }


# 使用示例
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


model = TinyCNN()
total, trainable, size_mb, size_gb = count_parameters(model)
latency_cpu = measure_latency(model, (1, 3, 32, 32), "cpu")


print(f"参数量: {total:,} | 训练参数量: {trainable:,}")
print(f"模型存储: {size_mb:.2f} MB ({size_gb:.4f} GB)")
print(f"CPU推理延迟: {latency_cpu:.2f} ms")

# 三种方式测 MACs / FLOPs(thop / fvcore 未装时会自动跳过)
input_shape = (1, 3, 32, 32)
print("\n--- MACs / FLOPs (三种方式对比) ---")
for name, fn in [
    ("profiler", count_macs_profiler),
    ("thop", count_macs_thop),
    ("fvcore", count_macs_fvcore),
]:
    res = fn(model, input_shape)
    if res is not None:
        macs, flops = res
        print(f"{name:>9}: {macs / 1e6:8.3f} MMACs | {flops / 1e6:8.3f} MFLOPs")

# 结合延迟估算实测算力与算术强度(roofline)
prof = compute_profile(model, input_shape, latency_cpu)
print("\n--- 实测算力 & 算术强度 (roofline) ---")
print(f"实测算力: {prof['achieved_GFLOPS']:.2f} GFLOPS")
print(f"算术强度: {prof['arithmetic_intensity']:.2f} FLOPs/Byte")
