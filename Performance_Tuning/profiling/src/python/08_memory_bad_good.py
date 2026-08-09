#!/usr/bin/env python3
"""Purpose: temporary/clone/contiguous峰值显存 vs buffer reuse/in-place（inference only）。"""
import time
import torch

if not torch.cuda.is_available():
    print("当前PyTorch为CPU版：脚本语法已验证，GPU显存实验跳过")
    raise SystemExit(0)

def measure(fn, x):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(5):
        fn(x)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    result = None
    for _ in range(50):
        result = fn(x)
    end.record()
    end.synchronize()
    return result, start.elapsed_time(end) / 50, {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak": torch.cuda.max_memory_allocated(),
    }

x = torch.randn(4096, 4096, device="cuda")
def bad(v):
    a = v.clone()
    b = a.contiguous().clone()
    c = torch.relu(b)
    return c * 1.1 + 0.2

buffer = torch.empty_like(x)
def good(v):
    torch.relu(v, out=buffer)
    buffer.mul_(1.1).add_(0.2)
    return buffer

bad_result, bad_ms, bad_mem = measure(bad, x)
good_result, good_ms, good_mem = measure(good, x)
torch.testing.assert_close(bad_result, good_result)
print(f"bad_ms={bad_ms:.3f} good_ms={good_ms:.3f}")
print("bad_memory=", bad_mem)
print("good_memory=", good_mem)
if hasattr(torch.cuda.memory, "_snapshot"):
    print("memory snapshot API available; call torch.cuda.memory._snapshot() for offline analysis")
