#!/usr/bin/env python3
"""Purpose: Transformer-like算子热点；Bad逐元素碎片化，Good融合表达式。
Profiler: torch.profiler -> nsys -> ncu；优化前先torch.testing.assert_close。
"""
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(512, 1536)
        self.proj = nn.Linear(512, 512)
        self.norm = nn.LayerNorm(512)
        self.ff1 = nn.Linear(512, 1024)
        self.ff2 = nn.Linear(1024, 512)

    def forward(self, x, good):
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
        score = torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
        attn = torch.softmax(score, dim=-1)
        x = x + self.proj(torch.matmul(attn, v))
        h = self.ff1(x)
        if good:
            h = torch.nn.functional.silu(h) * h
        else:
            sigmoid = torch.sigmoid(h)
            silu = h * sigmoid
            h = silu * h
        return self.ff2(h)

model = TinyBlock().to(device=device, dtype=dtype).eval()
x = torch.randn(2, 256, 512, device=device, dtype=dtype)
with torch.inference_mode():
    bad = model(x, False)
    good = model(x, True)
    torch.testing.assert_close(bad, good, rtol=2e-3, atol=2e-3)
    for _ in range(5):
        model(x, True)
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, record_shapes=True,
                 profile_memory=True, with_stack=True) as prof:
        for _ in range(10):
            with record_function("bad_fragmented_elementwise"):
                model(x, False)
            with record_function("good_compact_expression"):
                model(x, True)
    sort_key = "self_cuda_time_total" if device == "cuda" else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=20))
    print(f"device={device} correctness=PASS")
