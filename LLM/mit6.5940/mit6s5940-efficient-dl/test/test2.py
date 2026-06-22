from typing import Any

import torch
import torch.nn as nn
import time
import numpy as np

def count_parameters(model):
    """统计参数量和模型大小"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total * 4 / (1024**2)
    return total, trainable, size_mb

def measure_latency(model,input_shape,device = 'cpu',warmup = 10,repeat = 100):
    """测量推理延迟"""
    model = model.to(device).eval()
    dummy = torch.randn(*input_shape).to(device)

    # Warmup
    for _ in range(warmup):
        _ = model(dummy)

    # 如果是 GPU，同步
    if device == 'cuda':
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        _ = model(dummy)
        if device == 'cuda':
            torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_latency = (end - start) / repeat * 1000  # ms
    return avg_latency  


# 使用示例
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.fc = nn.Linear(32 * 8 * 8,10)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = TinyCNN()
total,trainable,size_mb = count_parameters(model)
latency_cpu = measure_latency(model,(1,3,32,32),'cpu')

print(f"参数量: {total:,} | 模型大小: {size_mb:.2f} MB")
print(f"CPU推理延迟: {latency_cpu:.2f} ms")




























