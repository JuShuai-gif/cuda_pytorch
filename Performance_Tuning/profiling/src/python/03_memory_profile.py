#!/usr/bin/env python3
import torch
from torch.profiler import ProfilerActivity, profile
d="cuda" if torch.cuda.is_available() else "cpu"; acts=[ProfilerActivity.CPU]+([ProfilerActivity.CUDA] if d=="cuda" else [])
with profile(activities=acts,profile_memory=True,record_shapes=True) as prof:
    blocks=[]
    for _ in range(20):
        blocks.append(torch.randn(1024,1024,device=d))
        if len(blocks)>4: blocks.pop(0)
print(prof.key_averages().table(sort_by="self_cpu_memory_usage",row_limit=15))
