#!/usr/bin/env python3
import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule, tensorboard_trace_handler
device = "cuda" if torch.cuda.is_available() else "cpu"
activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == "cuda" else [])
a, b = torch.randn(1024, 1024, device=device), torch.randn(1024, 1024, device=device)
with profile(activities=activities, record_shapes=True, profile_memory=True, with_stack=True,
             schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
             on_trace_ready=tensorboard_trace_handler("./torch_trace")) as prof:
    for _ in range(5):
        with record_function("matmul_relu"): c = (a @ b).relu()
        prof.step()
print(prof.key_averages().table(sort_by="self_cuda_time_total" if device == "cuda" else "self_cpu_time_total", row_limit=15))
