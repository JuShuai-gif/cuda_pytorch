#!/usr/bin/env python3
import time, torch
if not torch.cuda.is_available():
    print("CUDA 不可用：跳过 GPU 实验"); raise SystemExit(0)
x=torch.randn(4096,4096,device="cuda")
for _ in range(10): _=x@x
t=time.perf_counter();_=x@x;print(f"错误的未同步计时: {(time.perf_counter()-t)*1e3:.3f} ms")
torch.cuda.synchronize();t=time.perf_counter();_=x@x;torch.cuda.synchronize();print(f"同步 wall time: {(time.perf_counter()-t)*1e3:.3f} ms")
s,e=torch.cuda.Event(True),torch.cuda.Event(True);s.record();_=x@x;e.record();e.synchronize();print(f"CUDA Event: {s.elapsed_time(e):.3f} ms")
