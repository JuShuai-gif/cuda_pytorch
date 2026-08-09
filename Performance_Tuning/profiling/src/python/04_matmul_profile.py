#!/usr/bin/env python3
import argparse,time,torch
p=argparse.ArgumentParser();p.add_argument("--size",type=int,default=2048);p.add_argument("--iters",type=int,default=20);a=p.parse_args()
d="cuda" if torch.cuda.is_available() else "cpu";x=torch.randn(a.size,a.size,device=d);y=torch.randn_like(x)
for _ in range(5):z=x@y
if d=="cuda":torch.cuda.synchronize()
t=time.perf_counter()
for _ in range(a.iters):z=x@y
if d=="cuda":torch.cuda.synchronize()
sec=time.perf_counter()-t;print(f"device={d} mean_ms={sec/a.iters*1e3:.3f} TFLOP/s={2*a.size**3*a.iters/sec/1e12:.3f}")
