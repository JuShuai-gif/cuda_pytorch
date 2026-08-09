#!/usr/bin/env python3
import time,torch
from torch.nn.functional import scaled_dot_product_attention
d="cuda" if torch.cuda.is_available() else "cpu";q=torch.randn(2,8,512,64,device=d);k=torch.randn_like(q);v=torch.randn_like(q)
for _ in range(5):out=scaled_dot_product_attention(q,k,v)
if d=="cuda":torch.cuda.synchronize()
t=time.perf_counter()
for _ in range(20):out=scaled_dot_product_attention(q,k,v)
if d=="cuda":torch.cuda.synchronize()
print(f"device={d} attention_mean_ms={(time.perf_counter()-t)/20*1e3:.3f}")
