#!/usr/bin/env python3
import argparse,statistics,time,numpy as np
try: import torch
except Exception: torch=None
p=argparse.ArgumentParser();p.add_argument("--frames",type=int,default=60);a=p.parse_args();cuda=bool(torch and torch.cuda.is_available())
names=["camera","decode","preprocess","H2D","vision","projector","language_action","action_decode","control"];stages={k:[] for k in names};e2e=[]
def cpu(name,fn):
    t=time.perf_counter();r=fn();stages[name].append((time.perf_counter()-t)*1e3);return r
for frame in range(a.frames+10):
    begin=time.perf_counter();img=cpu("camera",lambda:np.random.randint(0,256,(480,640,3),dtype=np.uint8));img=cpu("decode",lambda:img.copy());arr=cpu("preprocess",lambda:img[::2,::2].astype(np.float32)/255)
    if torch:
        x=torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
        if cuda:
            s,e=torch.cuda.Event(True),torch.cuda.Event(True);s.record();x=x.pin_memory().cuda(non_blocking=True);e.record();e.synchronize();stages["H2D"].append(s.elapsed_time(e))
        else:stages["H2D"].append(0.0)
        def gpu(name,fn):
            if cuda:
                s,e=torch.cuda.Event(True),torch.cuda.Event(True);s.record();r=fn();e.record();e.synchronize();stages[name].append(s.elapsed_time(e));return r
            return cpu(name,fn)
        feat=gpu("vision",lambda:x.mean((-1,-2)));proj=gpu("projector",lambda:feat.repeat(1,256));act=gpu("language_action",lambda:torch.tanh(proj).mean());cpu("action_decode",lambda:float(act.cpu()));cpu("control",lambda:time.sleep(.0005))
    else:
        stages["H2D"].append(0);feat=cpu("vision",lambda:arr.mean((0,1)));proj=cpu("projector",lambda:np.tile(feat,256));act=cpu("language_action",lambda:np.tanh(proj).mean());cpu("action_decode",lambda:float(act));cpu("control",lambda:time.sleep(.0005))
    if frame>=10:e2e.append((time.perf_counter()-begin)*1e3)
for k,v in stages.items():print(f"{k:16s} mean={statistics.mean(v[10:]):7.3f} ms")
q=lambda x:float(np.percentile(e2e,x));print(f"E2E mean={statistics.mean(e2e):.3f} P50={q(50):.3f} P90={q(90):.3f} P95={q(95):.3f} P99={q(99):.3f} max={max(e2e):.3f} ms FPS/Hz={1000/statistics.mean(e2e):.2f}")
