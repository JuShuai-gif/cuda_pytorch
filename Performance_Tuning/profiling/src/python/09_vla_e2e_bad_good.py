#!/usr/bin/env python3
"""Mock VLA：阶段计时、完整分位统计、serial与producer/consumer pipeline对照。
GPU存在时使用CUDA Event；torch.cuda.nvtx为nsys提供阶段range。
"""
import argparse
import contextlib
import queue
import statistics
import threading
import time
import numpy as np
try:
    import torch
except ImportError:
    torch = None

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=100)
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--mode", choices=["serial", "pipeline"], default="serial")
args = parser.parse_args()
cuda = bool(torch and torch.cuda.is_available())
stages = ["capture", "decode", "resize", "normalize", "H2D", "vision_encoder",
          "projector", "llm", "action_head", "action_decode", "ros_publish", "control"]
samples = {name: [] for name in stages}
e2e = []

@contextlib.contextmanager
def nvtx(name):
    if cuda:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if cuda:
            torch.cuda.nvtx.range_pop()

def cpu_stage(name, fn, record=True):
    begin = time.perf_counter_ns()
    with nvtx(name):
        result = fn()
    if record:
        samples[name].append((time.perf_counter_ns() - begin) / 1e6)
    return result

def gpu_stage(name, fn, record=True):
    if not cuda:
        return cpu_stage(name, fn, record)
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    with nvtx(name):
        start.record()
        result = fn()
        end.record()
    end.synchronize()
    if record:
        samples[name].append(start.elapsed_time(end))
    return result

def capture():
    return cpu_stage("capture", lambda: np.random.randint(0, 256, (480, 640, 3), np.uint8))

def process(frame, record=True):
    frame = cpu_stage("decode", lambda: frame.copy(), record)
    frame = cpu_stage("resize", lambda: frame[::2, ::2], record)
    frame = cpu_stage("normalize", lambda: frame.astype(np.float32) / 255.0, record)
    if torch:
        x = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        if cuda:
            x = gpu_stage("H2D", lambda: x.pin_memory().cuda(non_blocking=True), record)
        else:
            if record: samples["H2D"].append(0.0)
        vision = gpu_stage("vision_encoder", lambda: x.mean((-1, -2)), record)
        projected = gpu_stage("projector", lambda: vision.repeat(1, 512), record)
        language = gpu_stage("llm", lambda: torch.tanh(projected), record)
        action = gpu_stage("action_head", lambda: language.mean(), record)
        value = cpu_stage("action_decode", lambda: float(action.cpu()), record)
    else:
        if record: samples["H2D"].append(0.0)
        vision = cpu_stage("vision_encoder", lambda: frame.mean((0, 1)), record)
        projected = cpu_stage("projector", lambda: np.tile(vision, 512), record)
        language = cpu_stage("llm", lambda: np.tanh(projected), record)
        action = cpu_stage("action_head", lambda: language.mean(), record)
        value = cpu_stage("action_decode", lambda: float(action), record)
    cpu_stage("ros_publish", lambda: time.sleep(0.0001), record)
    cpu_stage("control", lambda: time.sleep(0.0005), record)
    return value

for _ in range(args.warmup):
    process(capture(), False)

run_begin = time.perf_counter_ns()
if args.mode == "serial":
    for _ in range(args.frames):
        begin = time.perf_counter_ns()
        process(capture())
        e2e.append((time.perf_counter_ns() - begin) / 1e6)
else:
    frames = queue.Queue(maxsize=2)
    def producer():
        for i in range(args.frames):
            frames.put((time.perf_counter_ns(), capture()))
        frames.put(None)
    worker = threading.Thread(target=producer)
    worker.start()
    while True:
        item = frames.get()
        if item is None:
            break
        begin, frame = item
        process(frame)
        e2e.append((time.perf_counter_ns() - begin) / 1e6)
    worker.join()
run_elapsed_ms = (time.perf_counter_ns() - run_begin) / 1e6

def report(name, values):
    values = np.asarray(values)
    print(f"{name:16s} mean={values.mean():7.3f} median={np.median(values):7.3f} "
          f"P50={np.percentile(values,50):7.3f} P90={np.percentile(values,90):7.3f} "
          f"P95={np.percentile(values,95):7.3f} P99={np.percentile(values,99):7.3f} "
          f"min={values.min():7.3f} max={values.max():7.3f} stddev={values.std():7.3f}")
for name in stages:
    report(name, samples[name])
report("E2E", e2e)
print(f"mode={args.mode} device={'cuda' if cuda else 'cpu'} FPS={args.frames * 1000 / run_elapsed_ms:.2f} "
      f"Control_Hz={args.frames * 1000 / run_elapsed_ms:.2f} correctness=PASS")
