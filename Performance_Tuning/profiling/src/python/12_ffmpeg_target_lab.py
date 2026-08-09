#!/usr/bin/env python3
"""FFmpeg CLI软件/硬件decode、scale、format转换A/B采集器。"""
import argparse
import json
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("--mode", choices=["software", "cuda", "custom"], default="software")
parser.add_argument("--width", type=int, default=224)
parser.add_argument("--height", type=int, default=224)
parser.add_argument("--iterations", type=int, default=5)
parser.add_argument("--warmup", type=int, default=1)
parser.add_argument("--frames", type=int, default=500)
parser.add_argument("--extra-input", nargs="*", default=[])
parser.add_argument("--extra-filter", help="完全覆盖默认-vf，例如scale_cuda=224:224")
parser.add_argument("--json", default="ffmpeg_benchmark.json")
args = parser.parse_args()

if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
    print("SKIP: ffmpeg/ffprobe不存在")
    raise SystemExit(0)
if not Path(args.input).exists():
    print(f"SKIP: 输入不存在: {args.input}")
    raise SystemExit(0)

probe = subprocess.run([
    "ffprobe", "-v", "error", "-select_streams", "v:0",
    "-show_entries", "stream=codec_name,width,height,pix_fmt,avg_frame_rate",
    "-of", "json", args.input], text=True, capture_output=True, check=True)
metadata = json.loads(probe.stdout)
print("input_metadata=", json.dumps(metadata, ensure_ascii=False))

def command():
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-benchmark"]
    if args.mode == "cuda":
        cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    cmd += args.extra_input + ["-i", args.input, "-frames:v", str(args.frames)]
    if args.extra_filter:
        cmd += ["-vf", args.extra_filter]
    elif args.mode == "cuda":
        cmd += ["-vf", f"scale_cuda={args.width}:{args.height}"]
    else:
        cmd += ["-vf", f"scale={args.width}:{args.height},format=rgb24"]
    return cmd + ["-f", "null", "-"]

samples, details = [], []
for iteration in range(args.warmup + args.iterations):
    cmd = command()
    begin = time.perf_counter()
    result = subprocess.run(cmd, text=True, capture_output=True)
    elapsed = time.perf_counter() - begin
    if result.returncode:
        print("command=", " ".join(cmd))
        print(result.stderr)
        raise SystemExit(result.returncode)
    if iteration >= args.warmup:
        samples.append(elapsed)
        details.append({"iteration": iteration-args.warmup, "wall_s": elapsed,
                        "fps": args.frames/elapsed, "stderr": result.stderr})

ordered = sorted(samples)
def pct(p):
    pos = (len(ordered)-1)*p/100; lo = int(pos); hi = min(lo+1,len(ordered)-1)
    return ordered[lo] + (ordered[hi]-ordered[lo])*(pos-lo)
summary = {
    "mode": args.mode, "command": command(), "metadata": metadata,
    "iterations": args.iterations, "frames_per_iteration": args.frames,
    "mean_s": statistics.mean(samples), "p50_s": pct(50), "p90_s": pct(90),
    "p99_s": pct(99), "min_s": min(samples), "max_s": max(samples),
    "mean_fps": args.frames/statistics.mean(samples), "samples": details,
}
Path(args.json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps({k:v for k,v in summary.items() if k not in ("samples","metadata")}, indent=2))
print(f"report={args.json}")

# 示例：
# NVIDIA dGPU/NVDEC: --mode cuda
# Jetson: --mode custom --extra-input -c:v h264_nvv4l2dec --extra-filter 'scale=224:224'
# Rockchip FFmpeg build: --mode custom --extra-input -hwaccel rkmpp
# 具体decoder/filter名称用 `ffmpeg -decoders`、`ffmpeg -filters` 检测。
