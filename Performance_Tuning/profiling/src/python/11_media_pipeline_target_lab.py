#!/usr/bin/env python3
"""Camera/Decode/Preprocess目标机实验。

支持OpenCV设备/文件和自定义GStreamer pipeline。NVDEC、Jetson、MPP、RGA的
具体element名称依目标镜像插件而异，因此通过--pipeline注入，不硬编码不可移植API。
"""
import argparse
import csv
import statistics
import time
from pathlib import Path

try:
    import cv2
except ImportError:
    print("SKIP: OpenCV未安装")
    raise SystemExit(0)


def pct(values, p):
    values = sorted(values)
    pos = (len(values) - 1) * p / 100
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def report(name, values):
    if not values:
        return
    print(f"{name}: mean={statistics.mean(values):.3f} "
          f"P50={pct(values,50):.3f} P90={pct(values,90):.3f} "
          f"P95={pct(values,95):.3f} P99={pct(values,99):.3f} "
          f"min={min(values):.3f} max={max(values):.3f} "
          f"stddev={statistics.pstdev(values):.3f} ms")


parser = argparse.ArgumentParser()
parser.add_argument("--source", default="0", help="camera index或视频路径")
parser.add_argument("--pipeline", help="GStreamer pipeline，优先于source")
parser.add_argument("--backend", choices=["auto", "v4l2", "gstreamer"], default="auto")
parser.add_argument("--frames", type=int, default=300)
parser.add_argument("--warmup", type=int, default=30)
parser.add_argument("--width", type=int, default=224)
parser.add_argument("--height", type=int, default=224)
parser.add_argument("--sleep-ms", type=float, default=0.0, help="模拟下游阻塞")
parser.add_argument("--csv", default="media_pipeline_samples.csv")
args = parser.parse_args()

if args.pipeline:
    capture = cv2.VideoCapture(args.pipeline, cv2.CAP_GSTREAMER)
else:
    source = int(args.source) if args.source.isdigit() else args.source
    backend = cv2.CAP_V4L2 if args.backend == "v4l2" else cv2.CAP_ANY
    capture = cv2.VideoCapture(source, backend)
if not capture.isOpened():
    print("SKIP: 无法打开source/pipeline；检查设备权限和GStreamer插件")
    raise SystemExit(0)

for _ in range(args.warmup):
    ok, _ = capture.read()
    if not ok:
        break

rows, capture_ms, resize_ms, normalize_ms, e2e_ms = [], [], [], [], []
begin_run = time.perf_counter_ns()
for frame_id in range(args.frames):
    begin = time.perf_counter_ns()
    t = begin
    ok, frame = capture.read()
    after_capture = time.perf_counter_ns()
    if not ok:
        print(f"capture ended at frame={frame_id}")
        break
    resized = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
    after_resize = time.perf_counter_ns()
    normalized = resized.astype("float32") / 255.0
    after_normalize = time.perf_counter_ns()
    checksum = float(normalized[0, 0].sum())
    if args.sleep_ms:
        time.sleep(args.sleep_ms / 1000.0)
    end = time.perf_counter_ns()
    values = [(after_capture-t)/1e6, (after_resize-after_capture)/1e6,
              (after_normalize-after_resize)/1e6, (end-begin)/1e6]
    capture_ms.append(values[0]); resize_ms.append(values[1])
    normalize_ms.append(values[2]); e2e_ms.append(values[3])
    rows.append([frame_id, *values, checksum])

elapsed = (time.perf_counter_ns() - begin_run) / 1e9
capture.release()
report("capture_decode", capture_ms)
report("resize", resize_ms)
report("normalize", normalize_ms)
report("E2E", e2e_ms)
print(f"frames={len(rows)} FPS={len(rows)/elapsed:.2f} output={args.width}x{args.height}")
with Path(args.csv).open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "capture_decode_ms", "resize_ms",
                     "normalize_ms", "e2e_ms", "checksum"])
    writer.writerows(rows)

# pipeline示例（需按目标插件调整）：
# Jetson: nvarguscamerasrc ! ... ! nvvidconv ! video/x-raw,format=BGRx ! ... ! appsink
# NVIDIA dGPU: filesrc ! ... ! nvh264dec ! videoconvert ! appsink
# Rockchip: filesrc ! ... ! mppvideodec ! rgaconvert ! ... ! appsink
