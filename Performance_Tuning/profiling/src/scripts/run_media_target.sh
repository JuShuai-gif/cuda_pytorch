#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
echo "V4L2 raw capture:"
"$root/build/34_v4l2_capture_benchmark" --device "${VIDEO_DEVICE:-/dev/video0}" \
  --width "${VIDEO_WIDTH:-640}" --height "${VIDEO_HEIGHT:-480}" \
  --format "${VIDEO_FORMAT:-YUYV}" --frames "${VIDEO_FRAMES:-300}" || true
echo "OpenCV/GStreamer decode+preprocess:"
python3 "$root/python/11_media_pipeline_target_lab.py" "$@"
