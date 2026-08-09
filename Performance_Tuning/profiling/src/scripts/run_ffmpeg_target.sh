#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
input="${1:-}"
[[ -n "$input" ]] || { echo "usage: $0 INPUT [software|cuda|custom]"; exit 2; }
mode="${2:-software}"
command -v ffmpeg >/dev/null || { echo "SKIP: ffmpeg不存在"; exit 0; }
echo "===== FFmpeg capabilities ====="
ffmpeg -hide_banner -hwaccels
ffmpeg -hide_banner -decoders 2>/dev/null | grep -E 'cuvid|nvdec|v4l2|rkmpp' || true
ffmpeg -hide_banner -filters 2>/dev/null | grep -E 'scale_cuda|scale_npp|rga' || true
echo "===== CLI benchmark: $mode ====="
python3 "$root/python/12_ffmpeg_target_lab.py" "$input" --mode "$mode" \
  --json "$root/ffmpeg_${mode}_result.json"
if [[ -x "$root/build/35_ffmpeg_decode_benchmark" ]]; then
  echo "===== LibAV decode+swscale ====="
  "$root/build/35_ffmpeg_decode_benchmark" "$input" 500 224 224
else
  echo "SKIP LibAV C++ target: FFmpeg开发库未安装"
fi
