#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
python3 -c 'import torch' 2>/dev/null || { echo "PyTorch不存在，跳过"; exit 0; }
python3 "$root/python/07_operator_hotspot_bad_good.py"
python3 "$root/python/08_memory_bad_good.py"
python3 "$root/python/09_vla_e2e_bad_good.py" --frames 30 --mode serial
python3 "$root/python/09_vla_e2e_bad_good.py" --frames 30 --mode pipeline
