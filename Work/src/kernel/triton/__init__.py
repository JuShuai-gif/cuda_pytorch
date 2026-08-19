"""Triton kernel labs.

This package pins the Triton toolchain to the system ``ptxas`` (CUDA 13),
because the ``ptxas-blackwell`` bundled with Triton 3.6 is CUDA 12.9 and does
not recognize ``sm_110a`` (this machine's NVIDIA Thor).  See README.md.
"""

from __future__ import annotations

import os

# Triton picks ptxas-blackwell for arch >= 100; its bundled binary is CUDA 12.9
# and rejects sm_110a.  Point it at the system CUDA 13 ptxas (supports sm_110a)
# unless the user already set it.
os.environ.setdefault("TRITON_PTXAS_BLACKWELL_PATH", "/usr/local/cuda-13.0/bin/ptxas")
