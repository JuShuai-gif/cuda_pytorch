"""Final Project B: robot inference runtime (Python).

Two versions of the camera -> preprocess -> vision -> policy -> action path:

  naive     - strictly serial: decode/preprocess on CPU, synchronous H2D,
              then a normal model forward
  optimized - double buffering + async H2D + CUDA Graph, so the CPU can
              prepare frame N+1 while the GPU computes frame N

The optimized runtime uses two GPU input buffers (double buffering), an async
copy stream for H2D, and a captured CUDA graph for the forward pass.  The
benchmark compares sensor-to-action latency, jitter (p99-p50), and throughput.
"""

from __future__ import annotations

import torch

from inference.vlm.pipeline import VLM, decode_image, preprocess


class NaiveRuntime:
    def __init__(self, device):
        self.model = VLM().to(device).eval()
        self.device = device

    def infer(self, frame: bytes) -> torch.Tensor:
        img = decode_image(frame)                 # CPU
        x = preprocess(img).unsqueeze(0)          # CPU
        x = x.to(self.device)                     # synchronous H2D
        return self.model.vision_encode(x)        # GPU (vision+connector)


class OptimizedRuntime:
    def __init__(self, device):
        self.model = VLM().to(device).eval()
        self.device = device
        self.copy_stream = torch.cuda.Stream(device=device)
        self.compute_stream = torch.cuda.Stream(device=device)

        # Double-buffered GPU inputs.
        self.buffers = [torch.empty(1, 3, 224, 224, device=device) for _ in range(2)]
        self.idx = 0

        # Capture the forward pass into a CUDA graph (static buffer).
        static = self.buffers[0]
        g = torch.cuda.CUDAGraph()
        side = torch.cuda.Stream(device=device)
        side.wait_stream(self.compute_stream)
        with torch.cuda.stream(side):
            self.model.vision_encode(static)
        self.compute_stream.wait_stream(side)
        with torch.cuda.graph(g):
            self.model.vision_encode(static)
        self.graph = g

    def infer(self, frame: bytes) -> torch.Tensor:
        img = decode_image(frame)
        x = preprocess(img).unsqueeze(0)          # CPU

        buf = self.buffers[self.idx]
        # Async H2D on the copy stream.
        self.copy_stream.wait_stream(self.compute_stream)
        with torch.cuda.stream(self.copy_stream):
            buf.copy_(x, non_blocking=True)

        # Replay the graph on the compute stream after the copy lands.
        self.compute_stream.wait_stream(self.copy_stream)
        with torch.cuda.stream(self.compute_stream):
            self.graph.replay()

        self.idx = (self.idx + 1) % 2
        return buf  # result written in-place into buf (sync before read)

    def sync(self):
        self.compute_stream.synchronize()
