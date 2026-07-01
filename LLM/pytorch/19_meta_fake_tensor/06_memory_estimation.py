"""Meta FakeTensor case study 6: memory estimation and VRAM planning.

Companion script for meta_fake_tensor/meta_fake_tensor.md. Covers:
  1. Estimate model VRAM from meta tensors
  2. Activation memory estimation
  3. Planning batch size from memory budget

Run:
    python 06_memory_estimation.py
"""

import sys

import torch


def exp_model_memory_estimate():
    print("=" * 60)
    print("1. Estimate model memory without GPU")
    print("=" * 60)

    class TransformerForEstimate(torch.nn.Module):
        def __init__(self, hidden=1024, layers=12):
            super().__init__()
            self.embed = torch.nn.Embedding(32000, hidden)
            self.layers = torch.nn.ModuleList([
                torch.nn.TransformerEncoderLayer(
                    d_model=hidden, nhead=16, batch_first=True,
                    dim_feedforward=hidden * 4
                )
                for _ in range(layers)
            ])
            self.head = torch.nn.Linear(hidden, 32000)

        def forward(self, x):
            x = self.embed(x)
            for layer in self.layers:
                x = layer(x)
            return self.head(x)

    model = TransformerForEstimate(hidden=256, layers=6)

    # Parameter memory
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    grad_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)

    print(f"  Model: Transformer (hidden=256, layers=6, vocab=32000)")
    print(f"  Parameters:     {param_bytes / 1024**2:.1f} MB")
    print(f"  Gradients:      {grad_bytes / 1024**2:.1f} MB (same as params)")
    print(f"  Optimizer (Adam): {param_bytes * 2 / 1024**2:.1f} MB (moment1 + moment2)")
    total_param = param_bytes * 4 / 1024**2  # params + grads + 2x optimizer
    print(f"  Total param memory: {total_param:.1f} MB")

    # Activation memory estimate (rough: batch * seq * hidden * layers * 4 bytes * 2 for residual)
    batch, seq = 8, 1024
    hidden = 256
    n_layers = 6
    act_mem = batch * seq * hidden * n_layers * 4 * (4 + 1) / 1024**2  # ffn expands 4x + residual
    print(f"\n  Estimated activation memory (batch={batch}, seq={seq}): {act_mem:.1f} MB")
    print(f"  Total VRAM estimate: {total_param + act_mem:.1f} MB")
    print()


def exp_peak_memory_trace():
    print("=" * 60)
    print("2. Trace peak memory with meta tensors")
    print("=" * 60)

    # When you can't run the full model, meta tensors help estimate
    # Each layer: input + output of MHA + FFN activations

    hidden = 512
    batch, seq = 16, 2048

    # MHA: Q, K, V, attn_weights, attn_output
    mha_mem = batch * seq * hidden * 5 * 4 / 1024**2
    # FFN: expand 4x, activation, output
    ffn_mem = batch * seq * hidden * 4 * 2 * 4 / 1024**2

    print(f"  Per-layer activation estimate (hidden={hidden}, batch={batch}, seq={seq}):")
    print(f"    MHA activation: {mha_mem:.1f} MB")
    print(f"    FFN activation: {ffn_mem:.1f} MB")
    print(f"    Total per layer: {mha_mem + ffn_mem:.1f} MB")

    # With activation checkpointing: only 1 layer's activations stored
    with_ckpt = mha_mem + ffn_mem
    print(f"\n  With activation checkpoint: ~{with_ckpt:.1f} MB (one layer)")
    print(f"  Without checkpoint: ~{(mha_mem + ffn_mem) * 12:.1f} MB (all 12 layers)")
    print()


EXPERIMENTS = {
    "estimate": exp_model_memory_estimate,
    "peak": exp_peak_memory_trace,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[meta_fake_tensor case 6] DONE")


if __name__ == "__main__":
    main()
