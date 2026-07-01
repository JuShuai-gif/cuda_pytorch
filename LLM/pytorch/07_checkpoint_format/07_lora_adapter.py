"""Checkpoint Format case study 7: LoRA and adapter save/merge.

Companion script for checkpoint_format/checkpoint_format.md. Covers:
  1. LoRA adapter save/load
  2. Merge adapter into base model
  3. Multiple adapters management

Run:
    python 07_lora_adapter.py
"""

import sys

import torch


def exp_lora_basics():
    print("=" * 60)
    print("1. LoRA adapter: save only trainable params")
    print("=" * 60)

    class LoRALinear(torch.nn.Module):
        def __init__(self, in_features, out_features, rank=8):
            super().__init__()
            self.linear = torch.nn.Linear(in_features, out_features, bias=False)
            # LoRA: low-rank adaptation
            self.lora_A = torch.nn.Parameter(torch.randn(rank, in_features) * 0.01)
            self.lora_B = torch.nn.Parameter(torch.zeros(out_features, rank))
            # Freeze base weights
            self.linear.weight.requires_grad_(False)

        def forward(self, x):
            base = self.linear(x)
            lora = (x @ self.lora_A.T) @ self.lora_B.T
            return base + lora

    model = LoRALinear(128, 64, rank=8)
    base_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  LoRALinear(128, 64, rank=8):")
    print(f"    Base params (frozen):   {base_params:,}")
    print(f"    LoRA params (trainable): {lora_params:,}")
    print(f"    Ratio: {lora_params / (base_params + lora_params) * 100:.1f}%")

    # Save only adapter
    adapter_sd = {k: v for k, v in model.state_dict().items() if "lora" in k}
    torch.save(adapter_sd, "/tmp/lora_adapter.pt")
    import os
    print(f"\n  Adapter save size: {os.path.getsize('/tmp/lora_adapter.pt'):,} bytes")
    print(f"  Full model would be: ~{base_params * 4:,} bytes")
    os.remove("/tmp/lora_adapter.pt")
    print()


def exp_merge_adapter():
    print("=" * 60)
    print("2. Merge LoRA adapter into base model")
    print("=" * 60)

    # Simulate merge: W_merged = W + B @ A
    base_W = torch.randn(64, 128)
    lora_A = torch.randn(8, 128) * 0.01
    lora_B = torch.randn(64, 8) * 0.01

    merged_W = base_W + lora_B @ lora_A

    # Inference with merged (no extra compute for LoRA)
    print(f"  Base W shape:     {list(base_W.shape)}")
    print(f"  LoRA A shape:     {list(lora_A.shape)}")
    print(f"  LoRA B shape:     {list(lora_B.shape)}")
    print(f"  Merged W shape:   {list(merged_W.shape)}")
    print(f"\n  After merge: W_merged = W + B @ A")
    print(f"  -> Inference uses merged weight, no LoRA overhead")
    print()


EXPERIMENTS = {
    "lora": exp_lora_basics,
    "merge": exp_merge_adapter,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[checkpoint_format case 7] DONE")


if __name__ == "__main__":
    main()
