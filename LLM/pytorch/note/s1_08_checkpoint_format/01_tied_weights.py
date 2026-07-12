"""Checkpoint Format case study 1: tied weights, state_dict, and SafeTensors.

Companion script for checkpoint_format/checkpoint_format.md. Covers:
  1. Save/load state_dict correctly
  2. Tied weights (embedding/lm_head) handling
  3. SafeTensors save/load

Run:
    python 01_tied_weights.py
"""

import sys

import torch


class LLMWithTiedWeights(torch.nn.Module):
    def __init__(self, vocab_size=1000, hidden=256):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab_size, bias=False)
        # Tie weights
        self.lm_head.weight = self.embed.weight

    def forward(self, x):
        return self.lm_head(self.embed(x))


def exp_tied_weights_roundtrip():
    print("=" * 60)
    print("1. Tied weights save/load roundtrip")
    print("=" * 60)

    model = LLMWithTiedWeights()
    embed_ptr = model.embed.weight.data_ptr()
    head_ptr = model.lm_head.weight.data_ptr()
    print(f"  embed.data_ptr:     {embed_ptr}")
    print(f"  lm_head.data_ptr:   {head_ptr}")
    print(f"  Same storage?       {embed_ptr == head_ptr}")

    # state_dict() has two entries for tied weight
    sd = model.state_dict()
    print(f"\n  state_dict keys:")
    for k in sd:
        print(f"    {k}: shape={list(sd[k].shape)}")

    # The two entries point to same data at save time
    # After load, they become independent copies
    print(f"\n  embed_weight is lm_head_weight in state_dict: {sd['embed.weight'] is sd['lm_head.weight']}")
    print(f"  They are copies in the OrderedDict, not the same object")

    # Save and load
    torch.save(sd, "/tmp/tied_weights_test.pt")
    sd_loaded = torch.load("/tmp/tied_weights_test.pt", weights_only=True)

    # Load into fresh model
    model2 = LLMWithTiedWeights()
    model2.load_state_dict(sd_loaded, strict=False)

    # After loading, both params are separate tensors
    print(f"\n  After load_state_dict:")
    e_ptr = model2.embed.weight.data_ptr()
    l_ptr = model2.lm_head.weight.data_ptr()
    print(f"  embed.data_ptr:    {e_ptr}")
    print(f"  lm_head.data_ptr:  {l_ptr}")
    print(f"  Same storage?      {e_ptr == l_ptr}")

    # Must re-tie after loading
    model2.lm_head.weight = model2.embed.weight
    print(f"\n  After manual re-tie:")
    print(f"  Same storage?      {model2.embed.weight.data_ptr() == model2.lm_head.weight.data_ptr()}")

    import os
    os.remove("/tmp/tied_weights_test.pt")
    print()


def exp_safetensors_compare():
    print("=" * 60)
    print("2. SafeTensors vs torch.save")
    print("=" * 60)

    model = torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10),
    )

    sd = model.state_dict()

    # torch.save
    torch.save(sd, "/tmp/model.pt")
    import os
    pt_size = os.path.getsize("/tmp/model.pt")

    # SafeTensors
    has_safetensors = False
    try:
        from safetensors.torch import save_file, load_file
        save_file(sd, "/tmp/model.safetensors")
        st_size = os.path.getsize("/tmp/model.safetensors")
        has_safetensors = True
        print(f"  torch.save size:    {pt_size:,} bytes")
        print(f"  SafeTensors size:   {st_size:,} bytes")

        # Load back
        sd_loaded = load_file("/tmp/model.safetensors")
        for k in sd:
            assert torch.allclose(sd[k], sd_loaded[k])
        print(f"  Roundtrip: OK")
    except ImportError:
        print(f"  safetensors not installed: pip install safetensors")

    # Security demo: torch.save can execute arbitrary code
    print(f"\n  Security difference:")
    print(f"    torch.save: pickle-based -> can execute arbitrary code")
    print(f"    SafeTensors: pure tensor data -> safe by construction")
    print(f"    Use torch.load(weights_only=True) as mitigation")

    for f in ["/tmp/model.pt", "/tmp/model.safetensors"]:
        if os.path.exists(f):
            os.remove(f)
    print()


def exp_optimizer_state():
    print("=" * 60)
    print("3. Optimizer state_dict: per-parameter key")
    print("=" * 60)

    model = torch.nn.Linear(8, 4)
    optim = torch.optim.AdamW(model.parameters(), lr=0.01)

    # Run one step
    x = torch.randn(4, 8)
    loss = model(x).sum()
    loss.backward()
    optim.step()

    sd = optim.state_dict()
    print(f"  Optimizer state_dict keys:")
    for k in sd:
        print(f"    {k}")
        if k == "state":
            for param_id, state in sd[k].items():
                print(f"      param_id={param_id}")
                for sk in state:
                    print(f"        {sk}: shape={state[sk].shape if hasattr(state[sk], 'shape') else state[sk]}")

    print(f"\n  Key insight:")
    print(f"    - state keys are id(parameter) -> not portable across runs")
    print(f"    - Cannot load optimizer state into a model with different structure")
    print(f"    - Typically tied to exact model architecture and parameter order")

    # Save optimizer state alongside model state_dict
    full_ckpt = {
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "step": 10,
    }
    torch.save(full_ckpt, "/tmp/full_ckpt.pt")
    print(f"\n  Full checkpoint saved with model + optimizer state")

    import os
    os.remove("/tmp/full_ckpt.pt")
    print()


EXPERIMENTS = {
    "tied": exp_tied_weights_roundtrip,
    "safetensors": exp_safetensors_compare,
    "optim": exp_optimizer_state,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[checkpoint_format case 1] DONE")


if __name__ == "__main__":
    main()
