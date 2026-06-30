"""Mixture of Experts (MoE) demo: from scratch implementation.

Companion script for distributed_techniques/moe/README.md.
  - Router with top-k gating
  - Multiple expert FFNs (sparse activation)
  - Load balancing via auxiliary loss and expert capacity
  - Token routing visualization

Run:
    python test1.py              # full demo
    python test1.py basic        # basic MoE forward
    python test1.py topk         # compare top-1, top-2, top-4 routing
    python test1.py balance      # load balancing analysis
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============ 1. MoE layer from scratch ============
class SparseMoE(nn.Module):
    """A sparse Mixture-of-Experts layer.

    Args:
        d_model: input/output hidden size
        d_ff:    expert FFN intermediate size
        num_experts: total number of experts
        top_k:   how many experts to activate per token
        capacity: max tokens per expert (None = no limit, "auto" = balanced)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        capacity: float | None = None,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity = capacity
        self.aux_loss_coef = aux_loss_coef

        # Router: linear projection -> logits for each expert
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # Experts: stacked FFNs (each is [d_model -> d_ff -> d_model])
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Linear(d_ff, d_model),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq, d_model]
        Returns:
            output: [batch, seq, d_model]
            aux_loss: scalar auxiliary load-balancing loss
        """
        B, S, D = x.shape
        x_flat = x.view(-1, D)  # [B*S, D]
        N = x_flat.shape[0]

        # === Step 1: Router computes expert logits ===
        router_logits = self.router(x_flat)  # [N, num_experts]

        # === Step 2: Top-k gating ===
        topk_logits, topk_indices = router_logits.topk(self.top_k, dim=-1)
        # topk_logits:   [N, top_k]   raw logits for selected experts
        # topk_indices:  [N, top_k]   expert indices

        # Softmax over top-k experts only
        topk_gates = F.softmax(topk_logits, dim=-1)  # [N, top_k]

        # === Step 3: Initialize output ===
        output = torch.zeros_like(x_flat)

        # === Step 4: Expert capacity (optional) ===
        if self.capacity is not None:
            C = (
                self.capacity
                if isinstance(self.capacity, int)
                else int(N / self.num_experts * 1.5)
            )
        else:
            C = N  # no capacity limit

        # === Step 5: Token dispatch to each expert ===
        aux_loss = torch.tensor(0.0, device=x.device)

        # Count tokens per expert for load balancing
        expert_counts = torch.zeros(self.num_experts, device=x.device)
        for k in range(self.top_k):
            expert_indices_k = topk_indices[:, k]  # [N]
            expert_counts.scatter_add_(
                0, expert_indices_k, torch.ones(N, device=x.device)
            )

        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = topk_indices == expert_idx  # [N, top_k]
            token_mask = expert_mask.any(dim=-1)  # [N]
            token_indices = torch.where(token_mask)[0]  # [n_tokens]

            if len(token_indices) == 0:
                continue

            # Apply capacity: if too many tokens, drop overflow
            n_actual = min(len(token_indices), C)
            if n_actual < len(token_indices):
                token_indices = token_indices[:n_actual]

            # Get gate weights for this expert
            gate_vals = torch.zeros(N, device=x.device)
            for k in range(self.top_k):
                match_k = topk_indices[:, k] == expert_idx  # [N]
                gate_vals[match_k] = topk_gates[match_k, k]  # [N]

            expert_gates = gate_vals[token_indices]  # [n_actual]

            # Forward through this expert
            expert_input = x_flat[token_indices]  # [n_actual, D]
            expert_output = self.experts[expert_idx](expert_input)  # [n_actual, D]

            # Weight by gate and accumulate
            output.index_add_(
                0, token_indices, expert_output * expert_gates.unsqueeze(-1)
            )

        # === Step 6: Auxiliary load-balancing loss ===
        # Encourage uniform token distribution across experts
        if self.training:
            # Fraction of tokens assigned to each expert
            f_i = expert_counts / (N * self.top_k + 1e-10)
            # Mean router probability for each expert
            P_i = router_logits.softmax(dim=-1).mean(dim=0)  # [num_experts]
            aux_loss = self.aux_loss_coef * self.num_experts * (f_i * P_i).sum()

        output = output.view(B, S, D)
        return output, aux_loss


# ============ 2. Basic forward pass ============
def exp_basic():
    print("=" * 60)
    print("1. Basic MoE forward pass")
    print("=" * 60)

    torch.manual_seed(42)
    B, S, D = 2, 8, 64
    x = torch.randn(B, S, D)  # 2 sequences, 8 tokens each, d=64

    moe = SparseMoE(d_model=D, d_ff=128, num_experts=4, top_k=2)
    moe.eval()

    with torch.no_grad():
        y, _ = moe(x)

    # Equivalent dense FFN for comparison
    dense_ffn = nn.Sequential(nn.Linear(D, 4 * 128), nn.GELU(), nn.Linear(4 * 128, D))
    dense_ffn.eval()
    dense_params = sum(p.numel() for p in dense_ffn.parameters())
    moe_params = sum(p.numel() for p in moe.parameters())

    print(f"  Input:              {list(x.shape)}")
    print(f"  Output:             {list(y.shape)}")
    print(f"  Experts:            4, top_k=2")
    print(f"  Dense FFN params:   {dense_params:,}")
    print(f"  MoE params:         {moe_params:,}  (experts + router)")
    print(
        f"  Activated params:   ~{moe_params * 2 // 4:,}  (only 2/4 experts active per token)"
    )
    print("  -> MoE scales total capacity while keeping per-token compute bounded")
    print()


# ============ 3. Top-k comparison ============
def exp_topk():
    print("=" * 60)
    print("2. Top-k routing comparison")
    print("=" * 60)

    torch.manual_seed(42)
    x = torch.randn(4, 16, 128)  # [B=4, S=16, D=128]

    for top_k in [1, 2, 4]:
        moe = SparseMoE(d_model=128, d_ff=256, num_experts=8, top_k=top_k)
        moe.eval()

        with torch.no_grad():
            router_logits = moe.router(x.view(-1, 128))
            _, topk_idx = router_logits.topk(top_k, dim=-1)

            # Count which experts get how many tokens
            counts = torch.zeros(8, dtype=torch.long)
            for k in range(top_k):
                for e in range(8):
                    counts[e] += (topk_idx[:, k] == e).sum().item()

            y, _ = moe(x)

        print(f"  top_k={top_k}:")
        print(f"    expert token counts: {counts.tolist()}")
        print(f"    total tokens routed: {counts.sum().item()}")
        print(f"    mean tokens/expert:  {counts.float().mean():.1f}")
        print(f"    std  tokens/expert:  {counts.float().std():.1f}")
        print()
    print("  -> Larger top_k = more even distribution, but more compute per token")
    print()


# ============ 4. Load balancing ============
def exp_balance():
    print("=" * 60)
    print("3. Load balancing analysis")
    print("=" * 60)

    torch.manual_seed(42)
    x = torch.randn(2, 32, 256)

    # Run without aux loss
    moe_no_aux = SparseMoE(
        d_model=256, d_ff=512, num_experts=8, top_k=2, aux_loss_coef=0.0
    )
    moe_no_aux.eval()
    with torch.no_grad():
        logits = moe_no_aux.router(x.view(-1, 256))
        _, idx = logits.topk(2, dim=-1)
        counts_no_aux = torch.zeros(8)
        for k in range(2):
            for e in range(8):
                counts_no_aux[e] += (idx[:, k] == e).sum().item()

    print("  Without aux loss (eval, no training):")
    print(f"    expert token counts: {counts_no_aux.int().tolist()}")
    print(
        f"    load imbalance:      {counts_no_aux.max() / counts_no_aux.mean():.2f}x worst ratio"
    )

    # Simulate aux loss effect by perturbing logits toward uniform
    logits_uniform = (
        logits + torch.randn_like(logits) * 0.5
    )  # add noise -> more uniform
    _, idx2 = logits_uniform.topk(2, dim=-1)
    counts_uni = torch.zeros(8)
    for k in range(2):
        for e in range(8):
            counts_uni[e] += (idx2[:, k] == e).sum().item()

    print(f"\n  With aux loss effect (simulated):")
    print(f"    expert token counts: {counts_uni.int().tolist()}")
    print(
        f"    load imbalance:      {counts_uni.max() / counts_uni.mean():.2f}x worst ratio"
    )

    print("\n  -> Aux loss encourages uniform routing during training")
    print("     Expert capacity caps max tokens, overflow passed through as-is")
    print()


EXPERIMENTS = {
    "basic": exp_basic,
    "topk": exp_topk,
    "balance": exp_balance,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for exp in exps:
        if exp not in EXPERIMENTS:
            print(f"unknown exp '{exp}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[exp]()

    print("[MoE demo] DONE")


if __name__ == "__main__":
    main()
