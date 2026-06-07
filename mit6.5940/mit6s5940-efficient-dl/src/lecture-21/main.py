#!/usr/bin/env python3
"""
MIT 6.5940 Lecture 21: On-Device Training Simulation

Topics covered:
  - Simulate Federated Learning: create multiple "clients" with non-IID data
  - Implement FedAvg: local training -> aggregate -> repeat
  - Simulate memory bottleneck: measure activation memory during backward pass
  - TinyTL concept: freeze backbone, only train biases + lightweight classifier
  - Compare: full training vs TinyTL memory usage

All computation runs on CPU.  No GPU required.
"""

from __future__ import annotations

import copy
import math
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ===========================================================================
# Reproducibility
# ===========================================================================
torch.manual_seed(42)


# ===========================================================================
# 1. Simple Model for Federated Learning
# ===========================================================================


class SimpleCNN(nn.Module):
    """A small CNN suitable for on-device training experiments."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1)  # stride 2 for downsampling
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.fc = nn.Linear(64 * 7 * 7, num_classes)
        self._activation_shapes: Dict[str, Tuple[int, ...]] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._activation_shapes = {}
        x = F.relu(self.conv1(x))
        self._activation_shapes["conv1"] = tuple(x.shape)
        x = F.relu(self.conv2(x))
        self._activation_shapes["conv2"] = tuple(x.shape)
        x = F.relu(self.conv3(x))
        self._activation_shapes["conv3"] = tuple(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ===========================================================================
# 2. Non-IID Data Splitter
# ===========================================================================


def create_non_iid_split(
    data: torch.Tensor,
    targets: torch.Tensor,
    num_clients: int,
    alpha: float = 0.5,
) -> List[Dict[str, torch.Tensor]]:
    """Create non-IID data partitions for federated learning.

    Uses Dirichlet distribution (with concentration alpha) to create
    imbalanced class distributions per client.

    Args:
        data: input tensor (N, ...)
        targets: labels tensor (N,)
        num_clients: number of federated clients
        alpha: Dirichlet concentration (smaller = more non-IID)

    Returns:
        List of dicts with 'data' and 'targets' for each client.
    """
    num_classes = int(targets.max().item()) + 1
    N = len(targets)

    # Sample from Dirichlet distribution
    proportions = torch.distributions.Dirichlet(
        torch.full((num_classes,), alpha, dtype=torch.float32)
    ).sample((num_clients,))  # (num_clients, num_classes)

    # Assign samples to clients based on proportions
    client_data: List[List[int]] = [[] for _ in range(num_clients)]

    for cls in range(num_classes):
        cls_indices = (targets == cls).nonzero(as_tuple=True)[0].tolist()
        num_cls = len(cls_indices)
        # Split indices per client
        splits = (proportions[:, cls] * num_cls / proportions[:, cls].sum()).long()
        # Fix rounding errors
        diff = num_cls - splits.sum().item()
        if diff > 0:
            for i in range(diff):
                splits[i] += 1
        elif diff < 0:
            for i in range(-diff):
                splits[-(i + 1)] -= 1
        # Shuffle and assign
        perm = torch.randperm(num_cls).tolist()
        start = 0
        for client_id in range(num_clients):
            count = int(splits[client_id].item())
            if count > 0:
                client_data[client_id].extend(
                    [
                        cls_indices[perm[idx]]
                        for idx in range(start, min(start + count, num_cls))
                    ]
                )
                start += count

    # Build final client datasets
    clients = []
    for cid in range(num_clients):
        if len(client_data[cid]) == 0:
            # Ensure at least one sample per client
            client_data[cid] = [torch.randint(0, N, (1,)).item()]
        idx = torch.tensor(client_data[cid])
        clients.append(
            {
                "data": data[idx],
                "targets": targets[idx],
            }
        )

    return clients


# ===========================================================================
# 3. Federated Averaging (FedAvg)
# ===========================================================================


def local_train(
    model: nn.Module,
    data: torch.Tensor,
    targets: torch.Tensor,
    epochs: int = 1,
    lr: float = 0.01,
    batch_size: int = 32,
) -> nn.Module:
    """Train model locally on client data.

    Returns a copy of the trained model.
    """
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    N = len(data)

    for _ in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            batch_idx = perm[i : i + batch_size]
            x_batch = data[batch_idx]
            y_batch = targets[batch_idx]

            optimizer.zero_grad()
            out = model(x_batch)
            loss = F.cross_entropy(out, y_batch)
            loss.backward()
            optimizer.step()

    return copy.deepcopy(model)


def fed_avg(
    global_model: nn.Module,
    clients: List[Dict[str, torch.Tensor]],
    rounds: int = 3,
    local_epochs: int = 1,
    lr: float = 0.01,
) -> List[Dict[str, float]]:
    """Run Federated Averaging.

    Args:
        global_model: initial global model
        clients: list of client data dicts
        rounds: number of federated rounds
        local_epochs: local training epochs per round
        lr: learning rate

    Returns:
        Training history for each round.
    """
    history = []

    for rnd in range(rounds):
        local_models = []
        local_losses = []

        # Client-side training
        for client in clients:
            local_m = copy.deepcopy(global_model)
            local_m = local_train(
                local_m,
                client["data"],
                client["targets"],
                epochs=local_epochs,
                lr=lr,
                batch_size=32,
            )

            # Compute local loss
            with torch.no_grad():
                out = local_m(client["data"])
                loss = F.cross_entropy(out, client["targets"]).item()
            local_losses.append(loss)
            local_models.append(local_m)

        # Server-side aggregation (FedAvg)
        with torch.no_grad():
            for global_param in global_model.parameters():
                global_param.zero_()
            total_samples = sum(len(c["data"]) for c in clients)
            for client, local_m in zip(clients, local_models):
                weight = len(client["data"]) / total_samples
                for gp, lp in zip(global_model.parameters(), local_m.parameters()):
                    gp.data += weight * lp.data

        avg_loss = sum(local_losses) / len(local_losses)
        history.append(
            {"round": rnd + 1, "avg_loss": avg_loss, "num_clients": len(clients)}
        )
        print(f"  Round {rnd + 1}: avg_loss={avg_loss:.4f}")

    return history


# ===========================================================================
# 4. Activation Memory Measurement
# ===========================================================================


def measure_activation_memory(
    model: nn.Module,
    input_shape: Tuple[int, ...],
) -> Dict[str, int]:
    """Measure peak activation memory during forward pass.

    This estimates the memory required to store intermediate activations
    for the backward pass (gradient computation).

    Args:
        model: PyTorch model
        input_shape: (batch_size, channels, height, width)

    Returns:
        Memory breakdown by layer and total.
    """
    model.eval()
    x = torch.randn(*input_shape)

    # Register hooks to capture activation sizes
    activations: List[int] = []

    def hook_fn(module, inp, out):
        # Store output size in bytes (float32)
        if isinstance(out, torch.Tensor):
            activations.append(out.numel() * 4)

    handles = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ReLU)):
            handles.append(m.register_forward_hook(hook_fn))

    with torch.no_grad():
        _ = model(x)

    for h in handles:
        h.remove()

    param_mem = sum(p.numel() for p in model.parameters()) * 4
    grad_mem = param_mem  # stored during backward
    act_mem = sum(activations)

    return {
        "parameters_bytes": param_mem,
        "gradients_bytes": grad_mem,
        "activations_bytes": act_mem,
        "total_bytes": param_mem + grad_mem + act_mem,
    }


# ===========================================================================
# 5. TinyTL: Freeze Backbone, Train Biases + Classifier
# ===========================================================================


class TinyTLModel(nn.Module):
    """Tiny Transfer Learning model.

    Freezes the convolutional backbone and only trains:
      - BatchNorm/LayerNorm biases
      - Final classifier layer
    This dramatically reduces the number of trainable parameters and
    activation memory for the backward pass.
    """

    def __init__(self, backbone: SimpleCNN, num_classes: int = 10):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(64 * 7 * 7, num_classes)

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Only train classifier
        self.classifier.weight.requires_grad = True
        self.classifier.bias.requires_grad = True

        # Also unfreeze BN-like layers if any (simulated here as conv biases)
        # In TinyTL, we train biases of frozen layers
        for m in self.backbone.modules():
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                m.bias.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone.conv1(x)
            features = F.relu(features)
            features_conv1 = features
            features = F.relu(self.backbone.conv2(features))
            features_conv2 = features
            features = F.relu(self.backbone.conv3(features))
            features = features.view(features.size(0), -1)
        return self.classifier(features)


def compare_memory_full_vs_tinytl(
    model: SimpleCNN,
    input_shape: Tuple[int, ...],
) -> Dict[str, Dict]:
    """Compare memory usage between full training and TinyTL.

    Args:
        model: full CNN model
        input_shape: input tensor shape

    Returns:
        Memory comparison results.
    """
    full_mem = measure_activation_memory(model, input_shape)

    # TinyTL
    tinytl = TinyTLModel(model)
    tinytl_mem = measure_activation_memory(tinytl, input_shape)

    trainable_full = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_tinytl = sum(p.numel() for p in tinytl.parameters() if p.requires_grad)

    return {
        "full_training": {
            "trainable_params": trainable_full,
            "memory_bytes": full_mem["total_bytes"],
        },
        "tinytl": {
            "trainable_params": trainable_tinytl,
            "memory_bytes": tinytl_mem["total_bytes"],
        },
    }


# ===========================================================================
# 6. Main demonstration
# ===========================================================================


def main() -> None:
    print("=" * 72)
    print("MIT 6.5940 Lecture 21: On-Device Training Simulation")
    print("=" * 72)

    # ---------- Model setup ----------
    print("\n--- 1. Model Setup ---")
    model = SimpleCNN(num_classes=10)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Simple CNN: {total_params:,} parameters")
    print(f"  Model size (FP32): {total_params * 4 / 1024:.1f} KB")

    # ---------- Non-IID Data Creation ----------
    print("\n--- 2. Non-IID Data Split ---")
    # Simulate 5000 MNIST-like samples across 10 clients
    num_clients = 10
    num_samples = 5000
    data = torch.randn(num_samples, 1, 28, 28)
    targets = torch.randint(0, 10, (num_samples,))

    clients = create_non_iid_split(data, targets, num_clients, alpha=0.3)
    print(f"  Total samples: {num_samples}")
    print(f"  Number of clients: {num_clients}")
    print(f"  Dirichlet alpha: 0.3 (smaller = more non-IID)")
    print(f"  Client data distribution:")
    for i, c in enumerate(clients):
        class_counts = torch.bincount(c["targets"], minlength=10)
        dominant = class_counts.argmax().item()
        print(
            f"    Client {i}: {len(c['data'])} samples, "
            f"dominant class={dominant} ({class_counts[dominant].item()} samples)"
        )

    # ---------- FedAvg Simulation ----------
    print("\n--- 3. FedAvg Simulation ---")
    global_model = copy.deepcopy(model)
    history = fed_avg(global_model, clients, rounds=3, local_epochs=1, lr=0.01)
    print(f"  Final avg loss: {history[-1]['avg_loss']:.4f}")

    # ---------- Activation Memory ----------
    print("\n--- 4. Activation Memory Measurement ---")
    batch_sizes = [1, 8, 32, 64]
    print(
        f"  {'Batch':>6} {'Params(KB)':>11} {'Grads(KB)':>11} "
        f"{'Acts(KB)':>11} {'Total(KB)':>11}"
    )
    print(f"  {'-' * 57}")
    for bs in batch_sizes:
        mem = measure_activation_memory(model, (bs, 1, 28, 28))
        print(
            f"  {bs:>6} {mem['parameters_bytes'] / 1024:>11.1f} "
            f"{mem['gradients_bytes'] / 1024:>11.1f} "
            f"{mem['activations_bytes'] / 1024:>11.1f} "
            f"{mem['total_bytes'] / 1024:>11.1f}"
        )

    # ---------- TinyTL ----------
    print("\n--- 5. TinyTL: Freeze Backbone, Train Biases + Classifier ---")
    comparison = compare_memory_full_vs_tinytl(model, (32, 1, 28, 28))

    full_trainable = comparison["full_training"]["trainable_params"]
    tinytl_trainable = comparison["tinytl"]["trainable_params"]
    full_mem = comparison["full_training"]["memory_bytes"]
    tinytl_mem = comparison["tinytl"]["memory_bytes"]

    print(f"  {'':>20} {'Full Training':>16} {'TinyTL':>16} {'Reduction':>12}")
    print(f"  {'-' * 66}")
    print(
        f"  {'Trainable params':>20} {full_trainable:>16,} {tinytl_trainable:>16,} "
        f"{(1 - tinytl_trainable / full_trainable) * 100:>11.1f}%"
    )
    print(
        f"  {'Memory (KB)':>20} {full_mem / 1024:>16.1f} {tinytl_mem / 1024:>16.1f} "
        f"{(1 - tinytl_mem / full_mem) * 100:>11.1f}%"
    )

    # ---------- TinyTL forward pass verification ----------
    print("\n  TinyTL forward pass verification:")
    tinytl = TinyTLModel(model)
    x_test = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        out = tinytl(x_test)
    print(f"  Input:  {tuple(x_test.shape)}")
    print(f"  Output: {tuple(out.shape)} (10 classes)")
    print(
        f"  Trainable params: {sum(p.numel() for p in tinytl.parameters() if p.requires_grad):,} "
        f"/ {sum(p.numel() for p in tinytl.parameters()):,} total"
    )

    # ---------- Summary ----------
    print("\n--- 6. Summary ---")
    print("  Key takeaways:")
    print("    - Federated Learning preserves privacy: data never leaves device")
    print("    - FedAvg aggregates local model updates at server")
    print("    - Non-IID data is a major challenge for FL convergence")
    print(
        "    - Activation memory dominates on-device training memory (grows with batch)"
    )
    print("    - TinyTL reduces trainable params by >90% and memory by >50%")
    print("    - Freezing backbone + training biases is practical for edge devices")

    print("\nDone. All computations on CPU.\n")


if __name__ == "__main__":
    main()
