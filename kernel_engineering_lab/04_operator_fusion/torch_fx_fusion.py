"""
Demonstrate torch.fx-based operator fusion.

Defines a simple model with sequential operations, traces it with torch.fx,
and applies custom fusion passes for add+relu and bias+gelu.

This shows how torch.compile and torch.inductor perform fusion automatically,
and how you can write custom fusion passes.
"""

from __future__ import annotations

import operator
from typing import Any

import torch
import torch.fx


class SimpleModel(torch.nn.Module):
    """A simple sequential model: Linear -> ReLU -> Linear -> GELU."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear1 -> Add bias -> ReLU
        h = self.linear1(x)
        h = torch.nn.functional.relu(h)
        # Linear2 -> Add bias -> GELU
        h = self.linear2(h)
        h = torch.nn.functional.gelu(h, approximate="tanh")
        return h


def fuse_add_relu_pass(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Custom FX pass that fuses add + relu patterns.

    Looks for the pattern:
      %intermediate = call_function(operator.add, %x, %bias)
      %output = call_function(torch.nn.functional.relu, %intermediate)

    And replaces with a fused call.
    """
    graph = gm.graph
    changed = True

    while changed:
        changed = False
        for node in list(graph.nodes):
            if node.op == "call_function" and node.target is operator.add and len(node.users) == 1:
                user = list(node.users)[0]
                if user.op == "call_function" and user.target is torch.nn.functional.relu:
                    # Found the pattern: add -> relu
                    # Replace with a custom fused node
                    with graph.inserting_before(user):
                        fused_node = graph.call_function(
                            _fused_add_relu_marker,
                            args=node.args,
                        )
                    user.replace_all_uses_with(fused_node)
                    graph.erase_node(user)
                    graph.erase_node(node)
                    changed = True
                    break  # Restart after each fusion

    graph.lint()
    return gm


def _fused_add_relu_marker(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Marker function for fused add+relu. When executed, performs actual fusion."""
    return torch.nn.functional.relu(x + bias)


def _fused_bias_gelu_marker(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Marker function for fused bias+gelu. When executed, performs actual fusion."""
    return torch.nn.functional.gelu(x + bias, approximate="tanh")


def fuse_bias_gelu_pass(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Custom FX pass that fuses add + gelu patterns."""
    graph = gm.graph
    changed = True

    while changed:
        changed = False
        for node in list(graph.nodes):
            if node.op == "call_function" and node.target is operator.add and len(node.users) == 1:
                user = list(node.users)[0]
                if user.op == "call_function" and user.target is torch.nn.functional.gelu:
                    with graph.inserting_before(user):
                        fused_node = graph.call_function(
                            _fused_bias_gelu_marker,
                            args=node.args,
                        )
                    user.replace_all_uses_with(fused_node)
                    graph.erase_node(user)
                    graph.erase_node(node)
                    changed = True
                    break

    graph.lint()
    return gm


def trace_and_fuse(model: torch.nn.Module, sample_input: torch.Tensor) -> tuple[Any, Any, Any]:
    """Trace model with torch.fx and apply fusion passes.

    Returns:
        (original_graph_module, fused_graph_module, output_original)
    """
    # Trace the model
    gm = torch.fx.symbolic_trace(model)

    # Save original graph string
    original_graph_str = str(gm.graph)

    # Apply fusion passes
    gm = fuse_add_relu_pass(gm)
    gm = fuse_bias_gelu_pass(gm)
    gm.recompile()

    fused_graph_str = str(gm.graph)

    return original_graph_str, fused_graph_str, gm


def analyze_graph(graph_module: torch.fx.GraphModule) -> dict[str, int]:
    """Count operations by type in a torch.fx graph."""
    counts: dict[str, int] = {}
    for node in graph_module.graph.nodes:
        if node.op == "call_function":
            name = str(node.target).split("'")[1] if "'" in str(node.target) else str(node.target)
            if "built-in function" in name:
                name = name.replace("<built-in function ", "").replace(">", "")
            counts[name] = counts.get(name, 0) + 1
        elif node.op == "call_module":
            module_type = type(graph_module.get_submodule(node.target)).__name__
            key = f"module:{module_type}"
            counts[key] = counts.get(key, 0) + 1
    return counts


def main():
    """Demonstrate torch.fx fusion."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_dim = 512

    model = SimpleModel(hidden_dim=hidden_dim).to(device)
    sample_input = torch.randn(8, hidden_dim, device=device)

    original_str, fused_str, fused_gm = trace_and_fuse(model, sample_input)

    print("=" * 70)
    print("  TORCH.FX GRAPH FUSION DEMO")
    print("=" * 70)

    print(f"\n  Original Graph:")
    print(f"  {'-' * 50}")
    for line in original_str.strip().split("\n"):
        print(f"    {line}")

    print(f"\n  Fused Graph (after add+relu, bias+gelu fusion):")
    print(f"  {'-' * 50}")
    for line in fused_str.strip().split("\n"):
        print(f"    {line}")

    # Run both and compare
    output_orig = model(sample_input)
    output_fused = fused_gm(sample_input)
    err = (output_orig - output_fused).abs().max().item()

    print(f"\n  Output comparison:")
    print(f"    Max error (original vs fused): {err:.2e}")
    print(f"    Outputs match: {torch.allclose(output_orig, output_fused, atol=1e-5)}")

    # Show op counts
    orig_gm = torch.fx.symbolic_trace(model)
    orig_counts = analyze_graph(orig_gm)
    fused_counts = analyze_graph(fused_gm)

    print(f"\n  Operation counts:")
    print(f"    Original: {orig_counts}")
    print(f"    Fused:    {fused_counts}")

    print(f"\n  Note: This is a demonstration of the torch.fx fusion API.")
    print(f"  In production, torch.compile + inductor handle this automatically")
    print(f"  and much more aggressively than simple pattern matching.")


if __name__ == "__main__":
    main()
