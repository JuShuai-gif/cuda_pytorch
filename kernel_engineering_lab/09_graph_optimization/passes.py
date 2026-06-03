"""
Optimization passes for the computation graph IR.

Industrial context: These are the core of any ML compiler (XLA, TensorRT, torch.compile).
Each pass transforms the graph to be more efficient while preserving semantics.

Pass ordering matters:
1. constant_folding - reduce known values early to expose CSE opportunities
2. cse - remove redundant ops created by graph construction
3. pattern_fusion - fuse sequences into single ops for memory bandwidth savings
4. dead_code_elimination - clean up unreachable nodes from all previous passes

Multiple iterations may be needed until a fixed point is reached.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch

from ir import (
    DType,
    Graph,
    Node,
    OpType,
    TensorShape,
    node_signature,
    reconcile_outputs,
)


def dead_code_elimination(graph: Graph) -> Graph:
    """Remove nodes not reachable from outputs.

    Industrial: TF/PyTorch graphs accumulate dead nodes from gradient computation,
    constant creation, and intermediate values no longer needed. This pass reduces
    graph size and eliminates unnecessary computation.
    """
    reachable: set[int] = set()

    # BFS backwards from outputs
    stack: list[int] = list(graph.outputs)
    while stack:
        nid = stack.pop()
        if nid in reachable or nid not in graph.nodes:
            continue
        reachable.add(nid)
        node = graph.nodes[nid]
        for inp in node.inputs:
            stack.append(inp)

    # Remove unreachable nodes
    to_remove = set(graph.nodes.keys()) - reachable
    for nid in to_remove:
        del graph.nodes[nid]
        if nid in graph.inputs:
            graph.inputs = [i for i in graph.inputs if i != nid]
        if nid in graph.outputs:
            graph.outputs = [o for o in graph.outputs if o != nid]

    # Clean up outputs list references to removed nodes
    for node in graph.nodes.values():
        node.outputs = [out for out in node.outputs if out in graph.nodes]

    return graph


def constant_folding(graph: Graph) -> Graph:
    """Evaluate constant sub-expressions at graph optimization time.

    Industrial: Constants from config/conversion can be folded early, reducing
    runtime work. Supports: ADD, SUB, MUL, DIV on constant tensors.
    """
    FOLDABLE_OPS = {OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV}

    changed = True
    while changed:
        changed = False
        for node in list(graph.nodes.values()):
            if node.op not in FOLDABLE_OPS:
                continue
            if len(node.inputs) < 2:
                continue

            # Check if all inputs are CONSTANT nodes
            input_nodes = [graph.nodes.get(inp) for inp in node.inputs]
            if any(n is None for n in input_nodes):
                continue
            if not all(n.op == OpType.CONSTANT for n in input_nodes):
                continue

            const_vals = []
            for n in input_nodes:
                val = n.attrs.get("value")
                if not isinstance(val, torch.Tensor):
                    break
                const_vals.append(val)
            if len(const_vals) != len(input_nodes):
                continue

            # Compute folded value
            try:
                if node.op == OpType.ADD:
                    result = const_vals[0] + const_vals[1]
                elif node.op == OpType.SUB:
                    result = const_vals[0] - const_vals[1]
                elif node.op == OpType.MUL:
                    result = const_vals[0] * const_vals[1]
                elif node.op == OpType.DIV:
                    result = const_vals[0] / const_vals[1]
                else:
                    continue
            except Exception:
                continue

            # Replace this node with a CONSTANT
            node.op = OpType.CONSTANT
            node.inputs = []
            node.attrs = {"value": result}
            node.name = f"folded_const_{node.id}"
            changed = True

    # After folding, clean up disconnected constant nodes
    dead_code_elimination(graph)
    return graph


def common_subexpression_elimination(graph: Graph) -> Graph:
    """CSE: identical nodes with same op and inputs -> reuse.

    Industrial: especially common after inlining / graph construction where
    multiple graph regions produce the same intermediate value independently.
    """
    sig_to_nodes: dict[tuple, list[int]] = defaultdict(list)

    for node in graph.nodes.values():
        sig = node_signature(node)
        sig_to_nodes[sig].append(node.id)

    replacements: dict[int, int] = {}

    for sig, node_ids in sig_to_nodes.items():
        if len(node_ids) <= 1:
            continue
        canonical = node_ids[0]
        for dup_id in node_ids[1:]:
            replacements[dup_id] = canonical

    # Redirect all consumers of duplicates to the canonical node
    for dup_id, canon_id in replacements.items():
        if dup_id in graph.nodes:
            dup_node = graph.nodes[dup_id]
            for consumer in list(dup_node.outputs):
                if consumer in graph.nodes:
                    cons_node = graph.nodes[consumer]
                    cons_node.inputs = [
                        canon_id if inp == dup_id else inp for inp in cons_node.inputs
                    ]
                    # Add output edge from canonical to consumer
                    if consumer not in graph.nodes[canon_id].outputs:
                        graph.nodes[canon_id].outputs.append(consumer)

            # Remove from graph outputs if it was an output
            if dup_id in graph.outputs:
                graph.outputs = [canon_id if o == dup_id else o for o in graph.outputs]

            del graph.nodes[dup_id]
            if dup_id in graph.inputs:
                graph.inputs = [i for i in graph.inputs if i != dup_id]

    # Remove duplicate edges in outputs lists
    for node in graph.nodes.values():
        node.outputs = list(dict.fromkeys(node.outputs))

    return graph


def pattern_fusion(graph: Graph) -> Graph:
    """Fuse known patterns into fused ops.

    Industrial: this is the core of TensorRT/inductor fusion.
    Patterns:
    - ADD + RELU -> FUSED_ADD_RELU
    - ADD + GELU -> FUSED_BIAS_GELU
    - ADD + RMSNORM -> FUSED_RESIDUAL_RMSNORM
    """
    patterns = [
        (OpType.ADD, OpType.RELU, OpType.FUSED_ADD_RELU),
        (OpType.ADD, OpType.GELU, OpType.FUSED_BIAS_GELU),
        (OpType.ADD, OpType.RMSNORM, OpType.FUSED_RESIDUAL_RMSNORM),
    ]

    changed = True
    iteration = 0
    max_iterations = 20

    while changed and iteration < max_iterations:
        changed = False
        iteration += 1

        for node in list(graph.nodes.values()):
            for op1, op2, fused_op in patterns:
                if node.op != op1:
                    continue
                if len(node.outputs) != 1:
                    continue

                consumer_id = node.outputs[0]
                if consumer_id not in graph.nodes:
                    continue
                consumer = graph.nodes[consumer_id]

                if consumer.op != op2:
                    continue
                # Must be the only consumer of node's output
                # (otherwise fusing would break other consumers)
                consumer_inputs = [graph.nodes.get(i) for i in consumer.inputs]
                input_from_node = sum(1 for i in consumer.inputs if i == node.id)

                if input_from_node != 1:
                    # Consumer takes the node's output as input somewhere
                    # but we need it to be a direct sequential chain
                    if consumer.inputs.count(node.id) == 0:
                        continue

                # Fuse: merge both nodes into one
                # Collect inputs: all from node (except we don't need duplicates) +
                # additional inputs from consumer (excluding node.id)
                fused_inputs: list[int] = []
                for inp in node.inputs:
                    if inp not in fused_inputs:
                        fused_inputs.append(inp)
                for inp in consumer.inputs:
                    if inp != node.id and inp not in fused_inputs:
                        fused_inputs.append(inp)

                # Merge attrs
                fused_attrs: dict[str, Any] = {}
                fused_attrs.update(node.attrs)
                fused_attrs.update(consumer.attrs)

                # Shape: use consumer's shape (output shape of the fusion)
                fused_shape = consumer.shape

                # Create the fused node by reusing consumer ID
                consumer.op = fused_op
                consumer.inputs = fused_inputs
                consumer.attrs = fused_attrs
                consumer.shape = fused_shape
                consumer.name = f"fused_{fused_op.value}_{consumer.id}"

                # Remove the add node
                add_outputs = list(node.outputs)
                del graph.nodes[node.id]
                if node.id in graph.inputs:
                    graph.inputs = [i for i in graph.inputs if i != node.id]

                # Update references: consumer no longer takes node.id as input
                consumer.inputs = [i for i in consumer.inputs if i != node.id]

                # Rebuild outputs
                for n in graph.nodes.values():
                    n.outputs = [o for o in n.outputs if o != node.id and o in graph.nodes]

                changed = True
                break  # Process one fusion at a time, then rescan
            if changed:
                break

    if iteration >= max_iterations and changed:
        # If we hit max iterations, we may have an infinite loop of fusions
        pass

    return graph


def optimize_graph(graph: Graph, max_iterations: int = 5) -> Graph:
    """Run all passes in order. Returns optimized graph.

    Multiple iterations until fixed point (no further changes).
    """
    result = graph.clone()

    for iteration in range(max_iterations):
        nodes_before = len(result.nodes)

        result = constant_folding(result)
        reconcile_outputs(result)

        result = common_subexpression_elimination(result)
        reconcile_outputs(result)

        result = pattern_fusion(result)
        reconcile_outputs(result)

        result = dead_code_elimination(result)
        reconcile_outputs(result)

        nodes_after = len(result.nodes)
        if nodes_before == nodes_after:
            break  # Fixed point reached

    return result
