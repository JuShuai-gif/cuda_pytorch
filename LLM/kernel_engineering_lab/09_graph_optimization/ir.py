"""
Minimal computation graph IR for graph-level optimizations.

Industrial context: This is how XLA HLO, TensorRT, torch.fx, and MLIR represent
programs. A computation graph is a DAG of operations with edges representing data flow.

Key design decisions:
- Node IDs are integers for fast hashing/comparison
- Inputs/outputs are lists of node IDs (edges)
- Attrs dict for op-specific parameters
- Symbolic shapes (str dims) for dynamic shape support
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Sequence

import torch


class DType(Enum):
    FLOAT32 = auto()
    FLOAT16 = auto()
    BFLOAT16 = auto()

    def to_torch(self) -> torch.dtype:
        mapping = {
            DType.FLOAT32: torch.float32,
            DType.FLOAT16: torch.float16,
            DType.BFLOAT16: torch.bfloat16,
        }
        return mapping[self]


@dataclass
class TensorShape:
    dims: list[int | str]

    def __repr__(self) -> str:
        dim_strs = [str(d) for d in self.dims]
        return f"({', '.join(dim_strs)})"


class OpType(Enum):
    CONSTANT = "constant"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    EXP = "exp"
    LOG = "log"
    RESHAPE = "reshape"
    TRANSPOSE = "transpose"
    MATMUL = "matmul"
    SOFTMAX = "softmax"
    REDUCE_SUM = "reduce_sum"
    LAYERNORM = "layernorm"
    RMSNORM = "rmsnorm"
    FUSED_ADD_RELU = "fused_add_relu"
    FUSED_BIAS_GELU = "fused_bias_gelu"
    FUSED_RESIDUAL_RMSNORM = "fused_residual_rmsnorm"


@dataclass
class Node:
    id: int
    op: OpType
    inputs: list[int]
    outputs: list[int]
    attrs: dict[str, Any]
    shape: TensorShape | None = None
    dtype: DType = DType.FLOAT32
    name: str = ""

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.id == other.id


@dataclass
class Graph:
    nodes: dict[int, Node] = field(default_factory=dict)
    inputs: list[int] = field(default_factory=list)
    outputs: list[int] = field(default_factory=list)
    _next_id: int = field(default=0, repr=False, init=False)

    def add_node(
        self,
        op: OpType,
        inputs: Sequence[int],
        attrs: dict[str, Any] | None = None,
        shape: TensorShape | None = None,
        dtype: DType = DType.FLOAT32,
        name: str = "",
    ) -> int:
        node_id = self._next_id
        self._next_id += 1

        node = Node(
            id=node_id,
            op=op,
            inputs=list(inputs),
            outputs=[],
            attrs=attrs if attrs is not None else {},
            shape=shape,
            dtype=dtype,
            name=name or f"{op.value}_{node_id}",
        )
        self.nodes[node_id] = node

        for inp_id in inputs:
            if inp_id in self.nodes:
                self.nodes[inp_id].outputs.append(node_id)

        return node_id

    def _get_unsafe_rename_id(self, target_id: int) -> int:
        """Get the next available ID that a clone operation can use."""
        return target_id

    def topological_sort(self) -> list[int]:
        indegree: dict[int, int] = {}
        for nid in self.nodes:
            indegree[nid] = 0
        for node in self.nodes.values():
            for out_id in node.outputs:
                indegree[out_id] = indegree.get(out_id, 0) + 1

        queue: deque[int] = deque()
        for nid in self.inputs:
            indegree[nid] = indegree.get(nid, 0)
            queue.append(nid)
        for nid, deg in indegree.items():
            if deg == 0 and nid not in self.inputs:
                queue.append(nid)

        order: list[int] = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            node = self.nodes[nid]
            for out_id in node.outputs:
                indegree[out_id] -= 1
                if indegree[out_id] == 0:
                    queue.append(out_id)

        if len(order) != len(self.nodes):
            missing = set(self.nodes.keys()) - set(order)
            raise ValueError(
                f"Graph has cycles or disconnected components. Unreachable nodes: {missing}"
            )

        return order

    def clone(self) -> Graph:
        new_graph = Graph()
        id_map: dict[int, int] = {}

        for old_id, node in self.nodes.items():
            new_id = new_graph.add_node(
                op=node.op,
                inputs=[],  # will be patched after all nodes exist
                attrs=dict(node.attrs),
                shape=node.shape,
                dtype=node.dtype,
                name=node.name,
            )
            id_map[old_id] = new_id

        # Patch input edges
        for old_id, new_id in id_map.items():
            old_node = self.nodes[old_id]
            new_node = new_graph.nodes[new_id]
            new_node.inputs = [id_map[inp] for inp in old_node.inputs]
            # Rebuild outputs list
            new_node.outputs = [id_map[out] for out in old_node.outputs if out in id_map]

        new_graph.inputs = [id_map[i] for i in self.inputs]
        new_graph.outputs = [id_map[o] for o in self.outputs]
        new_graph._next_id = max(id_map.values()) + 1 if id_map else 0

        return new_graph

    def to_dot(self) -> str:
        lines = ["digraph G {", '  rankdir="TB";', "  node [shape=box, style=rounded];"]

        # Mark inputs and outputs
        for nid in self.inputs:
            node = self.nodes.get(nid)
            label = node.name if node else f"input_{nid}"
            lines.append(
                f'  {nid} [label="Input: {label}", fillcolor="#d0f0d0", style="filled,rounded"];'
            )
        for nid in self.outputs:
            node = self.nodes.get(nid)
            label = node.name if node else f"output_{nid}"
            lines.append(
                f'  {nid} [label="Output: {label}", fillcolor="#f0d0d0", style="filled,rounded"];'
            )

        for node in self.nodes.values():
            if node.id in self.inputs or node.id in self.outputs:
                continue
            attrs_str = ""
            if node.attrs:
                items = [f"{k}={v}" for k, v in sorted(node.attrs.items())]
                attrs_str = "\\n" + ", ".join(items)
            shape_str = f"\\n{node.shape}" if node.shape else ""
            label = f"{node.name}\\nop={node.op.value}{attrs_str}{shape_str}"
            lines.append(f'  {node.id} [label="{label}"];')

        for node in self.nodes.values():
            for out_id in node.outputs:
                if out_id in self.nodes:
                    lines.append(f"  {node.id} -> {out_id};")

        lines.append("}")
        return "\n".join(lines)

    def validate(self) -> bool:
        if not self.nodes:
            return True

        # Every output node must exist in the graph
        for nid in self.outputs:
            if nid not in self.nodes:
                raise ValueError(f"Output node {nid} not in graph")

        # Every input node must exist in the graph
        for nid in self.inputs:
            if nid not in self.nodes:
                raise ValueError(f"Input node {nid} not in graph")

        # Every node's inputs must exist
        for node in self.nodes.values():
            for inp_id in node.inputs:
                if inp_id not in self.nodes:
                    raise ValueError(
                        f"Node {node.id} ({node.name}) references missing input {inp_id}"
                    )

        # Check for cycles via topological sort
        try:
            self.topological_sort()
        except ValueError as e:
            raise ValueError(f"Graph validation failed: {e}") from e

        # Check output consistency: every node's outputs list must match
        # the inputs of other nodes that reference it
        for node in self.nodes.values():
            for out_id in node.outputs:
                if node.id not in self.nodes[out_id].inputs:
                    raise ValueError(
                        f"Output consistency error: {node.id} lists {out_id} "
                        f"as output, but {out_id} does not list {node.id} as input"
                    )

        return True


def reconcile_outputs(graph: Graph) -> None:
    """Rebuild each node's outputs list from the inputs of all other nodes."""
    for node in graph.nodes.values():
        node.outputs = []
    for node in graph.nodes.values():
        for inp_id in node.inputs:
            if inp_id in graph.nodes:
                graph.nodes[inp_id].outputs.append(node.id)


def node_signature(node: Node) -> tuple:
    """Create a hashable signature for a node to detect duplicates (CSE).

    Two nodes are equivalent if they have:
    - Same op type
    - Same inputs (in order)
    - Same attrs (excluding transitive shape/dtype fields)

    CONSTANT nodes with no stored value (no attrs) are input placeholders
    and must never be merged.
    """
    if node.op == OpType.CONSTANT:
        val = node.attrs.get("value")
        if val is None and not node.attrs:
            return (node.op, node.id, node.name)

    relevant_attrs = tuple(sorted((k, v) for k, v in node.attrs.items()))
    return (node.op, tuple(node.inputs), relevant_attrs)
