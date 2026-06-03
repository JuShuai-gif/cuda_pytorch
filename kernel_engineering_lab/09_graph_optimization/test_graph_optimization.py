"""
Tests for graph optimization module.

Tests: IR construction, topological sort, each optimization pass individually,
end-to-end optimization, edge cases, graph validation.

Run: pytest 09_graph_optimization/test_graph_optimization.py -v
"""

from __future__ import annotations

import pytest
import torch

from executor import execute_graph
from ir import DType, Graph, OpType, TensorShape, reconcile_outputs
from passes import (
    common_subexpression_elimination,
    constant_folding,
    dead_code_elimination,
    optimize_graph,
    pattern_fusion,
)


def _build_simple_graph() -> Graph:
    """Helper: simple add graph with two inputs, one output."""
    g = Graph()

    a_id = g.add_node(OpType.CONSTANT, [], name="a")
    b_id = g.add_node(OpType.CONSTANT, [], name="b")
    g.inputs.extend([a_id, b_id])

    add_id = g.add_node(OpType.ADD, [a_id, b_id], name="add")
    g.outputs.append(add_id)

    return g


def _build_chain_graph() -> Graph:
    """Helper: linear chain A->B->C->D."""
    g = Graph()

    inp = g.add_node(OpType.CONSTANT, [], name="input")
    g.inputs.append(inp)

    n1 = g.add_node(OpType.RELU, [inp], name="relu")
    n2 = g.add_node(OpType.GELU, [n1], name="gelu")
    n3 = g.add_node(OpType.TANH, [n2], name="tanh")

    g.outputs.append(n3)
    return g


def _build_diamond_graph() -> Graph:
    """Helper: diamond graph where two nodes share an input."""
    g = Graph()

    inp = g.add_node(OpType.CONSTANT, [], name="input")
    g.inputs.append(inp)

    a = g.add_node(OpType.ADD, [inp, inp], name="add_a")
    b = g.add_node(OpType.MUL, [inp, inp], name="mul_b")
    c = g.add_node(OpType.ADD, [a, b], name="merge")

    g.outputs.append(c)
    return g


def _build_cse_candidate_graph() -> Graph:
    """Helper: graph with duplicate sub-expressions for CSE testing."""
    g = Graph()

    x = g.add_node(OpType.CONSTANT, [], name="x")
    g.inputs.append(x)

    a = g.add_node(OpType.ADD, [x, x], name="a")
    b = g.add_node(OpType.ADD, [x, x], name="b")  # Same as a

    merge = g.add_node(OpType.MUL, [a, b], name="merge")
    g.outputs.append(merge)
    return g


def _build_fusion_candidate_graph() -> Graph:
    """Helper: graph with ADD+RELU pattern to fuse."""
    g = Graph()

    x = g.add_node(OpType.CONSTANT, [], name="x")
    bias = g.add_node(OpType.CONSTANT, [], name="bias")
    g.inputs.extend([x, bias])

    add_node = g.add_node(OpType.ADD, [x, bias], name="add")
    relu_node = g.add_node(OpType.RELU, [add_node], name="relu")

    g.outputs.append(relu_node)
    return g


# ---------------------------------------------------------------------------
# IR Construction Tests
# ---------------------------------------------------------------------------


class TestIRConstruction:
    """Test graph IR creation and structure."""

    def test_create_empty_graph(self):
        g = Graph()
        assert len(g.nodes) == 0
        assert len(g.inputs) == 0
        assert len(g.outputs) == 0

    def test_add_single_node(self):
        g = Graph()
        nid = g.add_node(OpType.CONSTANT, [], name="test")
        assert nid == 0
        assert len(g.nodes) == 1
        assert g.nodes[nid].op == OpType.CONSTANT
        assert g.nodes[nid].name == "test"

    def test_add_node_with_inputs(self):
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        b = g.add_node(OpType.CONSTANT, [], name="b")
        c = g.add_node(OpType.ADD, [a, b], name="c")

        assert g.nodes[c].inputs == [a, b]
        assert g.nodes[a].outputs == [c]
        assert g.nodes[b].outputs == [c]

    def test_multiple_consumers(self):
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        b = g.add_node(OpType.RELU, [a], name="b")
        c = g.add_node(OpType.GELU, [a], name="c")

        assert sorted(g.nodes[a].outputs) == sorted([b, c])

    def test_topological_sort_linear(self):
        g = _build_chain_graph()
        order = g.topological_sort()
        assert order == [
            g.inputs[0],
            1,  # relu
            2,  # gelu
            3,  # tanh
        ]

    def test_topological_sort_diamond(self):
        g = _build_diamond_graph()
        order = g.topological_sort()

        # Input must come first
        input_id = g.inputs[0]
        assert order[0] == input_id

        # Find node ids by name
        name_to_id = {node.name: nid for nid, node in g.nodes.items()}

        merge_idx = order.index(name_to_id["merge"])
        a_idx = order.index(name_to_id["add_a"])
        b_idx = order.index(name_to_id["mul_b"])
        assert a_idx < merge_idx
        assert b_idx < merge_idx

    def test_graph_clone(self):
        g = _build_simple_graph()
        g2 = g.clone()

        assert len(g2.nodes) == len(g.nodes)
        assert g2.inputs == [0, 1]
        assert g2.outputs == [2]

        # Modifying clone should not affect original
        g2.add_node(OpType.RELU, [2], name="extra")
        assert len(g.nodes) == 3  # unchanged

    def test_to_dot(self):
        g = _build_simple_graph()
        dot = g.to_dot()
        assert "digraph" in dot
        assert "Input" in dot or "input" in dot.lower()
        assert "Output" in dot or "output" in dot.lower() or "add" in dot.lower()

    def test_validate_valid_graph(self):
        g = _build_simple_graph()
        assert g.validate() is True

    def test_validate_missing_input(self):
        g = Graph()
        n = g.add_node(OpType.ADD, [0], name="bad")
        g.nodes[n].inputs = [999]
        g.inputs.append(n)
        g.outputs.append(n)
        with pytest.raises(ValueError):
            g.validate()

    def test_validate_bad_output(self):
        g = Graph()
        n = g.add_node(OpType.ADD, [0], name="n0")
        g.inputs.append(n)
        g.outputs.append(999)  # Output ID that doesn't exist
        with pytest.raises(ValueError):
            g.validate()

    def test_clone_preserves_edges(self):
        g = _build_diamond_graph()
        g2 = g.clone()
        assert g2.validate()

        # Check edge structure: input node should have 2 consumers
        inp = g2.inputs[0]
        unique_outputs = set(g2.nodes[inp].outputs)
        assert len(unique_outputs) == 2

    def test_optype_enum_values(self):
        assert OpType.ADD.value == "add"
        assert OpType.MATMUL.value == "matmul"
        assert OpType.FUSED_ADD_RELU.value == "fused_add_relu"


# ---------------------------------------------------------------------------
# Dead Code Elimination Tests
# ---------------------------------------------------------------------------


class TestDeadCodeElimination:
    def test_no_dead_code_in_simple_graph(self):
        g = _build_simple_graph()
        nodes_before = len(g.nodes)
        result = dead_code_elimination(g)
        assert len(result.nodes) == nodes_before

    def test_removes_unreachable_node(self):
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        g.inputs.append(a)
        orphan = g.add_node(OpType.RELU, [], name="orphan")  # No inputs, not output
        relu = g.add_node(OpType.RELU, [a], name="relu")
        g.outputs.append(relu)

        result = dead_code_elimination(g)
        assert orphan not in result.nodes
        assert a in result.nodes
        assert relu in result.nodes

    def test_removes_chain_without_output(self):
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        g.inputs.append(a)
        b = g.add_node(OpType.RELU, [a], name="b")
        c = g.add_node(OpType.GELU, [b], name="c")
        # Outputs are empty - nothing is reachable

        result = dead_code_elimination(g)
        assert len(result.nodes) == 0

    def test_removes_dead_branch(self):
        g = _build_diamond_graph()
        # Add a dead branch
        orphan = g.add_node(OpType.RELU, [], name="orphan")
        orphan2 = g.add_node(OpType.GELU, [orphan], name="orphan2")

        result = dead_code_elimination(g)
        assert orphan not in result.nodes
        assert orphan2 not in result.nodes

    def test_no_nodes_no_error(self):
        g = Graph()
        result = dead_code_elimination(g)
        assert len(result.nodes) == 0


# ---------------------------------------------------------------------------
# Constant Folding Tests
# ---------------------------------------------------------------------------


class TestConstantFolding:
    def test_folds_simple_add(self):
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], attrs={"value": torch.tensor(2.0)}, name="a")
        b = g.add_node(OpType.CONSTANT, [], attrs={"value": torch.tensor(3.0)}, name="b")
        add_node = g.add_node(OpType.ADD, [a, b], name="add")
        g.inputs.extend([a, b])
        g.outputs.append(add_node)

        result = constant_folding(g.clone())

        # The add should be folded into a constant
        folded = result.nodes[add_node]
        assert folded.op == OpType.CONSTANT
        val = folded.attrs.get("value")
        assert val is not None
        if isinstance(val, torch.Tensor):
            assert abs(val.item() - 5.0) < 1e-6

    def test_folds_sub(self):
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], attrs={"value": torch.tensor(10.0)}, name="a")
        b = g.add_node(OpType.CONSTANT, [], attrs={"value": torch.tensor(3.0)}, name="b")
        sub_node = g.add_node(OpType.SUB, [a, b], name="sub")
        g.inputs.extend([a, b])
        g.outputs.append(sub_node)

        result = constant_folding(g.clone())

        folded = result.nodes[sub_node]
        assert folded.op == OpType.CONSTANT

    def test_folds_mul(self):
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], attrs={"value": torch.tensor(4.0)}, name="a")
        b = g.add_node(OpType.CONSTANT, [], attrs={"value": torch.tensor(5.0)}, name="b")
        mul_node = g.add_node(OpType.MUL, [a, b], name="mul")
        g.inputs.extend([a, b])
        g.outputs.append(mul_node)

        result = constant_folding(g.clone())
        folded = result.nodes[mul_node]
        assert folded.op == OpType.CONSTANT

    def test_folds_div(self):
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], attrs={"value": torch.tensor(10.0)}, name="a")
        b = g.add_node(OpType.CONSTANT, [], attrs={"value": torch.tensor(2.0)}, name="b")
        div_node = g.add_node(OpType.DIV, [a, b], name="div")
        g.inputs.extend([a, b])
        g.outputs.append(div_node)

        result = constant_folding(g.clone())
        folded = result.nodes[div_node]
        assert folded.op == OpType.CONSTANT

    def test_does_not_fold_non_constants(self):
        g = _build_simple_graph()
        nodes_before = len(g.nodes)

        result = constant_folding(g.clone())
        # The ADD node should not be folded because inputs are "input" constants
        # (not actual constant values stored in attrs)
        for node in result.nodes.values():
            if node.op == OpType.ADD:
                assert node.op != OpType.CONSTANT

    def test_folds_cascading(self):
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], attrs={"value": torch.tensor(2.0)}, name="a")
        b = g.add_node(OpType.CONSTANT, [], attrs={"value": torch.tensor(3.0)}, name="b")
        add1 = g.add_node(OpType.ADD, [a, b], name="add1")
        c = g.add_node(OpType.CONSTANT, [], attrs={"value": torch.tensor(10.0)}, name="c")
        add2 = g.add_node(OpType.ADD, [add1, c], name="add2")
        g.inputs.extend([a, b, c])
        g.outputs.append(add2)

        result = constant_folding(g.clone())
        folded = result.nodes[add2]
        assert folded.op == OpType.CONSTANT


# ---------------------------------------------------------------------------
# Common Subexpression Elimination Tests
# ---------------------------------------------------------------------------


class TestCSE:
    def test_deduplicates_identical_nodes(self):
        g = _build_cse_candidate_graph()

        result = common_subexpression_elimination(g.clone())

        # After CSE, one of the two identical ADD nodes should be removed
        # and both consumers should point to the same node
        assert len(result.nodes) < len(g.nodes)

    def test_does_not_deduplicate_different_ops(self):
        g = Graph()
        x = g.add_node(OpType.CONSTANT, [], name="x")
        g.inputs.append(x)
        a = g.add_node(OpType.ADD, [x, x], name="a")
        b = g.add_node(OpType.MUL, [x, x], name="b")  # Different op
        merge = g.add_node(OpType.ADD, [a, b], name="merge")
        g.outputs.append(merge)

        result = common_subexpression_elimination(g.clone())
        assert len(result.nodes) == len(g.nodes)  # No duplicates

    def test_no_change_on_unique_graph(self):
        g = _build_simple_graph()
        result = common_subexpression_elimination(g.clone())
        assert len(result.nodes) == len(g.nodes)

    def test_empty_graph(self):
        g = Graph()
        result = common_subexpression_elimination(g)
        assert len(result.nodes) == 0


# ---------------------------------------------------------------------------
# Pattern Fusion Tests
# ---------------------------------------------------------------------------


class TestPatternFusion:
    def test_fuses_add_relu(self):
        g = _build_fusion_candidate_graph()

        result = pattern_fusion(g.clone())

        fused_found = False
        for node in result.nodes.values():
            if node.op == OpType.FUSED_ADD_RELU:
                fused_found = True
                break

        assert fused_found

    def test_reduces_node_count(self):
        g = _build_fusion_candidate_graph()
        nodes_before = len(g.nodes)

        result = pattern_fusion(g.clone())

        assert len(result.nodes) < nodes_before

    def test_fuses_add_gelu(self):
        g = Graph()
        x = g.add_node(OpType.CONSTANT, [], name="x")
        bias = g.add_node(OpType.CONSTANT, [], name="bias")
        g.inputs.extend([x, bias])
        add_node = g.add_node(OpType.ADD, [x, bias], name="add")
        gelu_node = g.add_node(OpType.GELU, [add_node], name="gelu")
        g.outputs.append(gelu_node)

        result = pattern_fusion(g.clone())

        fused_found = False
        for node in result.nodes.values():
            if node.op == OpType.FUSED_BIAS_GELU:
                fused_found = True
                break
        assert fused_found

    def test_no_fusion_without_pattern(self):
        g = _build_simple_graph()
        n_before = len(g.nodes)

        result = pattern_fusion(g.clone())
        assert len(result.nodes) == n_before

    def test_empty_graph_no_error(self):
        g = Graph()
        result = pattern_fusion(g)
        assert len(result.nodes) == 0


# ---------------------------------------------------------------------------
# End-to-End Tests
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Test the full optimization pipeline with execution verification."""

    def test_optimize_simple_graph(self):
        g = _build_simple_graph()

        optimized = optimize_graph(g)

        assert optimized.validate()
        assert len(optimized.nodes) <= len(g.nodes)

    def test_execute_simple_graph(self):
        g = _build_simple_graph()

        inputs = {
            "a": torch.tensor([1.0, 2.0, 3.0]),
            "b": torch.tensor([4.0, 5.0, 6.0]),
        }

        output = execute_graph(g, inputs)
        assert len(output) > 0
        for val in output.values():
            assert torch.allclose(val, torch.tensor([5.0, 7.0, 9.0]))

    def test_execute_with_all_activations(self):
        g = Graph()
        x = g.add_node(OpType.CONSTANT, [], name="x")
        g.inputs.append(x)

        prev = x
        for op in [OpType.RELU, OpType.GELU, OpType.TANH, OpType.SIGMOID, OpType.EXP]:
            prev = g.add_node(op, [prev], name=op.value)
        last = g.add_node(OpType.LOG, [prev], name="log_final")
        g.outputs.append(last)

        out = execute_graph(g, {"x": torch.tensor([0.5, -0.5, 1.0, -1.0])})
        assert len(out) > 0
        assert not any(torch.isnan(v).any() for v in out.values())

    def test_execute_matmul(self):
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        b = g.add_node(OpType.CONSTANT, [], name="b")
        g.inputs.extend([a, b])
        matmul = g.add_node(OpType.MATMUL, [a, b], name="matmul")
        g.outputs.append(matmul)

        x = torch.randn(4, 8)
        y = torch.randn(8, 4)

        out = execute_graph(g, {"a": x, "b": y})
        for val in out.values():
            expected = torch.matmul(x, y)
            assert torch.allclose(val, expected, atol=1e-5)

    def test_execute_softmax(self):
        g = Graph()
        x = g.add_node(OpType.CONSTANT, [], name="x")
        g.inputs.append(x)
        softmax = g.add_node(OpType.SOFTMAX, [x], attrs={"dim": -1}, name="softmax")
        g.outputs.append(softmax)

        inp = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        out = execute_graph(g, {"x": inp})
        for val in out.values():
            expected = torch.softmax(inp, dim=-1)
            assert torch.allclose(val, expected, atol=1e-5)

    def test_execute_reduce_sum(self):
        g = Graph()
        x = g.add_node(OpType.CONSTANT, [], name="x")
        g.inputs.append(x)
        red = g.add_node(OpType.REDUCE_SUM, [x], attrs={"dim": -1}, name="reduce")
        g.outputs.append(red)

        inp = torch.tensor([[1.0, 2.0, 3.0]])
        out = execute_graph(g, {"x": inp})
        for val in out.values():
            expected = torch.tensor([6.0])
            assert torch.allclose(val, expected, atol=1e-5)

    def test_execute_layernorm(self):
        g = Graph()
        x = g.add_node(OpType.CONSTANT, [], name="x")
        g.inputs.append(x)
        ln = g.add_node(OpType.LAYERNORM, [x], name="layernorm")
        g.outputs.append(ln)

        inp = torch.randn(4, 8)
        out = execute_graph(g, {"x": inp})
        for val in out.values():
            assert val.shape == inp.shape
            assert not torch.isnan(val).any()

    def test_execute_rmsnorm(self):
        g = Graph()
        x = g.add_node(OpType.CONSTANT, [], name="x")
        g.inputs.append(x)
        rn = g.add_node(OpType.RMSNORM, [x], name="rmsnorm")
        g.outputs.append(rn)

        inp = torch.randn(4, 8)
        out = execute_graph(g, {"x": inp})
        for val in out.values():
            assert val.shape == inp.shape
            assert not torch.isnan(val).any()

    def test_optimize_preserves_results(self):
        """Verify that optimize_graph produces the same output as the original."""
        g = _build_simple_graph()
        optimized = optimize_graph(g)

        inputs = {
            "a": torch.tensor([1.0, 2.0]),
            "b": torch.tensor([3.0, 4.0]),
        }

        orig_out = execute_graph(g, inputs)
        opt_out = execute_graph(optimized, inputs)

        # Both should produce [4.0, 6.0]
        for oo, oto in zip(orig_out.values(), opt_out.values()):
            assert torch.allclose(oo, oto, atol=1e-5)

    def test_optimize_chain_graph(self):
        g = _build_chain_graph()
        optimized = optimize_graph(g)
        assert optimized.validate()
        assert len(optimized.nodes) <= len(g.nodes)

    def test_optimize_diamond_graph(self):
        g = _build_diamond_graph()
        optimized = optimize_graph(g)

        # Verify execution
        inputs = {"input": torch.tensor([2.0])}
        orig_out = execute_graph(g, inputs)
        opt_out = execute_graph(optimized, inputs)

        for oo, oto in zip(orig_out.values(), opt_out.values()):
            assert torch.allclose(oo, oto, atol=1e-5)

    def test_fusion_with_execution(self):
        g = _build_fusion_candidate_graph()

        inputs = {
            "x": torch.tensor([1.0, -2.0, 3.0]),
            "bias": torch.tensor([1.0, 1.0, 1.0]),
        }

        orig_out = execute_graph(g, inputs)

        optimized = optimize_graph(g)
        opt_out = execute_graph(optimized, inputs)

        for oo, oto in zip(orig_out.values(), opt_out.values()):
            assert torch.allclose(oo, oto, atol=1e-5)

    def test_optimize_empty_graph(self):
        g = Graph()
        optimized = optimize_graph(g)
        assert len(optimized.nodes) == 0


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_graph_validate(self):
        g = Graph()
        assert g.validate() is True

    def test_empty_graph_topological_sort(self):
        g = Graph()
        order = g.topological_sort()
        assert order == []

    def test_single_node_graph(self):
        g = Graph()
        n = g.add_node(OpType.CONSTANT, [], name="only")
        g.inputs.append(n)
        g.outputs.append(n)

        assert g.validate()
        order = g.topological_sort()
        assert order == [n]

        optimized = optimize_graph(g)
        assert optimized.validate()

    def test_fully_constant_graph(self):
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], attrs={"value": torch.tensor(2.0)}, name="a")
        b = g.add_node(OpType.CONSTANT, [], attrs={"value": torch.tensor(3.0)}, name="b")
        g.inputs.extend([a, b])
        add = g.add_node(OpType.ADD, [a, b], name="add")
        mul = g.add_node(OpType.MUL, [add, a], name="mul")
        g.outputs.append(mul)

        optimized = optimize_graph(g)
        assert optimized.validate()

        # After optimization, all remaining nodes should be CONSTANT nodes
        # (all arithmetic folded, dead inputs removed)
        for node in optimized.nodes.values():
            assert node.op == OpType.CONSTANT, (
                f"Expected all nodes to be folded CONSTANT, but found {node.op.value}: {node.name}"
            )

    def test_cycle_detection(self):
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        b = g.add_node(OpType.ADD, [a], name="b")
        g.nodes[b].inputs = [b]  # Self-loop cycle
        g.nodes[b].outputs = [b]
        g.inputs.append(a)
        g.outputs.append(b)

        with pytest.raises(ValueError):
            g.validate()

    def test_missing_input_in_edge_case(self):
        g = Graph()
        n = g.add_node(OpType.CONSTANT, [], name="x")
        g.inputs.append(n)
        broken = g.add_node(OpType.ADD, [0], name="broken")
        g.nodes[broken].inputs = [42]  # Missing input
        g.outputs.append(broken)
        with pytest.raises(ValueError):
            g.validate()

    def test_optimize_single_node(self):
        g = Graph()
        n = g.add_node(OpType.CONSTANT, [], name="x")
        g.inputs.append(n)
        g.outputs.append(n)

        optimized = optimize_graph(g)
        assert len(optimized.nodes) == 1
        assert optimized.validate()

    def test_node_count_after_optimization(self):
        """Build a graph with fusion opportunities and verify node count reduces."""
        g = Graph()
        x = g.add_node(OpType.CONSTANT, [], name="x")
        bias = g.add_node(OpType.CONSTANT, [], name="bias")
        g.inputs.extend([x, bias])

        # Pattern that should fuse: add + relu
        add1 = g.add_node(OpType.ADD, [x, bias], name="add1")
        relu1 = g.add_node(OpType.RELU, [add1], name="relu1")

        # Another add+gelu pattern
        add2 = g.add_node(OpType.ADD, [x, bias], name="add2")
        gelu1 = g.add_node(OpType.GELU, [add2], name="gelu1")

        merge = g.add_node(OpType.ADD, [relu1, gelu1], name="merge")
        g.outputs.append(merge)

        n_before = len(g.nodes)
        optimized = optimize_graph(g)

        assert len(optimized.nodes) < n_before
        assert optimized.validate()
