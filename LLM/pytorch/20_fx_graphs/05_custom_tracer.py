import torch
from torch.fx import symbolic_trace, Tracer, Graph, GraphModule, Node
from typing import Any, Callable, Dict, Optional, Tuple, Union


# 如何创建和使用自定义 Tracer
#
# Tracer——实现 torch.fx.symbolic_trace 符号追踪功能的类——
# 可以被继承以覆盖追踪过程的多种行为。在本教程中，我们将演示
# 如何使用一些手写的 Tracer 自定义符号追踪过程。每个示例将展示
# 只需覆盖 Tracer 类中的少量方法，就能改变符号追踪产生的 Graph。
# 关于可以更改的方法的完整描述，请参考 Tracer 类中方法的文档字符串。
# 信息可在以下地址找到：https://pytorch.org/docs/master/fx.html#torch.fx.Tracer
#
# 如果你想看一个真实世界的自定义 tracer 示例，请查看 FX 的 AST
# 重写器 in rewriter.py。RewritingTracer 继承自 Tracer 但
# 覆盖了 trace 函数，以便将所有的 assert 调用重写为更 FX 友好的
# torch.assert。
#
# 注意调用 symbolic_trace(m) 等同于
# GraphModule(m, Tracer().trace(m))。（Tracer 是 Tracer 的默认
# 实现，定义在 symbolic_trace.py 中。）


# 自定义 Tracer #1：追踪所有 torch.nn.ReLU 子模块的内部
#
# 在符号追踪过程中，一些子模块会被追踪进去，其构成
# 操作会被记录；其他子模块则在 IR 中显示为
# 原子的 "call_module" Node。后一类模块称为
# "leaf module"。默认情况下，PyTorch 标准库中的所有模块
# (torch.nn) 都是 leaf module。我们可以通过创建自定义 Tracer 并
# 覆盖 is_leaf_module 来改变这一点。在本例中，我们保留
# 所有 torch.nn Module 的默认行为，除 ReLU 外。


class M1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(x)


default_traced: GraphModule = symbolic_trace(M1())
# 使用默认 tracer 追踪并调用 print_tabular 产生：
#
#     opcode       name    target    args       kwargs
#     -----------  ------  --------  ---------  --------
#     placeholder  x       x         ()         {}
#     call_module  relu_1  relu      (x,)       {}
#     output       output  output    (relu_1,)  {}

default_traced.graph.print_tabular()


class LowerReluTracer(Tracer):
    def is_leaf_module(self, m: torch.nn.Module, qualname: str):
        if isinstance(m, torch.nn.ReLU):
            return False
        return super().is_leaf_module(m, qualname)


# 使用我们的自定义 tracer 追踪并调用 print_tabular 产生：
#
#     opcode         name    target                             args       kwargs
#     -------------  ------  ---------------------------------  ---------  ------------------
#     placeholder    x       x                                  ()         {}
#     call_function  relu_1  <function relu at 0x7f66f7170b80>  (x,)       {'inplace': False}
#     output         output  output                             (relu_1,)  {}

lower_relu_tracer = LowerReluTracer()
custom_traced_graph: Graph = lower_relu_tracer.trace(M1())
custom_traced_graph.print_tabular()


# 自定义 Tracer #2：为每个 Node 添加额外属性
#
# 这里，我们将覆盖 create_node 以便在每个 Node 创建时
# 添加一个新的属性。


class M2(torch.nn.Module):
    def forward(self, a, b):
        return a + b


class TaggingTracer(Tracer):
    def create_node(
        self,
        kind: str,
        target: Union[str, Callable],
        args: Tuple[Any],
        kwargs: Dict[str, Any],
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
    ) -> Node:
        n = super().create_node(kind, target, args, kwargs, name)
        n.tag = "foo"
        return n


custom_traced_graph: Graph = TaggingTracer().trace(M2())


def assert_all_nodes_have_tags(g: Graph) -> bool:
    for n in g.nodes:
        if not hasattr(n, "tag") or not n.tag == "foo":
            return False
    return True


# 输出 "True"
print(assert_all_nodes_have_tags(custom_traced_graph))
