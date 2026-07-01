import torch
from torch.fx import symbolic_trace
import operator

# 如何用一个 Op 替换另一个 Op
#
# 1. 遍历 GraphModule 的 Graph 中的所有 Node。
# 2. 判断当前 Node 是否应被替换。（建议：匹配 Node 的 target 属性）。
# 3. 创建一个替换 Node 并将其添加到 Graph 中。
# 4. 使用 FX 内置的 replace_all_uses_with 将当前 Node 的所有使用替换为替换 Node。
# 5. 从图中删除旧 Node。
# 6. 在 GraphModule 上调用 recompile。这会更新生成的 Python 代码以反映新的 Graph 状态。
#
# 目前，FX 不提供任何方式来保证替换后的运算符在语法上有效。由用户确认
# 任何新运算符是否与现有操作数兼容。
#
# 下面的代码演示了将任意加法实例替换为按位 AND 的示例。
#
# 要检查图在 Op 替换过程中如何演变，在要检查的行之后添加
# print(traced.graph)。或者调用 traced.graph.print_tabular() 以表格格式查看 IR。


class M(torch.nn.Module):
    def forward(self, x, y):
        return x + y, torch.add(x, y), x.add(y)


traced = symbolic_trace(M())

print("原始代码")
print(traced)

# 如上例所示，表示加法有几种不同的方式。可能的情况有：
#     1. x + y - 一个 call_function Node，target 为 operator.add。
#         我们可以直接匹配 operator.add 本身。
#     2. torch.add(x, y) - 一个 call_function Node，target 为
#         torch.add。同样，我们可以直接匹配此函数。
#     3. x.add(y) - Tensor 方法调用，其 target 可以用字符串匹配。

patterns = set([operator.add, torch.add, "add"])

# 遍历图中的所有节点
for n in traced.graph.nodes:
    # 如果 target 匹配其中一个模式
    if any(n.target == pattern for pattern in patterns):
        # 设置插入点，添加新节点，并替换 n 的所有使用
        # 为新节点
        with traced.graph.inserting_after(n):
            new_node = traced.graph.call_function(torch.bitwise_and, n.args, n.kwargs)
            n.replace_all_uses_with(new_node)
        # 从图中删除旧节点
        traced.graph.erase_node(n)

traced.recompile()  # 看看如果不重新编译会发生什么

print("FX 图操作")
print(traced)
