import torch
from torch.fx import symbolic_trace, replace_pattern


# 如何使用 FX 子图重写器
#
# 对于简单的子图重写，FX 暴露了工具函数：
#
#     replace_pattern(gm : GraphModule,
#                     pattern : Callable,
#                     replacement : Callable)
#                     -> None
#
# replace_pattern 在 GraphModule (gm) 的 Graph 中匹配所有可能的非重叠
# 运算符集合及其数据依赖关系 (pattern)，然后将每个匹配到的子图替换为
# 另一个子图 (replacement)。
#
# replace_pattern 的文档字符串（位于 subgraph_rewriter.py 中）
# 深入解释了 pattern 和 replacement 应如何指定、模式匹配期间
# 发生什么以及其他重要的技术细节。因此，本教程仅旨在概述
# FX 子图重写器的基本功能。让我们重写一个图吧！


# 示例模块
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, w1, w2):
        val1 = torch.neg(w1)
        m1 = torch.cat([val1, w2]).sum()
        val2 = torch.neg(w1)
        m2 = torch.cat([val2, w2]).sum()
        return x + torch.max(m1) + torch.max(m2)


traced = symbolic_trace(M())


# 定义 pattern。FX 子图重写器将匹配较大图中所有
# 非重叠的 pattern 实例。
# 注意 Pattern 匹配是基于数据依赖的，而非 Node 名称。
# 即使我们操作的是名为 a1 和 a2 的 Node 而非 w1 和 w2，
# pattern 仍然对上面 torch.cat([w1, w2]).sum() 的两个实例是有效匹配。
# 只有贡献到 pattern 单一输出值的操作
# 才会被考虑。
def pattern(a1, a2):
    val1 = torch.neg(a1)
    return torch.cat([val1, a2]).sum()


# 定义 replacement（与 pattern 规则相同）
def replacement(w1, w2):
    return torch.stack([w1, w2])


# 将 traced 中的 pattern 替换为 replacement
replace_pattern(traced, pattern, replacement)

# 调用 replace_pattern 后，生成的代码是：
"""
def forward(self, x, w1, w2):
    stack = torch.stack([w1, w2])
    max_1 = torch.max(stack);  stack = None
    add = x + max_1;  x = max_1 = None
    stack_1 = torch.stack([w1, w2]);  w1 = w2 = None
    max_2 = torch.max(stack_1);  stack_1 = None
    add_1 = add + max_2;  add = max_2 = None
    return add_1
"""

print(traced)
