from typing import List
import torch


def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print(">>> my_compiler() 被调用：")
    print(">>> FX 图：")
    gm.graph.print_tabular()
    print(f">>> 代码：\n{gm.code}")
    return gm


@torch.compile(backend=my_compiler)
def foo(x, y):
    return (x + y) * x


if __name__ == "__main__":
    a, b = torch.randn(10), torch.ones(10)
    foo(a, b)


# 另一种做法是使用 torch logs


import torch


@torch.compile
def foo(x, y):
    return (x + y) * x


x = torch.randn(10)
y = torch.ones(10)
foo(x, y)


# TORCH_LOGS=graph_code python3 dynamo/01.py
