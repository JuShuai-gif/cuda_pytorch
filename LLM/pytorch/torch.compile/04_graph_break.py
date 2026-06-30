import torch

torch._logging.set_logs(graph_code=True)
torch._logging.set_logs(graph_breaks=True)  # 查看图断裂


def bar(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b


opt_bar = torch.compile(bar)
inp1 = torch.ones(10)
inp2 = torch.ones(10)

torch._dynamo.reset()  # 重置以清除 torch.compile 缓存
opt_bar(inp1, inp2)
opt_bar(inp1, -inp2)


# 第一次调用 bar 时，我们看到两个图被追踪：
# torch.abs 部分 + b < 0 部分
# 第二次调用时，torch.abs 部分已缓存，所以只有 b < 0 部分运行

# 为了最大化加速，应限制图断裂数量。通过使用 fullgraph=True
# 可以强制 TorchDynamo 在遇到第一个图断裂时抛出错误。

# 当 TD 遇到不支持的 Python 语法（如数据相关的控制流）时，
# 它会退出计算图，让 Python 解释器处理不支持的代码，
# 然后继续捕获计算图。
# 具体来说：在遇到条件分支 if b.sum() < 0 之前，TD 捕获计算图并正常执行。
# 遇到条件分支时，TD 让 Python 决定分支的结果。

import traceback as tb

torch._dynamo.reset()

opt_bar_fullgraph = torch.compile(bar, fullgraph=True)
try:
    opt_bar_fullgraph(torch.randn(10), torch.randn(10))
except:
    tb.print_exc()
