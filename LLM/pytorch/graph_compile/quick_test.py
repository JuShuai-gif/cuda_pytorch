"""
最简 torch.compile 示例 —— 两行代码跑通编译

装饰器写法:
  @torch.compile
  def fn(x): return torch.sin(torch.cos(x))

函数调用写法（等价）:
  fn = torch.compile(lambda x: torch.sin(torch.cos(x)))

运行:
  python quick_test.py
  TORCH_LOGS="+dynamo" python quick_test.py   # 看 Dynamo 抓图过程
  TORCH_LOGS="output_code" python quick_test.py # 看生成的 kernel
"""

import torch


# 写法 1: 装饰器
@torch.compile(backend="inductor")
def fn1(x):
    return torch.sin(torch.cos(x))


# 写法 2: 函数调用（等价）
def fn2(x):
    return torch.sin(torch.cos(x))


fn2 = torch.compile(fn2, backend="inductor")


x = torch.randn(10000).cuda()

# 第一次调用 → 触发 JIT 编译
a = fn1(x)

# 第二次调用 → 直接执行编译后的 kernel
b = fn2(x)

print(f"Compiled 1: {a[:3]}")
print(f"Compiled 2: {b[:3]}")
print(f"一致: {(a == b).all().item()}")
