"""nn 容器模块 demo + 源码验证。

Companion script for 46_containers/containers.md.
  1. sequential:      构造两种方式、forward 链式、切片、+/*
  2. registration:    list vs ModuleList — 参数是否被注册
  3. modulelist:      手写 forward 遍历、append/extend/repr 折叠
  4. moduledict:      按 key 动态选分支、保序
  5. parameterlist:   裸 Tensor 自动 wrap 成 Parameter、不能被调用
  6. pitfalls:        常见坑点复现

Run:
    python test1.py               # full demo
    python test1.py sequential
    python test1.py registration
    python test1.py modulelist
    python test1.py moduledict
    python test1.py parameterlist
    python test1.py pitfalls
"""

import sys
from collections import OrderedDict

import torch
import torch.nn as nn


# ============ 1. Sequential ============
def exp_sequential():
    print("=" * 60)
    print("1. Sequential: 构造 / forward / 切片 / 运算符")
    print("=" * 60)

    # 位置参数 → 命名为 "0","1","2"
    a = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    print("  位置参数构造, 子模块 key:", list(a._modules.keys()))

    # OrderedDict → 自定义命名
    b = nn.Sequential(
        OrderedDict(
            [("fc1", nn.Linear(4, 8)), ("act", nn.ReLU()), ("fc2", nn.Linear(8, 2))]
        )
    )
    print("  OrderedDict 构造, 子模块 key:", list(b._modules.keys()))
    print("  b.fc1 可按名字访问:", b.fc1.__class__.__name__)

    # forward 链式
    x = torch.randn(3, 4)
    print("  forward: input", tuple(x.shape), "-> output", tuple(a(x).shape))

    # 切片返回新 Sequential
    head = a[0:2]
    print("  a[0:2] 类型:", head.__class__.__name__, "长度:", len(head))

    # + 拼接, * 重复
    c = a[0:1] + a[1:2]
    print("  a[0:1] + a[1:2] 长度:", len(c))
    d = nn.Sequential(nn.ReLU()) * 3
    print("  ReLU * 3 长度:", len(d))


# ============ 2. list vs ModuleList 参数注册 ============
def exp_registration():
    print("=" * 60)
    print("2. Python list vs ModuleList: 参数是否被注册")
    print("=" * 60)

    class WithList(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [nn.Linear(4, 4) for _ in range(3)]  # 裸 list

    class WithModuleList(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    m1, m2 = WithList(), WithModuleList()
    print(f"  裸 list      -> parameters() 数量: {len(list(m1.parameters()))}  (丢失!)")
    print(f"  ModuleList   -> parameters() 数量: {len(list(m2.parameters()))}")
    print(f"  裸 list      -> state_dict keys: {len(m1.state_dict())}")
    print(f"  ModuleList   -> state_dict keys: {len(m2.state_dict())}")


# ============ 3. ModuleList ============
def exp_modulelist():
    print("=" * 60)
    print("3. ModuleList: 手写 forward / append / repr 折叠")
    print("=" * 60)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

        def forward(self, x):
            for blk in self.blocks:  # 手动遍历
                x = torch.relu(blk(x))
            return x

    net = Net()
    print("  forward 输出:", tuple(net(torch.randn(2, 4)).shape))

    net.blocks.append(nn.Linear(4, 4))
    net.blocks.extend([nn.Linear(4, 4), nn.Linear(4, 4)])
    print("  append+extend 后长度:", len(net.blocks))

    # repr 折叠: 连续相同层压缩显示
    print("  repr (注意 N x 折叠):")
    for line in repr(net.blocks).splitlines():
        print("   ", line)

    # 直接调用 ModuleList 会报错
    try:
        net.blocks(torch.randn(2, 4))
    except (NotImplementedError, TypeError) as e:
        print("  直接调用 ModuleList 报错:", type(e).__name__)


# ============ 4. ModuleDict ============
def exp_moduledict():
    print("=" * 60)
    print("4. ModuleDict: 按 key 动态选分支 / 保序")
    print("=" * 60)

    class Router(nn.Module):
        def __init__(self):
            super().__init__()
            self.ops = nn.ModuleDict(
                {
                    "path_a": nn.Linear(4, 4),
                    "path_b": nn.Linear(4, 4),
                }
            )

        def forward(self, x, which):
            return self.ops[which](x)  # 运行时按 key 选层

    r = Router()
    x = torch.randn(1, 4)
    print("  choice='path_a' 输出:", tuple(r(x, "path_a").shape))
    print("  keys:", list(r.ops.keys()))
    print("  'path_b' in ops:", "path_b" in r.ops)
    r.ops["path_c"] = nn.Linear(4, 4)
    print("  新增后 keys (保序):", list(r.ops.keys()))


# ============ 5. ParameterList ============
def exp_parameterlist():
    print("=" * 60)
    print("5. ParameterList: 裸 Tensor 自动 wrap / 不能被调用")
    print("=" * 60)

    pl = nn.ParameterList([torch.randn(2, 2), nn.Parameter(torch.randn(2, 2))])
    print("  存入裸 Tensor 后类型:", type(pl[0]).__name__)  # 被 wrap 成 Parameter
    print("  pl[0].requires_grad:", pl[0].requires_grad)
    print("  parameters() 数量:", len(list(pl.parameters())))

    pd = nn.ParameterDict({"w": torch.randn(2, 2)})
    print("  ParameterDict['w'] 类型:", type(pd["w"]).__name__)

    # 不能被调用
    try:
        pl(torch.randn(2, 2))
    except RuntimeError as e:
        print("  调用 ParameterList 报错:", str(e))


# ============ 6. 常见坑点 ============
def exp_pitfalls():
    print("=" * 60)
    print("6. 常见坑点复现")
    print("=" * 60)

    # 坑 1: Sequential 位置参数无法按名字访问
    seq = nn.Sequential(nn.Linear(4, 4))
    print("  坑1: seq._modules key =", list(seq._modules.keys()), "(不是变量名)")

    # 坑 2: Sequential + list 报错
    try:
        _ = seq + [nn.ReLU()]
    except ValueError as e:
        print("  坑2: Sequential + list ->", type(e).__name__)

    # 坑 3: 删除中间层后重新连续编号
    s = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 1))
    del s[1]
    print("  坑3: 删除 s[1] 后 key 重编号 =", list(s._modules.keys()))


ALL = {
    "sequential": exp_sequential,
    "registration": exp_registration,
    "modulelist": exp_modulelist,
    "moduledict": exp_moduledict,
    "parameterlist": exp_parameterlist,
    "pitfalls": exp_pitfalls,
}


def main():
    if len(sys.argv) > 1:
        key = sys.argv[1]
        if key not in ALL:
            print(f"unknown: {key}\navailable: {', '.join(ALL)}")
            return
        ALL[key]()
    else:
        for fn in ALL.values():
            fn()
            print()


if __name__ == "__main__":
    main()
