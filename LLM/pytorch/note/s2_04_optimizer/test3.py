"""Optimizer 源码分析: state dict 内部存储, param_groups 指针引用。

使用工具: id(param) 追踪 / state dict keys / torch.save/load 内部

运行:
  python test3.py                    # 全链路分析
  python test3.py state_structure    # state dict 内部结构
  python test3.py param_identity     # Parameter 对象身份追踪
  python test3.py param_group_map    # param_groups → state 映射

参考源码:
  torch/optim/optimizer.py           — Optimizer 基类
  torch/optim/sgd.py                 — SGD step 中的 state 维护
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim


# ============ 1. State dict 内部结构 ============
def exp_state_structure():
    """探查 optimizer.state_dict() 的完整结构。"""
    print("=" * 60)
    print("1. State Dict: optimizer 如何存储训练状态")
    print("=" * 60)

    torch.manual_seed(42)
    model = nn.Linear(4, 2)
    opt = optim.AdamW(model.parameters(), lr=0.01)

    # 运行几步，生成 state
    x = torch.randn(8, 4)
    y = torch.randn(8, 2)
    for _ in range(3):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()

    sd = opt.state_dict()

    print(f"  state_dict keys: {list(sd.keys())}")
    print(f"  param_groups count: {len(sd['param_groups'])}")
    print(f"  state entries:      {len(sd['state'])}")

    # param_groups 结构
    for i, pg in enumerate(sd["param_groups"]):
        print(f"\n  param_groups[{i}]:")
        for k, v in pg.items():
            if k != "params":
                print(f"    {k}: {v}")
            else:
                print(f"    params: {len(v)} ids = {v}")

    # state 结构
    print(f"\n  state entries (每个 parameter id → 状态 dict):")
    for param_id, state in sd["state"].items():
        print(f"    param id={param_id}: keys={list(state.keys())}")
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                print(
                    f"      {k}: shape={list(v.shape)} dtype={v.dtype} "
                    f"mean={v.mean().item():.4f}"
                )
            else:
                print(f"      {k}: {v}")

    print(f"\n  state dict 原理 (optimizer.py):")
    print(f"  1. param_groups: [{'params': [id(p) ...], 'lr': ..., ...}]")
    print(f"     → 用 id(p) 识别参数 (不依赖 tensor 内容)")
    print(f"  2. state: {id(p): {'step': ..., 'exp_avg': ..., ...}}")
    print(f"     → 每个参数的动量 buffer / step 计数")
    print(f"  3. load_state_dict: 用 id 匹配 → copy buffer 回新 optimizer")
    print()


# ============ 2. Parameter 对象身份追踪 ============
def exp_param_identity():
    """追踪 Parameter 在 optimizer 中的身份。"""
    print("=" * 60)
    print("2. Param Identity: optimizer 如何识别参数")
    print("=" * 60)

    torch.manual_seed(42)
    model = nn.Linear(4, 2)
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 记录参数的 id
    param_ids = {p: id(p) for p in model.parameters()}
    print(f"  Parameter identities (id):")
    for p, pid in param_ids.items():
        shape = list(p.shape)
        print(f"    {shape:15s} id={pid}")

    # optimizer state 用 id 做 key
    print(f"\n  Optimizer state keys: {list(opt.state.keys())}")
    for p, state in opt.state.items():
        pid = id(p)
        print(f"    id={pid} → state={dict(state)}")

    # 运行一步后 state 更新
    x = torch.randn(4, 4)
    y = torch.randn(4, 2)
    opt.zero_grad()
    nn.functional.mse_loss(model(x), y).backward()
    opt.step()

    print(f"\n  After step:")
    for p, state in opt.state.items():
        print(f"    id={id(p)} → state={dict(state)}")

    print(f"\n  为什么用 id(p) 而不是 p 自身?")
    print(f"  1. p 的内容在 training 中不断变化")
    print(f"  2. id(p) 在对象生命周期内不变")
    print(f"  3. torch.save/load state_dict 时也保持 id 稳定")
    print(f"  4. 但重新创建 model → 新的 Parameter 对象 → id 变化")
    print(f"     → load_state_dict 必须先创建 model 再 load")
    print()


# ============ 3. param_groups → state 映射 ============
def exp_param_group_map():
    """探究 param_groups 和 state 的映射关系。"""
    print("=" * 60)
    print("3. param_groups ↔ state 映射")
    print("=" * 60)

    torch.manual_seed(42)

    # 多 param_groups
    model = nn.Sequential(
        nn.Linear(8, 16),  # group 0
        nn.Linear(16, 4),  # group 1
        nn.Linear(4, 2),  # group 2
    )

    opt = optim.Adam(
        [
            {"params": model[0].parameters(), "lr": 0.1, "name": "layer1"},
            {"params": model[1].parameters(), "lr": 0.01, "name": "layer2"},
            {"params": model[2].parameters(), "lr": 0.001, "name": "layer3"},
        ]
    )

    # 查看每个 group 有哪些参数
    for i, pg in enumerate(opt.param_groups):
        print(f"  Group {i} ({pg['name']}): lr={pg['lr']}")
        for p in pg["params"]:
            pid = id(p)
            shape = list(p.shape)
            has_state = p in opt.state
            print(f"    param id={pid} shape={shape} has_state={has_state}")

    # 运行一步后，每个 param 都有 state
    x = torch.randn(4, 8)
    y = torch.randn(4, 2)
    opt.zero_grad()
    nn.functional.mse_loss(model(x), y).backward()
    opt.step()

    print(f"\n  After step — all params have state:")
    for i, pg in enumerate(opt.param_groups):
        for p in pg["params"]:
            st = opt.state[p]
            print(
                f"    Group {i}: param id={id(p)} "
                f"state={'step' in st} step={st.get('step', 'N/A')}"
            )

    print(f"\n  param_groups 遍历: step() 中 for group in param_groups:")
    print(f"    对每个 param → 取 group 的 lr/weight_decay")
    print(f"    对每个 param → 取 state 的 momentum buffer")
    print(f"  → 不同 group 可以有不同的 lr, 同一模型的不同层")
    print()


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else []
    if exps:
        for name in exps:
            globals()[f"exp_{name}"]()
    else:
        exp_state_structure()
        exp_param_identity()
        exp_param_group_map()

    print("[Optimizer source analysis] DONE")


if __name__ == "__main__":
    main()
