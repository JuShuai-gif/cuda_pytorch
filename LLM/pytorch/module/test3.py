"""Module 源码分析: __call__ → _call_impl → hook 执行 → C++ 调度链。

使用工具: 自定义 hook 追踪 / torch._C 内部 API /
         _forward_hooks 字典探查 / Module 内部状态 dump

运行:
  python test3.py                  # 全链路分析
  python test3.py call_chain       # __call__ 完整执行路径
  python test3.py hook_inside      # hook 内部存储探究
  python test3.py state_inside     # _parameters/_buffers 内部
  python test3.py compile_interaction  # compile + hook 交互

参考源码:
  torch/nn/modules/module.py    — Module 完整实现
  torch/csrc/Module.cpp         — Python-C 绑定
"""

import sys
import torch
import torch.nn as nn


# ============ 1. __call__ 执行链追踪 ============
def exp_call_chain():
    """追踪 model(x) 从 Python 到 C++ 的完整调用链。"""
    print("=" * 60)
    print("1. __call__ 执行链: model(x) 的完整路径")
    print("=" * 60)

    class TraceModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
            # 注册各种 hook 看执行顺序
            self.register_forward_pre_hook(
                lambda m, a: print("   [1] forward_pre_hook")
            )
            self.register_forward_hook(lambda m, a, o: print("   [3] forward_hook"))

        def forward(self, x):
            print("   [2] forward() executing")
            return self.linear(x)

    model = TraceModule()
    x = torch.ones(1, 4)

    print("  model(x) 调用链:")
    model(x)

    print("\n  源码路径 (module.py):")
    print("  model(x)")
    print("    → Module.__call__ (:1917)  [别名 _wrapped_call_impl]")
    print("    → _wrapped_call_impl (:1774)  [检查 _compiled_call_impl]")
    print("    → _call_impl (:1782)")
    print("       ├─ forward_pre_hooks (:1805)  [修改 args/kwargs]")
    print("       ├─ setup backward hooks (:1827)")
    print("       ├─ forward() (:1832)  [用户定义的 forward]")
    print("       ├─ forward_hooks (:1833)  [修改 output]")
    print("       └─ setup output hooks (:1850)")
    print()


# ============ 2. Hook 内部存储探究 ============
def exp_hook_inside():
    """探究 Module 如何存储和管理 hooks。"""
    print("=" * 60)
    print("2. Hook 内部存储: _forward_hooks / _backward_hooks")
    print("=" * 60)

    model = nn.Linear(4, 4)

    # 注册多个 hooks
    model.register_forward_pre_hook(lambda m, a: None)
    model.register_forward_pre_hook(lambda m, a: None, prepend=True)
    fwd_handle = model.register_forward_hook(lambda m, a, o: None)
    bw_handle = model.register_full_backward_hook(lambda m, gi, go: None)

    # 探查内部 OrderedDict
    print(f"  _forward_pre_hooks:         {dict(model._forward_pre_hooks).keys()}")
    print(f"  _forward_pre_hooks (size):  {len(model._forward_pre_hooks)}")
    print(f"  _forward_hooks:             {dict(model._forward_hooks).keys()}")
    print(f"  _backward_hooks:            {dict(model._backward_hooks).keys()}")
    print(f"  _state_dict_hooks:          {len(model._state_dict_hooks)}")
    print(f"  _state_dict_pre_hooks:      {len(model._state_dict_pre_hooks)}")

    print(f"\n  Hook 存储使用 OrderedDict (保持插入顺序):")
    print(f"  key = RemovableHandle.id (唯一的 int)")
    print(f"  value = hook function")

    # 移除 hook 后检查
    fwd_handle.remove()
    print(f"\n  移除 fwd_handle 后:")
    print(f"  _forward_hooks (size): {len(model._forward_hooks)}")

    bw_handle.remove()
    print(f"\n  RemovableHandle 原理:")
    print(f"  存储一个弱引用到 OrderedDict")
    print(f"  remove() → del ordered_dict[self.id]")
    print()


# ============ 3. State 内部探究 ============
def exp_state_inside():
    """探究 _parameters / _buffers / _modules 的底层存储。"""
    print("=" * 60)
    print("3. Module State 内部: _parameters / _buffers / _modules")
    print("=" * 60)

    class InspectModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 3)
            self.register_buffer("running", torch.ones(3))
            self.register_buffer("step", torch.tensor(0), persistent=False)
            self.sub = nn.ReLU()

    m = InspectModel()

    # 探查各 OrderedDict
    for attr in ["_parameters", "_buffers", "_modules", "_non_persistent_buffers_set"]:
        val = getattr(m, attr)
        if isinstance(val, dict):
            print(f"  {attr}: {dict(val) if hasattr(val, 'items') else val}")
        elif isinstance(val, set):
            print(f"  {attr}: {val}")
        else:
            print(f"  {attr}: {type(val)}")

    # __setattr__ 如何路由
    print(f"\n  __setattr__ 路由规则 (module.py:1971):")
    print(f"  isinstance(value, Parameter) → register_parameter()")
    print(f"  isinstance(value, Module)    → _modules[name] = value")
    print(f"  isinstance(value, Tensor/Buffer) → register_buffer() [torch >= 2.3]")
    print(f"  否则                          → super().__setattr__() → __dict__")

    # 参数形状和数据类型
    print(f"\n  参数详情:")
    for name, p in m.named_parameters():
        print(
            f"    {name:20s} shape={list(p.shape)} dtype={p.dtype} "
            f"device={p.device} requires_grad={p.requires_grad}"
        )
    print()


# ============ 4. Compile + Hook 交互 ============
def exp_compile_interaction():
    """探究 torch.compile 如何影响 forward + hook 执行。"""
    print("=" * 60)
    print("4. torch.compile + Hook 交互分析")
    print("=" * 60)

    class HookModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)
            self._trace = []

        def forward(self, x):
            self._trace.append("forward")
            return self.fc(x).relu()

    model = HookModel()
    model.register_forward_pre_hook(lambda m, a: m._trace.append("pre"))
    model.register_forward_hook(lambda m, a, o: m._trace.append("fwd"))

    # Before compile
    model(torch.randn(4))
    print(f"  Without compile: {model._trace}")
    assert model._trace == ["pre", "forward", "fwd"]

    # After compile
    model._trace.clear()
    compiled = torch.compile(model)
    compiled(torch.randn(4))
    print(f"  With compile:    {model._trace}")

    # 查看内部 compiled_call_impl
    print(f"\n  _compiled_call_impl: {model._compiled_call_impl is not None}")
    print()
    print("  源码逻辑 (module.py:1774):")
    print("  def _wrapped_call_impl(self, *args, **kwargs):")
    print("      if self._compiled_call_impl is not None:")
    print("          return self._compiled_call_impl(*args, **kwargs)  # compile 路径")
    print("      else:")
    print("          return self._call_impl(*args, **kwargs)          # 正常路径")
    print()
    print("  → torch.compile 替换了 __call__ 的底层实现")
    print("  → Dynamo hook 机制可能 bypass Python hook (版本差异)")
    print()


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else []
    if exps:
        for name in exps:
            globals()[f"exp_{name}"]()
    else:
        exp_call_chain()
        exp_hook_inside()
        exp_state_inside()
        exp_compile_interaction()

    print("[Module source analysis] DONE")


if __name__ == "__main__":
    main()
