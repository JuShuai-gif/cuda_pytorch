"""nn.Module core mechanisms demo: hooks, parameters, state_dict, _apply.

Companion script for module/module.md. Covers:
  1. forward hooks:      pre-hook modifies args, post-hook modifies output
  2. backward hooks:     capture grad_input/grad_output
  3. always_call hooks:  hooks that fire even on exception (debugging aid)
  4. __setattr__:        automatic Parameter/Buffer/Module detection
  5. name conflicts:     silent overwrite between param/buffer/module (common bug)
  6. state_dict:         parameter/buffer serialization + strict mode
  7. missing_keys debug: how to find shape/dtype mismatches
  8. _apply:             batch type conversion (cuda, half, float)
  9. parameters:         deduplication of shared parameters
  10. hook chain debug:  print all hooks on a module

Run:
    python test1.py                # full demo
    python test1.py hooks          # forward & backward hooks
    python test1.py always_call    # always_call hooks (survives exception)
    python test1.py conflict       # name conflict between param/buffer/module
    python test1.py param          # parameter/buffer registration
    python test1.py state          # state_dict serialization + missing keys
    python test1.py apply          # _apply type conversion
    python test1.py dedup          # parameter deduplication
    python test1.py hook_chain     # inspect all hooks on a module

=== DEBUG 常见问题 ===
  Q: hook 不触发?
  A: 检查 hook 是否注册在正确的 module 上, hook 是否被 .remove() 了,
     forward 是否走了 _compiled_call_impl (torch.compile) 路径

  Q: state_dict 有 unexpected keys?
  A: 检查 __setattr__ 是否不小心注册了不想要的 parameter/buffer,
     用 model.named_parameters() 和 model.named_buffers() 检查

  Q: load_state_dict 报 size mismatch?
  A: 用 TORCH_DUMP_STATE_DICT=1 环境变量, 或手动对比:
     for (k1,v1),(k2,v2) in zip(model.state_dict().items(), ckpt.items()):
         if v1.shape != v2.shape: print(k1, v1.shape, v2.shape)

  Q: _apply 后 optimizer 状态丢失?
  A: _apply 默认保留 Parameter 对象身份 (param.data = new), 但如果模型
     用 swap_tensors 路径 (FakeTensor subclass), 检查是否重建了 Parameter
"""

import sys

import torch
import torch.nn as nn


# ============ 1. Forward hooks: modify args & output ============
def exp_hooks():
    print("=" * 60)
    print("1. Forward hooks: modify args before, output after forward")
    print("=" * 60)

    class HookDemo(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
            self.linear.weight.data.fill_(2.0)
            self.linear.bias.data.fill_(0.0)

        def forward(self, x):
            return self.linear(x)

    model = HookDemo()
    x = torch.ones(1, 4)

    # Pre-hook: multiply input by 3
    def pre_hook(module, args):
        x = args[0]
        print(f"  pre-hook:  input sum = {x.sum().item():.1f}")
        return (x * 3,)  # replace args

    # Forward hook: multiply output by 5
    def fwd_hook(module, args, output):
        print(f"  fwd-hook:  output sum = {output.sum().item():.1f}")
        return output * 5  # replace result

    handle_pre = model.register_forward_pre_hook(pre_hook)
    handle_fwd = model.register_forward_hook(fwd_hook)

    y = model(x)
    expected = (x * 3) @ model.linear.weight.T * 5
    print(f"  result:    sum = {y.sum().item():.1f}")
    print(f"  expected:  sum = {expected.sum().item():.1f}")
    print(f"  match:     {torch.allclose(y, expected)}")

    handle_pre.remove()
    handle_fwd.remove()

    print()


# ============ 1b. always_call hooks ============
def exp_always_call():
    print("=" * 60)
    print("1b. always_call hooks: fire even on exception (debug tool)")
    print("=" * 60)

    class BuggyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x):
            if x.sum() > 100:
                raise ValueError("simulated crash!")
            return self.linear(x)

    model = BuggyModule()
    caught = []

    # Normal hook: NOT fired on exception
    model.register_forward_hook(lambda m, a, o: caught.append("normal_hook"))

    # always_call hook: fires even if forward crashes
    model.register_forward_hook(
        lambda m, a, o: caught.append("always_call_hook"), always_call=True
    )

    # Normal run: both fire
    x = torch.ones(4) * 0.1
    model(x)
    print(f"  Normal forward: {caught}")

    caught.clear()
    # Crash run: only always_call fires
    x = torch.ones(4) * 1000.0
    try:
        model(x)
    except ValueError:
        pass
    print(f"  After crash:    {caught}")
    print("  -> always_call hooks survive exceptions (use for logging/debug state)")
    print()


# ============ 1c. Name conflicts ============
def exp_conflict():
    print("=" * 60)
    print("1c. __setattr__ name conflict (common bug)")
    print("=" * 60)

    # BUG DEMO: assigning a Module to a name already used by a Parameter
    m = nn.Module()
    m.register_parameter("weight", nn.Parameter(torch.randn(3)))
    print(f"  weight before: {type(m.weight).__name__} in _parameters")
    print(f"  _parameters: {list(m._parameters.keys())}")

    # Assigning a Module to "weight" silently removes the Parameter!
    m.weight = nn.Linear(3, 3)
    print(f"  weight after:  {type(m.weight).__name__} in _modules")
    print(f"  _parameters: {list(m._parameters.keys())}")
    print(f"  _modules:    {list(m._modules.keys())}")
    print("  -> __setattr__ removes name from _parameters when assigning a Module!")
    print("  -> This is why 'weight' / 'bias' should not be used for submodules")

    # Another pitfall: plain Tensor auto-registered as buffer (torch >= 2.x)
    m2 = nn.Module()
    m2.cache = torch.zeros(10)  # may become a buffer!
    has_buffer = "cache" in dict(m2.named_buffers())
    print(f"\n  m2.cache = tensor: in buffers? {has_buffer}")
    print("  -> In torch >= 2.3, Tensors in __setattr__ auto-register as buffers")
    print("  -> Use register_buffer() explicitly to control persistence")
    print()


# ============ 1d. Backward hooks ============
def exp_backward():
    print("=" * 60)
    print("1d. Backward hooks: capture grad_input & grad_output")
    print("=" * 60)

    model = nn.Linear(4, 4)
    model.weight.data.fill_(1.0)
    model.bias.data.fill_(0.0)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)

    grad_inputs = []
    grad_outputs = []

    def bw_hook(module, grad_input, grad_output):
        grad_inputs.append(grad_input[0].clone())
        grad_outputs.append(grad_output[0].clone())

    model.register_full_backward_hook(bw_hook)
    y = model(x).sum()
    y.backward()

    print(f"  grad_output: {grad_outputs[0]}")
    print(f"  grad_input:  {grad_inputs[0]}")
    print(f"  x.grad:      {x.grad}")
    print(f"  match:       {torch.allclose(x.grad, grad_inputs[0])}")

    # DEBUG: check if backward hook fires
    # If grad_input has NaN: check forward for NaN → use torch.autograd.detect_anomaly()
    # If grad_input is all zeros: check if grad_output is zero (loss doesn't depend on this module)
    nonzeros = (grad_outputs[0] != 0).any().item()
    print(f"  has non-zero grad_output: {nonzeros}")
    print("  -> use torch.autograd.detect_anomaly() to find NaN-grad sources")
    print()


# ============ 1e. Hook chain: inspect all hooks ============
def exp_hook_chain():
    print("=" * 60)
    print("1e. Hook chain inspection (debug tool)")
    print("=" * 60)

    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
    model.register_forward_pre_hook(lambda m, a: None)
    model[0].register_forward_hook(lambda m, a, o: None)
    model[2].register_full_backward_hook(lambda m, gi, go: None)

    print("  All registered hooks:")
    for name, mod in model.named_modules():
        if name == "":
            name = "root"
        hooks_info = []
        if mod._forward_pre_hooks:
            hooks_info.append(f"fwd_pre={len(mod._forward_pre_hooks)}")
        if mod._forward_hooks:
            hooks_info.append(f"fwd={len(mod._forward_hooks)}")
        if mod._backward_hooks:
            hooks_info.append(f"bwd={len(mod._backward_hooks)}")
        if hooks_info:
            print(f"    {name}: {', '.join(hooks_info)}")

    # Check if torch.compile intercepts hooks
    print(f"\n  root._compiled_call_impl: {model._compiled_call_impl}")
    print(
        "  -> if not None, torch.compile is active, hooks may go through compiled path"
    )
    print()


# ============ 2. Parameter/Buffer registration via __setattr__ ============
def exp_param():
    print("=" * 60)
    print("2. Parameter/Buffer auto-detection via __setattr__")
    print("=" * 60)

    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            # Parameter: automatically goes to _parameters
            self.weight = nn.Parameter(torch.randn(3, 3))
            # Buffer: use register_buffer if persistent matters
            self.register_buffer("running_mean", torch.zeros(3))
            self.register_buffer("counter", torch.tensor(0), persistent=False)
            # Plain int: goes to __dict__
            self.extra_info = 42
            # Submodule: goes to _modules
            self.sub = nn.Linear(3, 3)

    m = MyModule()

    print(f"  _parameters: {list(m._parameters.keys())}")
    print(f"  _buffers:    {list(m._buffers.keys())}")
    print(f"  _modules:    {list(m._modules.keys())}")
    print(
        f"  __dict__:    {[k for k in m.__dict__ if k not in ('_parameters', '_buffers', '_modules', '_non_persistent_buffers_set', 'training')]}"
    )

    # Non-persistent buffer set
    print(f"\n  non_persistent_buffers: {m._non_persistent_buffers_set}")

    # Verify types
    print(f"\n  type(weight):       {type(m.weight)}")
    print(f"  type(running_mean): {type(m.running_mean)}")
    print(f"  type(extra_info):   {type(m.extra_info)}")
    print("  -> Parameter/Buffer/Module auto-detected by __setattr__")
    print()


# ============ 3. state_dict serialization ============
def exp_state():
    print("=" * 60)
    print("3. state_dict: parameter & persistent buffer serialization")
    print("=" * 60)

    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 3)
            self.register_buffer("mean", torch.randn(3))
            self.register_buffer("step", torch.tensor(100), persistent=False)

    m = MyModule()
    sd = m.state_dict()

    print(f"  state_dict keys: {list(sd.keys())}")
    print(f"  fc.weight shape: {sd['fc.weight'].shape}")
    print(f"  fc.bias shape:   {sd['fc.bias'].shape}")
    print(f"  mean shape:      {sd['mean'].shape}")
    print(f"  'step' in sd:    {'step' in sd}")
    print("  -> persistent=False buffers excluded from state_dict")
    print("  -> detach() used in state_dict to break autograd graph")

    # load_state_dict with _IncompatibleKeys debug
    m3 = nn.Linear(4, 3)
    result = m3.load_state_dict(
        {"weight": torch.randn(3, 4), "bias": torch.zeros(3)}, strict=False
    )
    print(f"\n  strict=False result:")
    print(f"    missing_keys:    {result.missing_keys}")
    print(f"    unexpected_keys: {result.unexpected_keys}")
    print("  -> missing: keys in model but not in checkpoint")
    print("  -> unexpected: keys in checkpoint but not in model")

    # DEBUG: how to find shape mismatches
    m4 = nn.Linear(4, 3)
    try:
        m4.load_state_dict({"weight": torch.randn(2, 2)})  # shape mismatch!
    except RuntimeError as e:
        print(f"\n  Shape mismatch error:")
        print(f"    {(str(e)[:120])}")
    print("  -> Fix: compare shapes before load with:")
    print("     for k, v in ckpt.items(): print(k, v.shape)")
    print()


# ============ 4. _apply: batch type conversion ============
def exp_apply():
    print("=" * 60)
    print("4. _apply: batch type/device conversion")
    print("=" * 60)

    m = nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3))

    print(f"  Before _apply:")
    print(f"    Linear.weight device: {m[0].weight.device}")
    print(f"    Linear.weight dtype:  {m[0].weight.dtype}")
    print(f"    BN.num_batches_tracked dtype: {m[1].num_batches_tracked.dtype}")

    # .half() calls _apply internally
    m.half()

    print(f"  After half():")
    print(f"    Linear.weight dtype:  {m[0].weight.dtype}")
    print(f"    BN.running_mean dtype: {m[1].running_mean.dtype}")
    print(f"    BN.num_batches_tracked dtype: {m[1].num_batches_tracked.dtype}")
    print("    -> is_floating_point() guards integer-type buffers from conversion")

    # Verify object identity preserved (important for optimizer)
    m2 = nn.Linear(3, 3)
    weight_before = m2.weight
    m2._apply(lambda t: t * 2.0)
    weight_after = m2.weight
    print(f"\n  Object identity preserved: {weight_before is weight_after}")
    print("  -> optimizer references remain valid after _apply")
    print()


# ============ 5. Parameter deduplication ============
def exp_dedup():
    print("=" * 60)
    print("5. Parameter deduplication: shared weights appear once")
    print("=" * 60)

    # Shared embedding (weight tying)
    class SharedModel(nn.Module):
        def __init__(self):
            super().__init__()
            shared = nn.Parameter(torch.randn(10, 64))
            self.embed = nn.Embedding(10, 64)
            self.embed.weight = shared  # shared
            self.head = nn.Linear(64, 10)
            self.head.weight = shared.T  # different tensor, not shared
            # Actually share:
            self.embed2 = nn.Embedding(10, 64)
            self.embed2.weight = shared  # truly shared

    m = SharedModel()
    params = list(m.parameters())
    named = list(m.named_parameters())

    print(
        f"  Total parameter tensors (counting shared): {sum(1 for _ in m._parameters.values() if _ is not None)}"
    )
    print(f"  parameters() count (deduplicated): {len(params)}")
    print(f"  named_parameters():")
    for name, p in named:
        print(f"    {name:30s} shape={list(p.shape)} id={id(p)}")

    # Verify embed.weight and embed2.weight are the same object
    print(f"\n  embed.weight is embed2.weight: {m.embed.weight is m.embed2.weight}")
    print("  -> _named_members uses set() to deduplicate shared parameters")
    print()


EXPERIMENTS = {
    "hooks": exp_hooks,
    "always_call": exp_always_call,
    "conflict": exp_conflict,
    "backward": exp_backward,
    "hook_chain": exp_hook_chain,
    "param": exp_param,
    "state": exp_state,
    "apply": exp_apply,
    "dedup": exp_dedup,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[module demo] DONE")


if __name__ == "__main__":
    main()
