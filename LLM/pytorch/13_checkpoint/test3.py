"""Checkpoint 源码分析: RNG state 序列化, re-forward trace, SavedVariable。

使用工具: torch.random.get_rng_state / torch.cuda.get_rng_state /
         backward hook / ctx.saved_tensors

运行:
  python test3.py                 # 全链路分析
  python test3.py rng_compare     # RNG 状态保存/恢复对比
  python test3.py reforward_trace # re-forward 过程追踪
  python test3.py saved_peek      # saved_tensors 内部探查

参考源码:
  torch/utils/checkpoint.py       — CheckpointFunction
  torch/csrc/autograd/saved_variable.cpp — SavedVariable
"""

import sys
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


# ============ 1. RNG 状态保存/恢复对比 ============
def exp_rng_compare():
    """对比 RNG 状态在 checkpoint 前后是否一致。"""
    print("=" * 60)
    print("1. RNG State: 保存 vs 恢复 的完整对比")
    print("=" * 60)

    torch.manual_seed(42)

    class DropoutBlock(nn.Module):
        def forward(self, x):
            return nn.functional.dropout(x, p=0.5, training=True)

    model = DropoutBlock()

    # 用 record_function 抓 RNG state
    rng_before_forward = torch.get_rng_state().clone()
    x = torch.ones(10)

    fwd_rng = torch.get_rng_state().clone()
    y = model(x)
    rng_after_forward = torch.get_rng_state().clone()

    # Checkpoint 版
    torch.manual_seed(42)
    rng_before_ckpt = torch.get_rng_state().clone()
    y_ckpt = checkpoint(model, x, use_reentrant=False, preserve_rng_state=True)
    rng_after_ckpt = torch.get_rng_state().clone()

    print(f"  Forward output match: {torch.allclose(y, y_ckpt)}")

    # RNG 应该在 preserve_rng_state=True 时完全一致
    print(f"\n  RNG comparison:")
    print(f"    Forward before: {rng_before_forward[:8].tolist()}")
    print(f"    Forward after:  {rng_after_forward[:8].tolist()}")
    print(f"    Ckpt    before: {rng_before_ckpt[:8].tolist()}")
    print(f"    Ckpt    after:  {rng_after_ckpt[:8].tolist()}")
    print(f"    Forward == Ckpt: {torch.equal(rng_after_forward, rng_after_ckpt)}")

    if torch.cuda.is_available():
        # CUDA RNG 同样会被保存/恢复
        cuda_rng_init = torch.cuda.get_rng_state().clone()
        cuda_x = torch.ones(10, device="cuda")
        cuda_y = checkpoint(
            lambda x: nn.functional.dropout(x, p=0.5, training=True),
            cuda_x,
            use_reentrant=False,
            preserve_rng_state=True,
        )
        cuda_rng_after = torch.cuda.get_rng_state().clone()
        print(f"\n    CUDA RNG preserved: {torch.equal(cuda_rng_init, cuda_rng_after)}")

    print(f"\n  RNG 恢复机制 (checkpoint.py:246-256, 298-307):")
    print(f"  forward: ctx.fwd_cpu_state = torch.get_rng_state()")
    print(f"           ctx.fwd_device_states = get_device_states(*args)")
    print(f"  backward: torch.set_rng_state(ctx.fwd_cpu_state)")
    print(f"            set_device_states(ctx.fwd_devices, ...)")
    print()


# ============ 2. Re-forward 过程追踪 ============
def exp_reforward_trace():
    """追踪 checkpoint backward 中的 re-forward 过程。"""
    print("=" * 60)
    print("2. Re-forward Trace: backward 中如何重新运行 forward")
    print("=" * 60)

    class TraceBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8)
            self.fwd_count = 0

        def forward(self, x):
            self.fwd_count += 1
            return self.linear(x).relu()

    model = TraceBlock()
    x = torch.randn(4, 8, requires_grad=True)

    # Checkpoint: forward runs once, backward runs again
    y = checkpoint(model, x, use_reentrant=False)
    print(f"  After forward: fwd_count = {model.fwd_count}  (1 call)")

    loss = y.sum()
    loss.backward()
    print(f"  After backward: fwd_count = {model.fwd_count}  (2 calls — re-forward!)")

    # Reference: without checkpoint
    model2 = TraceBlock()
    x2 = x.detach().clone().requires_grad_(True)
    y2 = model2(x2)
    loss2 = y2.sum()
    loss2.backward()
    print(
        f"  Without checkpoint: fwd_count = {model2.fwd_count}  (1 call, no re-forward)"
    )

    print(f"\n  Re-forward 关键步骤 (checkpoint.py:278-331):")
    print(f"  1. detach_variable(inputs) — 切断旧计算图")
    print(f"  2. torch.set_rng_state()   — 恢复 RNG")
    print(f"  3. torch.enable_grad()     — 启用梯度追踪")
    print(f"  4. ctx.run_function(*inputs)— 重新运行 forward")
    print(f"  5. torch.autograd.backward(outputs, grads) — 新图 backward")
    print(f"  6. 收集 inp.grad")
    print()


# ============ 3. Saved Variables 内部探查 ============
def exp_saved_peek():
    """探查 checkpoint 中 ctx.save_for_backward 的内容。"""
    print("=" * 60)
    print("3. Saved Tensors: 查看 checkpoint 保存了什么")
    print("=" * 60)

    class MyCheckpoint(torch.autograd.Function):
        @staticmethod
        def forward(ctx, fn, *args):
            ctx.fn = fn
            tensor_args = [a for a in args if torch.is_tensor(a)]
            ctx.save_for_backward(*tensor_args)
            ctx.non_tensor_count = len(args) - len(tensor_args)
            with torch.no_grad():
                return fn(*args)

        @staticmethod
        def backward(ctx, *grads):
            saved = ctx.saved_tensors
            print(f"    saved_tensors count: {len(saved)}")
            for i, st in enumerate(saved):
                print(
                    f"    [{i}] shape={list(st.shape)} dtype={st.dtype} "
                    f"requires_grad={st.requires_grad}"
                )

            # Reconstruct inputs
            tensor_idx = 0
            inputs = []
            for _ in range(len(saved) + ctx.non_tensor_count):
                if tensor_idx < len(saved):
                    t = saved[tensor_idx].detach().requires_grad_(True)
                    inputs.append(t)
                    tensor_idx += 1
            return (None,) + tuple(inp.grad for inp in inputs)

    def block(x):
        return x * 2 + 1

    x = torch.tensor([2.0, 3.0], requires_grad=True)
    y = MyCheckpoint.apply(block, x)
    loss = y.sum()
    loss.backward()

    print(f"\n  源码对应 (checkpoint.py:260-271):")
    print(f"  ctx.inputs = []")
    print(f"  for i, arg in enumerate(args):")
    print(f"      if torch.is_tensor(arg):")
    print(f"          tensor_inputs.append(arg)")
    print(f"          ctx.inputs.append(None)  # placeholder")
    print(f"      else:")
    print(f"          ctx.inputs.append(arg)")
    print(f"  ctx.save_for_backward(*tensor_inputs)")
    print()


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else []
    if exps:
        for name in exps:
            globals()[f"exp_{name}"]()
    else:
        exp_rng_compare()
        exp_reforward_trace()
        exp_saved_peek()

    print("[Checkpoint source analysis] DONE")


if __name__ == "__main__":
    main()
